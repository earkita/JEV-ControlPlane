"""Serve, score, benchmark, and evaluate from one resident model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .config import Settings


def parser():
    p = argparse.ArgumentParser(prog="openjev-semif")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("serve", "score", "bench", "eval"):
        x = sub.add_parser(name)
        x.add_argument("--config", type=Path)
        for flag in ("model", "revision", "backend", "device", "dtype", "scorer", "host"):
            x.add_argument("--" + flag)
        for flag in ("port", "max-context"):
            x.add_argument("--" + flag, type=int)
        x.add_argument("--temperature", type=float)
        if name == "score":
            x.add_argument("--state", required=True)
            x.add_argument("--question", required=True)
            x.add_argument("--option", action="append", required=True)
        if name in ("bench", "eval"):
            x.add_argument("--input", type=Path, required=True)
            if name == "bench": x.add_argument("--repeats", type=int, default=3)
            x.add_argument("--output", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    values = json.loads(args.config.read_text()) if args.config else {}
    values.update({k: v for k, v in vars(args).items()
                   if k in Settings.__dataclass_fields__ and v is not None})
    settings = Settings.from_env(**values)
    if args.command == "serve":
        import uvicorn
        from .api.server import create_app
        uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1)
        return
    from .backends.registry import create_backend
    model = create_backend(settings)
    from .scoring.calibration import effective_temperature
    temperature = effective_temperature(settings, model)
    if args.command == "score":
        from .scoring.base import Decision, Option
        from .scoring.semif import SemIfScorer
        from .scoring.likelihood import LikelihoodScorer
        decision = Decision(args.state, args.question, [Option(x, x) for x in args.option])
        scorer = SemIfScorer(model) if settings.scorer == "semif" else LikelihoodScorer(model)
        print(json.dumps(scorer.score(decision, temperature).as_dict(), indent=2))
        return
    from .benchmark import bench, evaluate
    output = bench(model, args.input, temperature, args.repeats) if args.command == "bench" else evaluate(model, args.input, settings.scorer, temperature)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))

if __name__ == "__main__": main()
