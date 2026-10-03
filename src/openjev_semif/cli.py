"""Serve, score, benchmark, and evaluate from one resident model."""
from __future__ import annotations
import argparse
import json
import os
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
    client = sub.add_parser("client", help="Call a running HTTP service without loading a model")
    client_sub = client.add_subparsers(dest="client_command", required=True)
    for name in ("health", "score", "systemone"):
        x = client_sub.add_parser(name)
        x.add_argument("--url", default=os.environ.get("OPENJEV_URL", "http://127.0.0.1:8000"))
        x.add_argument("--api-key", default=os.environ.get("OPENJEV_API_KEY"))
        x.add_argument("--timeout", type=float, default=30.0)
        if name == "score":
            x.add_argument("--state", required=True)
            x.add_argument("--question", required=True)
            x.add_argument("--option", action="append", required=True)
            x.add_argument("--scorer", choices=("semif", "likelihood"))
            x.add_argument("--temperature", type=float)
        elif name == "systemone":
            x.add_argument("--input", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "client":
        from .client import OpenJEVClient
        with OpenJEVClient(args.url, api_key=args.api_key, timeout=args.timeout) as client:
            if args.client_command == "health":
                result = client.health()
            elif args.client_command == "score":
                request = {"state": args.state, "question": args.question,
                           "options": args.option}
                if args.scorer: request["scorer"] = args.scorer
                if args.temperature is not None: request["temperature"] = args.temperature
                result = client.score(request)
            else:
                result = client.systemone(json.loads(args.input.read_text()))
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
        return
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
