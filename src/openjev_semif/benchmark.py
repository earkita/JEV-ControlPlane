"""Small local benchmark and labelled evaluation runners."""
from __future__ import annotations
import json
import statistics
import time
from pathlib import Path
from .scoring.base import Decision, Option
from .scoring.semif import SemIfScorer
from .scoring.likelihood import LikelihoodScorer
from .runtime.shared_state import score_shared


def _decision(state, row):
    return Decision(state, row["question"], [Option(str(x.get("id", x.get("description"))), x["description"])
        if isinstance(x, dict) else Option(x, x) for x in row["options"]])


def _metrics(results, elapsed, model):
    latencies = sorted(r.timings["total_s"] for r in results)
    p95 = latencies[min(len(latencies)-1, int(0.95 * (len(latencies)-1)))]
    return {"requests_per_s": len(results)/elapsed, "decisions_per_s": len(results)/elapsed,
            "latency_p50_s": statistics.median(latencies), "latency_p95_s": p95,
            "prefill_s": sum(r.timings.get("prefill_s", 0) for r in results),
            "shared_prefill_s": results[0].timings.get("shared_prefill_s", 0),
            "gpu_peak_memory_bytes": model.gpu_memory_bytes() if hasattr(model, "gpu_memory_bytes") else None}


def bench(model, path, temperature=1.0):
    data = json.loads(Path(path).read_text())
    decisions = [_decision(data["state"], row) for row in data["questions"]]
    if len(decisions) < 2: raise ValueError("benchmark requires at least two questions")
    report = {"model": model.model_name, "revision": model.model_revision, "input": str(path), "runs": {}}
    for name, runner in (
        ("likelihood", lambda: [LikelihoodScorer(model).score(d, temperature) for d in decisions]),
        ("semif_direct", lambda: [SemIfScorer(model).score(d, temperature) for d in decisions]),
        ("semif_shared", lambda: score_shared(model, decisions, temperature)),
    ):
        mark = time.perf_counter()
        results = runner()
        elapsed = time.perf_counter()-mark
        report["runs"][name] = {**_metrics(results, elapsed, model), "choices": [r.best for r in results],
                                "wall_s": elapsed}
    return report


def evaluate(model, path, scorer="semif", temperature=1.0):
    engine = SemIfScorer(model) if scorer == "semif" else LikelihoodScorer(model)
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    results = [engine.score(_decision(row["state"], row), temperature) for row in rows]
    correct = sum(r.best == row["label"] for r, row in zip(results, rows))
    return {"model": model.model_name, "revision": model.model_revision, "scorer": scorer,
            "accuracy": correct/len(rows), "count": len(rows),
            "predictions": [{"best": r.best, "label": row["label"], "prompt_hash": r.prompt_hash}
                            for r, row in zip(results, rows)]}
