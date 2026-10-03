"""Same-model benchmark and labelled evaluation runners."""
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
    return Decision(state, row["question"], [
        Option(str(x.get("id", x["description"])), x["description"])
        if isinstance(x, dict) else Option(x, x) for x in row["options"]
    ])


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    low = int(position)
    high = min(low+1, len(ordered)-1)
    return ordered[low] + (ordered[high]-ordered[low])*(position-low)


def _metrics(results, latencies, elapsed, model):
    return {
        "requests_per_s": len(latencies)/elapsed,
        "decisions_per_s": len(results)/elapsed,
        "latency_p50_s": statistics.median(latencies),
        "latency_p95_s": _percentile(latencies, 0.95),
        "prefill_s": sum(r.timings.get("prefill_s", 0) for r in results),
        "shared_prefill_s": sum(r.timings.get("shared_prefill_s", 0) for r in results
                                if r.timings.get("shared_question_index") == 0),
        "branch_s": sum(r.timings.get("branch_s", 0) for r in results),
        "gpu_peak_memory_bytes": model.gpu_memory_bytes() if hasattr(model, "gpu_memory_bytes") else None,
    }


def bench(model, path, temperature=1.0, repeats=3):
    if repeats < 1: raise ValueError("repeats must be positive")
    data = json.loads(Path(path).read_text())
    state = data["state"] * data.get("state_repeat", 1) if isinstance(data["state"], str) else data["state"]
    decisions = [_decision(state, row) for row in data["questions"]]
    if len(decisions) < 2: raise ValueError("benchmark requires at least two questions")
    report = {"model": model.model_name, "revision": model.model_revision,
              "input": str(path), "repeats": repeats, "state_characters": len(str(state)),
              "environment": getattr(model, "metadata", {}), "runs": {}}
    for name in ("likelihood", "semif_direct", "semif_shared"):
        if hasattr(model, "torch") and str(model.device).startswith("cuda"):
            model.torch.cuda.reset_peak_memory_stats(model.device)
        results, latencies = [], []
        mark = time.perf_counter()
        for _ in range(repeats):
            if name == "semif_shared":
                request_start = time.perf_counter()
                batch = score_shared(model, decisions, temperature)
                latencies.append(time.perf_counter()-request_start)
            else:
                engine = LikelihoodScorer(model) if name == "likelihood" else SemIfScorer(model)
                batch = []
                for decision in decisions:
                    request_start = time.perf_counter()
                    batch.append(engine.score(decision, temperature))
                    latencies.append(time.perf_counter()-request_start)
            results.extend(batch)
        elapsed = time.perf_counter()-mark
        report["runs"][name] = {**_metrics(results, latencies, elapsed, model),
                                "choices": [r.best for r in results[:len(decisions)]],
                                "wall_s": elapsed}
    report["input_tokens_first_question"] = results[0].input_tokens
    report["shared_prefix_tokens"] = results[0].timings.get("shared_prefix_tokens")
    direct = report["runs"]["semif_direct"]["choices"]
    shared = report["runs"]["semif_shared"]["choices"]
    report["shared_direct_agreement"] = sum(a == b for a, b in zip(direct, shared)) / len(direct)
    return report


def evaluate(model, path, scorer="semif", temperature=1.0):
    engine = SemIfScorer(model) if scorer == "semif" else LikelihoodScorer(model)
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows: raise ValueError("evaluation file is empty")
    results = [engine.score(_decision(row["state"], row), temperature) for row in rows]
    correct = sum(r.best == row["label"] for r, row in zip(results, rows))
    return {"model": model.model_name, "revision": model.model_revision, "scorer": scorer,
            "accuracy": correct/len(rows), "count": len(rows),
            "predictions": [{"best": r.best, "label": row["label"], "prompt_hash": r.prompt_hash}
                            for r, row in zip(results, rows)]}
