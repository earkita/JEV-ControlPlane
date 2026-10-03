"""Native shared KV prefix for multiple decisions over the same state."""
from __future__ import annotations
import time
from ..scoring.semif import SemIfScorer


def score_shared(backend, decisions, temperature=1.0):
    if len(decisions) < 2: raise ValueError("shared mode requires at least two questions")
    if any(d.state != decisions[0].state for d in decisions):
        raise ValueError("shared mode requires the same state")
    scorer = SemIfScorer(backend)
    started = time.perf_counter()
    prepared = [scorer.prepare(d) for d in decisions]
    sequences = [p[0][1] for p in prepared]
    prefix_length = 0
    for tokens in zip(*sequences):
        if len(set(tokens)) != 1: break
        prefix_length += 1
    prefix_length = min(prefix_length, min(map(len, sequences)) - 1)
    marker = prepared[0][0][0].find("State:\n")
    if marker < 0: raise ValueError("prompt lacks state marker")
    state_end = prepared[0][0][0].find("\n\nQuestion:", marker)
    if state_end < 0: raise ValueError("prompt lacks state boundary")
    state_tokens = len(backend.tokenizer.encode(prepared[0][0][0][:state_end], add_special_tokens=False)) - 1
    if prefix_length < state_tokens or prefix_length < 1:
        raise ValueError("tokenized state is not a safe shared prefix")
    mark = time.perf_counter()
    cache = backend.prefill(sequences[0][:prefix_length])
    shared_prefill = time.perf_counter()-mark
    results = []
    for index, (decision, (encoded, tokenization)) in enumerate(zip(decisions, prepared)):
        suffix = encoded[1][prefix_length:]
        mark = time.perf_counter()
        logits = backend.get_last_logits(cache, suffix)
        branch = time.perf_counter()-mark
        results.append(scorer.finish(decision, encoded, logits, temperature,
            {"tokenization_s": tokenization, "prefill_s": 0.0, "shared_prefill_s": shared_prefill, "shared_prefix_tokens": prefix_length,
             "branch_s": branch, "shared_question_index": index, "scoring_s": branch,
             "total_s": time.perf_counter()-started}, mode="shared"))
    return results
