"""P(option text | state and question), distinct from SemIf's slot logits."""
from __future__ import annotations
import hashlib
import time
from ..prompting.semif import state_text
from .base import Decision, Result
from .calibration import probabilities, confidence

class LikelihoodScorer:
    name = "likelihood"
    def __init__(self, backend): self.backend = backend

    def score(self, decision: Decision, temperature: float = 1.0):
        started = time.perf_counter()
        text = f"State:\n{state_text(decision.state)}\n\nQuestion:\n{decision.question}\n\nAnswer:\n"
        tok = self.backend.tokenizer
        prefix = tok.encode(text, add_special_tokens=False)
        if not prefix: raise ValueError("empty likelihood context")
        continuations = []
        for option in decision.options:
            ids = tok.encode(option.description, add_special_tokens=False)
            if not ids: raise ValueError("empty option tokens")
            if tok.encode(text + option.description, add_special_tokens=False) != prefix + ids:
                raise ValueError("option text changes tokenization at answer boundary")
            if len(prefix) + len(ids) > self.backend.max_context:
                raise ValueError("likelihood prompt exceeds max_context; truncation disabled")
            continuations.append(ids)
        tokenization = time.perf_counter()-started
        mark = time.perf_counter()
        cache = self.backend.prefill(prefix)
        prefill = time.perf_counter()-mark
        mark = time.perf_counter()
        raw = self.backend.score_continuations(cache, continuations)
        scoring = time.perf_counter()-mark
        probs = probabilities(raw, temperature)
        best = max(range(len(probs)), key=probs.__getitem__)
        return Result(
            best=decision.options[best].id, best_index=best,
            options=[{"option": o.id, "description": o.description, "raw_score": score,
                      "probability": p, "n_tokens": len(ids)}
                     for o, score, p, ids in zip(decision.options, raw, probs, continuations)],
            confidence=confidence(probs), scorer=self.name, model=self.backend.model_name,
            model_revision=self.backend.model_revision,
            tokenizer_revision=self.backend.tokenizer_revision,
            prompt_hash=hashlib.sha256(text.encode()).hexdigest(), prompt_version="likelihood-v1",
            input_tokens=len(prefix), truncated=False, temperature=temperature,
            timings={"tokenization_s": tokenization, "prefill_s": prefill,
                     "scoring_s": scoring, "total_s": time.perf_counter()-started},
        )
