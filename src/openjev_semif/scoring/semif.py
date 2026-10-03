"""Read only final-position option-letter logits; never generate text."""
from __future__ import annotations
import time
from ..prompting.semif import encode, prompt, PROMPT_VERSION
from .base import Decision, Result
from .calibration import probabilities, confidence

class SemIfScorer:
    name = "semif"
    def __init__(self, backend): self.backend = backend

    def prepare(self, decision: Decision):
        started = time.perf_counter()
        text = prompt(self.backend.tokenizer, decision.state, decision.question,
                      [x.description for x in decision.options])
        ids, slots, digest = encode(self.backend.tokenizer, text, len(decision.options), self.backend.max_context)
        return (text, ids, slots, digest), time.perf_counter() - started

    def finish(self, decision, prepared, logits, temperature, timings, mode="direct"):
        _, ids, slots, digest = prepared
        selected = [float(logits[token]) for token in slots]
        probs = probabilities(selected, temperature)
        best = max(range(len(probs)), key=probs.__getitem__)
        return Result(
            best=decision.options[best].id, best_index=best,
            options=[{"option": o.id, "description": o.description, "raw_score": raw,
                      "probability": p} for o, raw, p in zip(decision.options, selected, probs)],
            confidence=confidence(probs), scorer=self.name, model=self.backend.model_name,
            model_revision=self.backend.model_revision,
            tokenizer_revision=self.backend.tokenizer_revision, prompt_hash=digest,
            prompt_version=PROMPT_VERSION, input_tokens=len(ids), truncated=False,
            temperature=temperature, timings=timings, mode=mode,
        )

    def score(self, decision: Decision, temperature: float = 1.0):
        started = time.perf_counter()
        prepared, tokenization = self.prepare(decision)
        mark = time.perf_counter()
        logits = self.backend.forward(prepared[1])
        prefill = time.perf_counter() - mark
        return self.finish(decision, prepared, logits, temperature,
                           {"tokenization_s": tokenization, "prefill_s": prefill,
                            "scoring_s": 0.0, "total_s": time.perf_counter()-started})
