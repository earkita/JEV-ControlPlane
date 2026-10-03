"""Mapping between TypeSafe-style questions and the shared decision contract."""
from __future__ import annotations
import json
from .schemas import ChoiceQuestion, ScoreQuestion, NoulQuestion
from ..scoring.base import Decision, Option


def render(value):
    if value is None: return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def decision_for(state, question):
    instructions = render(question.instructions)
    if isinstance(question, ChoiceQuestion):
        return Decision(state, instructions or "Choose the best option.",
                        [Option(key, render(desc) or key) for key, desc in question.criteria.items()])
    if isinstance(question, ScoreQuestion):
        return Decision(state, instructions or "Choose the best rating.",
                        [Option(str(i), render(desc) or str(i)) for i, desc in enumerate(question.criteria)])
    assert isinstance(question, NoulQuestion)
    criteria = question.criteria or {}
    return Decision(state, instructions or "Is this true?",
                    [Option("true", render(criteria.get("true")) or "Yes"),
                     Option("false", render(criteria.get("false")) or "No")])


def answer_for(question, result):
    probs = {x["option"]: x["probability"] for x in result.options}
    common = {"probabilities": probs, "confidence": result.confidence,
              "timings": result.timings, "scorer": result.scorer,
              "model_revision": result.model_revision, "tokenizer_revision": result.tokenizer_revision,
              "prompt_hash": result.prompt_hash, "input_tokens": result.input_tokens,
              "temperature": result.temperature, "truncated": result.truncated, "mode": result.mode}
    if isinstance(question, ChoiceQuestion):
        return {"type": "choice", "choice": result.best, **common}
    if isinstance(question, ScoreQuestion):
        return {"type": "score", "score": sum(i * p for i, p in enumerate(probs.values())),
                "legend": {str(i): c for i, c in enumerate(question.criteria)}, **common}
    return {"type": "noul", "noul": probs["true"], **common}
