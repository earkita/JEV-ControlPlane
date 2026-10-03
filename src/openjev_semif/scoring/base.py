"""Common request and result objects for distinct scoring semantics."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Option:
    id: str
    description: str

@dataclass(frozen=True)
class Decision:
    state: Any
    question: str
    options: list[Option]

@dataclass
class Result:
    best: str
    best_index: int
    options: list[dict]
    confidence: float
    scorer: str
    model: str
    model_revision: str
    tokenizer_revision: str
    prompt_hash: str
    prompt_version: str
    input_tokens: int
    truncated: bool
    temperature: float
    timings: dict[str, float]
    mode: str = "direct"

    def as_dict(self): return self.__dict__
