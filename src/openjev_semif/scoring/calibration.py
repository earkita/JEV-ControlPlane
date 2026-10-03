"""Temperature scaling; profiles may be saved per model and workload."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path


def probabilities(scores: list[float], temperature: float = 1.0) -> list[float]:
    if len(scores) < 2 or not all(math.isfinite(x) for x in scores):
        raise ValueError("at least two finite scores are required")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = [x / temperature for x in scores]
    peak = max(values)
    weights = [math.exp(x - peak) for x in values]
    total = sum(weights)
    return [x / total for x in weights]


def confidence(probs: list[float]) -> float:
    """Normalized entropy concentration; not empirical correctness probability."""
    return 1.0 + sum(p * math.log(p) for p in probs if p > 0) / math.log(len(probs))

@dataclass(frozen=True)
class CalibrationProfile:
    model: str
    revision: str
    workload: str
    scorer: str
    temperature: float

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        profile = cls(**json.loads(Path(path).read_text()))
        probabilities([0.0, 1.0], profile.temperature)
        return profile

    def save(self, path: str | Path) -> None:
        probabilities([0.0, 1.0], self.temperature)
        Path(path).write_text(json.dumps(self.__dict__, indent=2) + "\n")
