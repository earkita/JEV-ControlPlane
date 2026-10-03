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


def effective_temperature(settings, backend) -> float:
    """Use a profile only for the exact model revision and configured scorer."""
    if not settings.calibration_profile:
        return settings.temperature
    profile = CalibrationProfile.load(settings.calibration_profile)
    if (profile.model, profile.revision, profile.scorer) != (
        backend.model_name, backend.model_revision, settings.scorer
    ):
        raise ValueError("calibration profile does not match model revision and scorer")
    return profile.temperature


def fit_temperature(rows: list[tuple[list[float], int]]) -> float:
    """Fit a positive temperature by minimizing labelled negative log likelihood."""
    if not rows:
        raise ValueError("at least one labelled row is required")
    for logits, label in rows:
        if label < 0 or label >= len(logits):
            raise ValueError("label index is out of range")
        probabilities(logits)
    def loss(log_t: float) -> float:
        temp = math.exp(log_t)
        total = 0.0
        for logits, label in rows:
            scaled = [x / temp for x in logits]
            peak = max(scaled)
            total += peak + math.log(sum(math.exp(x-peak) for x in scaled)) - scaled[label]
        return total / len(rows)
    lo, hi = math.log(0.05), math.log(20.0)
    for _ in range(80):
        left, right = lo + (hi-lo)/3, hi - (hi-lo)/3
        if loss(left) <= loss(right): hi = right
        else: lo = left
    return math.exp((lo+hi)/2)
