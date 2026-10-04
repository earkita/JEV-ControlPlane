"""Runtime settings shared by CLI and server."""
from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    model: str = "Qwen/Qwen3.5-4B"
    revision: str | None = None
    backend: str = "torch"
    device: str = "cuda"
    dtype: str = "bfloat16"
    scorer: str = "semif"
    max_context: int = 4096
    temperature: float = 1.0
    calibration_profile: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self):
        if self.backend != "torch": raise ValueError("backend must be torch")
        if self.scorer not in ("semif", "likelihood"):
            raise ValueError("scorer must be semif or likelihood")
        if self.max_context < 2: raise ValueError("max_context must be >= 2")
        if self.temperature <= 0: raise ValueError("temperature must be positive")

    @classmethod
    def from_env(cls, config=None, **overrides):
        values = dict(config or {})
        values.update({field: value for field in cls.__dataclass_fields__
                       if (value := os.environ.get(field.upper())) is not None})
        values.update({k: v for k, v in overrides.items() if v is not None})
        for key in ("max_context", "port"):
            if key in values: values[key] = int(values[key])
        if "temperature" in values: values["temperature"] = float(values["temperature"])
        if "model" in values: values["model"] = os.path.expanduser(values["model"])
        result = cls(**values)
        return result
