from __future__ import annotations

def create_backend(settings):
    if settings.backend != "torch":
        raise ValueError(f"backend {settings.backend!r} is unavailable; supported: torch")
    from .torch_hf import TorchHFBackend
    return TorchHFBackend(settings)
