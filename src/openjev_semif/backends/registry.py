from __future__ import annotations

def create_backend(settings):
    if settings.backend == "llama_cpp":
        from .llama_cpp import LlamaCppBackend
        return LlamaCppBackend(settings)
    if settings.backend == "winnow":
        from .winnow import WinnowBackend
        return WinnowBackend(settings)
    if settings.backend != "torch":
        raise ValueError(f"backend {settings.backend!r} is unavailable; supported: torch, llama_cpp, winnow")
    from .torch_hf import TorchHFBackend
    return TorchHFBackend(settings)
