# Agent work in this repository

- Work in `/home/ea/ai/JEV-ControlPlane` when this workspace is available. Keep reference clones outside this repository.
- Preserve the distinction between likelihood (`P(option text | context)`) and SemIf (softmax of final-position answer-letter logits). Never present one as the other.
- Keep model loading in the backend/service lifecycle. A server worker should load one model once. HTTP client paths must never create a model backend.
- Preserve exact token-boundary checks, prompt hashes, model/tokenizer revision metadata, and explicit context-limit failures. Never silently truncate.
- Shared mode must actually prefill a native KV prefix once. Compare its probabilities with direct mode on the target model after cache changes.
- Run `pytest -q` for CPU tests. Run `RUN_MODEL_INTEGRATION=1 pytest -q` only on a CUDA host with cached Qwen3.5-4B weights.
- Benchmarks must state the exact model revision, hardware, dtype, prompt size, repetitions, and request versus decision throughput. Do not present synthetic sanity fixtures as quality evaluations.
- Review upstream licenses and record attribution before adapting code. Do not commit model weights, caches, third-party datasets, secrets, or reference repositories.
