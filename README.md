# OpenJEV-SemIf

A single-model decision service with two explicit scoring methods. **SemIf** selects final-position letter logits for listed options; **likelihood** computes the conditional log probability of each option's text. Neither method generates an answer. The service loads one Hugging Face model at startup.

## Install

Python 3.12+ and a CUDA PyTorch build are required for the target GPU. In this repository:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[torch,test]'
```

For CUDA wheels, follow the [PyTorch installation selector](https://pytorch.org/get-started/locally/) if your environment does not already supply a compatible build.

## Run on RTX 3090

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/openjev-semif serve \
  --model Qwen/Qwen3.5-4B --backend torch --scorer semif \
  --device cuda --dtype bfloat16 --port 8000
```

The Hugging Face model revision is resolved to a commit before loading and is returned in responses. Set `--revision` to pin one explicitly. `MODEL`, `REVISION`, `BACKEND`, `DEVICE`, `DTYPE`, `SCORER`, `MAX_CONTEXT`, `TEMPERATURE`, `CALIBRATION_PROFILE`, `HOST`, and `PORT` are supported environment settings. `--config settings.json` may supply the same fields; CLI flags override config and environment values.

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Build failed after enabling a new MXFP4 kernel.",
  "questions": {
    "next_action": {
      "type": "choice", "instructions": "What should the agent do next?",
      "criteria": {
        "debug": "Inspect the failing kernel and logs",
        "rollback": "Disable the optimization",
        "retest": "Run the tests again"
      }
    }
  }
}'
```

`GET /health` reports load status, model, backend, device, dtype, scorer and revision. `POST /score` accepts `state`, `question`, `options`, optional `scorer` and `temperature`. Options may be strings or `{id, description}` objects. `/v1/systemone` accepts typed `choice`, `score`, and `noul` questions. Responses include option probabilities, raw scores (on `/score`), entropy concentration, timing, input token count, prompt hash, scorer and model/tokenizer revision. Probabilities are conditional over listed options; confidence is normalized entropy concentration, **not** a calibrated accuracy estimate.

Use `"mode":"shared"` with at least two questions over the same state to prefill an exact token prefix once and branch independent KV caches. It is supported for SemIf only. The implementation checks that full token sequences actually share the state prefix. The default is direct mode. Prompts exceeding `max_context` fail clearly; no truncation occurs.

## CLI

```bash
openjev-semif score --state 'The server is down' --question 'What next?' \
  --option 'Inspect logs' --option 'Ignore alert'
openjev-semif bench --input examples/bench.json --output benchmark-results/run.json
openjev-semif eval --input examples/eval.jsonl --output benchmark-results/eval.json
```

The benchmark reports requests/s, decisions/s, p50/p95 latency, prefill time, and peak GPU memory. It compares likelihood, SemIf direct, and SemIf shared on the same model. The evaluation file is JSONL with `state`, `question`, `options`, and `label` (matching an option id).

## Architecture

- `api/`: HTTP lifecycle and typed question contracts.
- `prompting/`: stable prompts, hashes, exact answer-token and boundary checks.
- `scoring/`: separate likelihood and final-position SemIf scorers, temperature scaling.
- `backends/`: model-independent scoring contract and resident Torch/Hugging Face backend.
- `runtime/`: shared token-prefix/KV branching.
- `benchmarks/`: local comparison tooling.

The backend protocol has `forward`, `prefill`, `get_last_logits`, and `sequence_logprob`; this leaves room for remote or quantized backends without changing the HTTP contract. Calibration profiles can be stored per model revision and workload using `CalibrationProfile`; set `CALIBRATION_PROFILE` to apply one. The service rejects profiles for another model revision or scorer. `fit_temperature` fits a labelled validation set, which should be disjoint from evaluation data.

## Acknowledgements and licenses

The service lifecycle, `/score` and `/v1/systemone` typed API, and likelihood scoring are conceptually adapted from [daseinlabs/open-jev](https://github.com/daseinlabs/open-jev). Final-position option logits, strict answer-token validation, direct/shared modes, prompt hashing, reproducibility metadata, and temperature scaling are conceptually adapted from [TheoLeeCJ/SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev). This implementation was written as a new package; source directories, tests, benchmark datasets, model weights and demo assets were not copied. Both upstream projects use MIT; their copyright notices are retained in [LICENSE](LICENSE) and [LICENSE-SemIf](LICENSE-SemIf). Model weights remain subject to their own license.

MLX, browser demos, training heads, chess/Doom examples, serial mode, and publication-specific evaluations were left out of the service scope. The initial shared mode uses safe independent KV branches; a batched branch path can follow after direct/shared numerical equivalence is established on the target model.

## RTX 3090 benchmark

The committed [report](benchmarks/qwen3.5-4b-rtx3090.json) uses Qwen3.5-4B revision `851bf6e...`, BF16, a 3435-token first prompt, three questions per state, and three repetitions. The shared prefix contained 3394 tokens. Times include tokenization and inference but exclude model startup. The bundled evaluation has two synthetic sanity cases; it is not a quality benchmark.

| Mode | Requests/s | Decisions/s | p50 request latency | p95 request latency | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| Likelihood (shared context within each decision) | 1.00 | 1.00 | 0.895 s | 1.432 s | 9.52 GB |
| SemIf direct | 1.25 | 1.25 | 0.800 s | 0.811 s | 9.37 GB |
| SemIf shared | 1.06 | 3.18 | 0.942 s | 0.947 s | 9.52 GB |

Direct and shared selected the same option in all three benchmark questions. The report also includes prefill and branch timings. Shared mode's request groups three decisions, so request throughput and decision throughput differ. Results depend on this workload and installed kernels; the run used PyTorch reference fallbacks for two Qwen kernels.
