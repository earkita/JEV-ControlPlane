# OpenJEV-SemIf

A single-model decision service with two explicit scoring methods. **SemIf** selects final-position letter logits for listed options; **likelihood** computes the conditional log probability of each option's text. Neither method generates an answer. The Torch profile loads one Hugging Face model at startup; the GGUF profiles use one resident inference process behind the same API.

## Install

Python 3.12+ and a CUDA PyTorch build are required for the target GPU. In this repository:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[torch,test]'
```

For CUDA wheels, follow the [PyTorch installation selector](https://pytorch.org/get-started/locally/) if your environment does not already supply a compatible build.

## Run on RTX 3090

The repository launcher reads [config/serve.json](config/serve.json). Its default model path is `~/ai/models/qwen/Qwen3.5`, with the pinned revision, CUDA, BF16, SemIf, and port 8000:

```bash
CUDA_VISIBLE_DEVICES=0 ./launch.sh
```

Override any default with environment variables or CLI flags. The order is **repository config < environment < CLI**:

```bash
PORT=8001 ./launch.sh
./launch.sh --port 8002 --scorer likelihood
OPENJEV_CONFIG=/path/to/serve.json ./launch.sh --port 8003
```

The launcher finds the repository's `.venv/bin/openjev-semif`, or an installed `openjev-semif` on `PATH`. It can be run from any working directory. The equivalent direct CLI command is:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/openjev-semif serve \
  --model Qwen/Qwen3.5-4B --backend torch --scorer semif \
  --device cuda --dtype bfloat16 --port 8000
```

The Hugging Face model revision is resolved to a commit before loading and is returned in responses. Set `--revision` to pin one explicitly. `MODEL`, `REVISION`, `BACKEND`, `DEVICE`, `DTYPE`, `SCORER`, `MAX_CONTEXT`, `TEMPERATURE`, `CALIBRATION_PROFILE`, `HOST`, and `PORT` are supported environment settings. `--config settings.json` may supply the same fields.

### Keep model weights outside the repository

For an explicit local model directory, download the pinned snapshot once:

```bash
hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir "$HOME/ai/models/qwen/Qwen3.5"
```

Then start the service from those local files:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/openjev-semif serve \
  --model "$HOME/ai/models/qwen/Qwen3.5" \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --backend torch --scorer semif --device cuda --dtype bfloat16 --port 8000
```

The `--revision` value records the upstream snapshot in responses when `--model` is a local path. The model directory is outside Git; `--model Qwen/Qwen3.5-4B` remains available and uses the normal Hugging Face cache.

### OpenJev 27B on RTX 3090

The [OpenJev 27B S1MB submission](https://huggingface.co/datasets/hotchpotch/s1mb-result/discussions/2) evaluated the 16-bit model. Its weights need about 54 GB; the local 24 GB profile uses the **text-only Q4_K_M GGUF** (16.5 GB). The [GGUF model card](https://huggingface.co/openjev/openjev-GGUF) reports 82.84% on its 1,789-row validation set versus 83.17% for the 16-bit reference. This is a different precision and evaluation set from the leaderboard entry, so local results should not be called a reproduction of its score.

Install the Python package as above, then download the pinned GGUF, tokenizer, and a CUDA `llama-server` outside Git:

```bash
mkdir -p "$HOME/ai/models/openjev/OpenJev-27B-Q4_K_M/tokenizer" "$HOME/ai/llama.cpp-prebuilt/llama-b11393"
hf download openjev/openjev-GGUF OpenJev-Q4_K_M.gguf SHA256SUMS LICENSE NOTICE MANIFEST.json \
  --revision 7c0a4c624342dd283f8d1bd215fbbe72cb0059f7 \
  --local-dir "$HOME/ai/models/openjev/OpenJev-27B-Q4_K_M"
hf download openjev/openjev tokenizer.json tokenizer_config.json chat_template.jinja \
  --revision 5ec9e5fd2f80a6fff386779b1e5ac7e389971889 \
  --local-dir "$HOME/ai/models/openjev/OpenJev-27B-Q4_K_M/tokenizer"
cd "$HOME/ai/llama.cpp-prebuilt/llama-b11393"
curl -fL -o llama-cuda.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b11393/llama-b11393-bin-ubuntu-cuda-12.8-x64.tar.gz
tar -xzf llama-cuda.tar.gz
cd -
```

Verify the GGUF hash against the downloaded `SHA256SUMS` before serving it. Use a compatible NVIDIA driver for the CUDA 12.8 binary; alternatively build `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp) and set `LLAMA_SERVER_BIN` to that binary. With the 4B server stopped to free GPU memory:

```bash
CUDA_VISIBLE_DEVICES=0 ./launch-openjev27b.sh
curl -s http://127.0.0.1:8002/health
```

The 27B API and workbench are at `http://127.0.0.1:8002` and `/ui`. The launcher starts and stops `llama-server` with the API process. It reads [config/serve-openjev27b.json](config/serve-openjev27b.json); `PORT=8003 ./launch-openjev27b.sh` overrides the API port, `LLAMA_SERVER_BIN`, `OPENJEV27B_MODEL`, `LLAMA_SERVER_PORT`, `LLAMA_CONTEXT`, and `LLAMA_N_GPU_LAYERS` control the inference process. The default context is 8192 tokens and the default choice temperature is 0.85. The model is loaded once, and requests with excess input fail without truncation.

This profile supports `/score`, typed `choice`, `score`, and `noul` under `/v1/systemone`, and the existing HTTP client and UI. It supports **SemIf direct only**; likelihood and shared KV mode require separate correctness work for this GGUF runtime. Its `bench` and `eval` CLI commands remain Torch-only. The GGUF is text-only. Its weights are licensed **CC BY-NC 4.0** (attribution and noncommercial use); see [NOTICE](NOTICE.md) and the upstream [model card](https://huggingface.co/openjev/openjev-GGUF).

### Winnow-12B Q8_0 with vision on RTX 3090

The [Winnow model card](https://huggingface.co/EldanRing/Winnow-12B) distributes Q8_0 weights and a separate F16 vision projector under Apache 2.0. Put both outside this repository:

```bash
mkdir -p /mnt/ai/models/classifiers/Winnow-12B
hf download EldanRing/Winnow-12B gguf/Winnow-12B-Q8_0.gguf gguf/mmproj-Winnow-12B.gguf \
  --revision abb21621114b10690259a7517fa59b153675176e \
  --local-dir /mnt/ai/models/classifiers/Winnow-12B
```

The launcher accepts files either directly in `/mnt/ai/models/classifiers/Winnow-12B` or under its `gguf/` directory. `WINNOW_MODEL` and `WINNOW_MMPROJ` override these defaults. Verify hashes against the [release manifest](https://huggingface.co/EldanRing/Winnow-12B/blob/main/release-manifest.json): Q8_0 `b710efc4…a50818ea`, projector `91f08697…a7a219e`.

Build the pinned [Winnow inference server](https://github.com/EldanRing/winnow-inference) outside this repository, for RTX 3090 CUDA architecture 86. The reference checkout is expected at `~/ai/winnow-inference-reference`; set `WINNOW_RUNTIME_DIR` to another path if needed. This host uses CUDA 12.8 with a locally installed GCC 13 toolchain and a glibc compatibility header; see the [upstream build instructions](https://github.com/EldanRing/winnow-inference/blob/main/docs/INSTALL.md) for other systems. Then run:

```bash
CUDA_VISIBLE_DEVICES=0 ./launch-winnow12b.sh
```

The API and UI are at `http://127.0.0.1:8002` and `/ui`. The launcher reads [config/serve-winnow12b.json](config/serve-winnow12b.json); `PORT`, `WINNOW_MODEL`, `WINNOW_MMPROJ`, `WINNOW_SERVER_BIN`, `WINNOW_SERVER_PORT`, `WINNOW_CONTEXT`, `WINNOW_BATCH`, and `WINNOW_UBATCH` override defaults. The profile uses an 8192-position context, 4096 microbatch, Q8 KV, exclusive context scheduling, and one loaded model with projector. The native server evaluates typed questions and reuses a state prefix for `"mode":"shared"`; direct mode submits questions separately. It rejects context overflows without truncation.

The config records the pinned Hugging Face repository revision as `model_revision` and the GGUF checksum as `tokenizer_revision`, since this GGUF embeds its tokenizer. Update both when replacing the model file.

The UI accepts one uploaded PNG, JPEG, or WebP image up to 8 MiB. Choose or paste a `/score` or `/v1/systemone` question and answer options to classify the image. The API accepts `"images": ["data:image/png;base64,..."]` alongside `state`; case batches place `images` on each case. Image bytes remain local to the browser, API, and inference server. Vision decisions are SemIf option probabilities; they do not generate a free-form image description. Native Winnow decisions use its model-specific prompt and final-position candidate logits. Their `prompt_hash` covers the exact native prefix/suffix token IDs and image content hashes.

See the [Winnow deployment guide](docs/winnow.md) for artifact hashes, native build details, API image format, and operating settings.

## API example

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

Use `"mode":"shared"` with at least two questions over the same state to prefill an exact token prefix once and branch independent KV caches. It is supported for SemIf on the Torch backend only. The implementation checks that full token sequences actually share the state prefix. The default is direct mode. Prompts exceeding `max_context` fail clearly; no truncation occurs.

## Generated decision cases

For AI-generated test inputs, send arrays of cases with a `state`, named questions, candidate `{id, description}` options and an optional `expected_option`. `POST /v1/cases/validate` checks the JSON without inference; `POST /v1/cases/evaluate` scores it and compares labelled answers. See the [case API guide](docs/decision-cases.md) and [example batch](examples/decision-cases.json). Generated labels should be reviewed before using aggregate matches as an accuracy measurement.

Ready-to-paste UI examples: [several cases](examples/ui-cases-demo.json), [typed SystemOne questions](examples/ui-systemone-demo.json), and [one `/score` decision](examples/ui-score-demo.json).

## Web UI

The server also serves a built-in decision workbench at **`http://127.0.0.1:8000/ui`** (or `/`; the GGUF profiles use port 8002). Its default **Wklej JSON** tab accepts a `cases` batch, one case, `/score` input or `/v1/systemone` input and draws the resulting decisions and probability bars. Manual single-decision and multi-question forms remain available. On Torch, choose SemIf or likelihood and direct or shared mode. OpenJev 27B uses SemIf direct; Winnow also supports shared mode and image upload. Results show option distributions, confidence, timings, token counts, prompt hashes, and model revision; JSON can be copied or downloaded.

The UI uses local HTML/CSS/JavaScript assets in `src/openjev_semif/web/`. It is served by the same FastAPI process and does not load a second model, need Gradio, or require a frontend build. The API documentation remains available at `/docs`. If `OPENJEV_API_KEY` protects `/v1/systemone`, enter the key in the UI's authorization field for that browser tab; it is not saved by the page.

## HTTP client

`openjev-semif serve` hosts the API. The separate Python and CLI clients call that service **without loading a model in the client process**. `openjev-semif score` remains the local, in-process scorer.

```bash
openjev-semif client health --url http://127.0.0.1:8000
openjev-semif client score --url http://127.0.0.1:8000 \
  --state 'The build failed' --question 'What next?' \
  --option 'Inspect logs' --option 'Ignore it'
openjev-semif client systemone --url http://127.0.0.1:8000 \
  --input examples/systemone.json
```

```python
from openjev_semif import OpenJEVClient

with OpenJEVClient("http://127.0.0.1:8000") as client:
    print(client.health().status)
    result = client.score({
        "state": "The build failed",
        "question": "What next?",
        "options": ["Inspect logs", "Ignore it"],
    })
    print(result.best, result.options[0].probability)
```

For a protected `/v1/systemone`, pass `api_key=...` to `OpenJEVClient` or set `OPENJEV_API_KEY` for the CLI. The CLI also accepts `OPENJEV_URL` and `--timeout`. HTTP errors raise `OpenJEVHTTPError` with `status_code` and `detail`; connection and timeout errors come from `httpx`.

## CLI

```bash
openjev-semif score --state 'The server is down' --question 'What next?' \
  --option 'Inspect logs' --option 'Ignore alert'
openjev-semif bench --input examples/bench.json --output benchmark-results/run.json
openjev-semif eval --input examples/eval.jsonl --output benchmark-results/eval.json
```

The benchmark reports requests/s, decisions/s, p50/p95 latency, prefill time, and peak GPU memory. It compares likelihood, SemIf direct, and SemIf shared on the same model. The evaluation file is JSONL with `state`, `question`, `options`, and `label` (matching an option id).

## Architecture

See [architecture](docs/architecture.md) and [roadmap](docs/roadmap.md). Future AI contributors should follow [AGENTS.md](AGENTS.md).


- `api/`: HTTP lifecycle and typed question contracts.
- `client.py`: typed HTTP client for a running service.
- `prompting/`: stable prompts, hashes, exact answer-token and boundary checks.
- `scoring/`: separate likelihood and final-position SemIf scorers, temperature scaling.
- `backends/`: model-independent scoring contract and resident Torch/Hugging Face backend.
- `runtime/`: shared token-prefix/KV branching.
- `benchmarks/`: local comparison tooling.

The Torch backend protocol has `forward`, `prefill`, `get_last_logits`, and `sequence_logprob`. The OpenJev GGUF adapter asks a resident llama.cpp server for one-token log probabilities; the Winnow adapter calls the native typed-decision endpoint and vision projector. Neither GGUF backend claims likelihood scoring. Calibration profiles can be stored per model revision and workload using `CalibrationProfile`; set `CALIBRATION_PROFILE` to apply one. The service rejects profiles for another model revision or scorer. `fit_temperature` fits a labelled validation set, which should be disjoint from evaluation data.

## Acknowledgements and licenses

The service lifecycle, `/score` and `/v1/systemone` typed API, and likelihood scoring are conceptually adapted from [daseinlabs/open-jev](https://github.com/daseinlabs/open-jev). Final-position option logits, strict answer-token validation, direct/shared modes, prompt hashing, reproducibility metadata, and temperature scaling are conceptually adapted from [TheoLeeCJ/SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev). This implementation was written as a new package; source directories, tests, benchmark datasets, model weights and demo assets were not copied. Both upstream projects use MIT; their copyright notices are retained in [LICENSE](LICENSE) and [LICENSE-SemIf](LICENSE-SemIf). Model weights remain subject to their own license.

The Winnow integration uses the public API of [EldanRing/winnow-inference](https://github.com/EldanRing/winnow-inference) (MIT); no native source is copied into this repository. The [Winnow-12B weights and projector](https://huggingface.co/EldanRing/Winnow-12B) are Apache 2.0 and stay outside Git. Retain the model repository's LICENSE and NOTICE when redistributing those artifacts.

MLX, browser demos, training heads, chess/Doom examples, serial mode, and publication-specific evaluations were left out of the service scope. The initial shared mode uses safe independent KV branches; a batched branch path can follow after direct/shared numerical equivalence is established on the target model.

## RTX 3090 benchmark

The committed [report](benchmarks/qwen3.5-4b-rtx3090.json) uses Qwen3.5-4B revision `851bf6e...`, BF16, a 3435-token first prompt, three questions per state, and three repetitions. The shared prefix contained 3394 tokens. Times include tokenization and inference but exclude model startup. The bundled evaluation has two synthetic sanity cases; it is not a quality benchmark.

| Mode | Requests/s | Decisions/s | p50 request latency | p95 request latency | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| Likelihood (shared context within each decision) | 1.00 | 1.00 | 0.895 s | 1.432 s | 9.52 GB |
| SemIf direct | 1.25 | 1.25 | 0.800 s | 0.811 s | 9.37 GB |
| SemIf shared | 1.06 | 3.18 | 0.942 s | 0.947 s | 9.52 GB |

Direct and shared selected the same option in all three benchmark questions. The report also includes prefill and branch timings. Shared mode's request groups three decisions, so request throughput and decision throughput differ. Results depend on this workload and installed kernels; the run used PyTorch reference fallbacks for two Qwen kernels.
