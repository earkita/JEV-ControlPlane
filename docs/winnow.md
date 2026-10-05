# Winnow-12B Q8_0 with vision

This profile serves [EldanRing/Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B) through the native [Winnow inference server](https://github.com/EldanRing/winnow-inference). Our FastAPI process provides `/health`, `/score`, `/v1/systemone`, `/v1/cases/evaluate`, and `/ui`. One native process keeps the Q8_0 GGUF and F16 vision projector loaded on the GPU. The browser sends image data to FastAPI; it never loads model weights.

## Files and revisions

- Model: `/mnt/ai/models/classifiers/Winnow-12B/Winnow-12B-Q8_0.gguf`, SHA-256 `b710efc4c0d048ee61eed92c5fef5ce323a4d17e7c51f9f0533cc72ae50818ea`.
- Projector: `/mnt/ai/models/classifiers/Winnow-12B/mmproj-Winnow-12B.gguf`, SHA-256 `91f086971e56d7a7d8d39e271873fccdb49541bd259d6e02c401a4f1cb7a219e`.
- Hugging Face repository revision: `abb21621114b10690259a7517fa59b153675176e`.
- Native inference source revision used here: `77d14580c6732ca2f3745750c1dc1fd446d8bcee` in `~/ai/winnow-inference-reference`, outside this repository.
- `LICENSE`, `NOTICE`, `SHA256SUMS`, and `release-manifest.json` live beside the model files. The model files are Apache 2.0; the native runtime is MIT.

The default config is [serve-winnow12b.json](../config/serve-winnow12b.json). Change `revision` and `tokenizer_revision` if replacing the GGUF. The tokenizer is embedded in that file, so `tokenizer_revision` records its checksum.

## Build and launch

The native runtime must be built once. Follow its [installation guide](https://github.com/EldanRing/winnow-inference/blob/main/docs/INSTALL.md). This host has RTX 3090 (SM 86), CUDA 12.8, glibc with new C23 math declarations, and a local GCC 13 toolchain. The build used:

```bash
cd ~/ai/winnow-inference-reference
CC=/usr/bin/gcc-13 \
CXX=~/ai/toolchains/gcc13/bin/g++-13-local \
CUDAHOSTCXX=~/ai/toolchains/gcc13/bin/g++-13-local \
cmake -S . -B .build-gcc13 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_METAL=OFF -DCMAKE_CUDA_ARCHITECTURES=86 \
  '-DCMAKE_CUDA_FLAGS=-U_GNU_SOURCE -D_DEFAULT_SOURCE -include /home/ea/ai/toolchains/gcc13/pthread_clock_compat.h'
cmake --build .build-gcc13 --target llama-server winnow-unit -j 8
ctest --test-dir .build-gcc13 --output-on-failure
```

The compatibility header declares the four `pthread_*_clock*` functions whose feature macros are hidden by `-U_GNU_SOURCE`. It is local build support, not model code. The flags follow the [NVIDIA CUDA samples compatibility discussion](https://github.com/NVIDIA/cuda-samples/pull/403/files). On systems with a newer compatible CUDA toolkit, use the upstream build script directly.

From this repository:

```bash
./launch-winnow12b.sh
```

The launcher starts both processes and stops the native server when the API exits. It defaults to API port `8002`, native port `8091`, 8192 context positions, Q8 KV, and exclusive memory scheduling. `PORT`, `WINNOW_MODEL`, `WINNOW_MMPROJ`, `WINNOW_SERVER_BIN`, `WINNOW_SERVER_PORT`, and `WINNOW_CONTEXT` override defaults. Keep the native and API context sizes equal. The service is local to `127.0.0.1`.

## Image decisions

Open `http://127.0.0.1:8002/ui`. Choose **Wczytaj przykład rozpoznawania**, select a PNG, JPEG, or WebP image, then run the decision. You can replace the example's state, question, and options. The UI accepts one file up to 8 MiB; the JSON API accepts up to four still images with a 20 MB decoded request limit.

For API calls, encode each image as a `data:image/png;base64,...` URL in the top-level `images` array beside `state` for `/score` or `/v1/systemone`. For `/v1/cases/evaluate`, place `images` on each case. The native server does not fetch remote image URLs or read caller-supplied file paths.

Winnow vision decisions return probabilities over the supplied options. The confidence field is normalized entropy concentration, not an empirical correctness probability. If you need a free-form image description, that is a separate chat/generation workflow outside this decision API.

The adapter calls native `/v1/winnow/inspect` before scoring. This obtains the exact token IDs and image-position count, checks the context limit, and hashes token IDs plus image content. The native `/v1/systemone` call performs scoring. Shared mode submits all questions together for one native state prefill; direct mode submits them separately.
