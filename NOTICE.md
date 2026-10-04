# Attribution and provenance

The implementation in `src/openjev_semif/` was written for this repository. No upstream Python modules, tests, datasets, model weights, or browser assets were copied.

| Source | Revision reviewed | Concepts adapted | License |
|---|---|---|---|
| [daseinlabs/open-jev](https://github.com/daseinlabs/open-jev) | `4627a04a075bf714f4b60b69de19fa2de0a8a431` | Resident HTTP service, typed SystemOne questions, likelihood option scoring, CLI service workflow | MIT; copyright Dasein Labs, full text in `LICENSE` |
| [TheoLeeCJ/SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev) | `23cf1f39fc9534fe81437200959b6dfc7106e45a` | Final-position option logits, prompt hashing, answer-token/boundary checks, shared KV branching, temperature scaling, reproducibility fields | MIT; copyright TheoLeeCJ, full text in `LICENSE-SemIf` |

Qwen/Qwen3.5-4B model files are downloaded by the operator and are not included in this repository. Check the [model card](https://huggingface.co/Qwen/Qwen3.5-4B) for its own terms.

The optional [openjev/openjev-GGUF](https://huggingface.co/openjev/openjev-GGUF) Q4_K_M weights and tokenizer from [openjev/openjev](https://huggingface.co/openjev/openjev) are downloaded separately outside this repository. The weights are licensed CC BY-NC 4.0 and require attribution for noncommercial use. Their model repositories contain the license and base-model attribution; no weights are committed here. The 27B adapter follows the published decision prompt and calibration defaults conceptually, without copying the upstream Python helper. The llama.cpp runtime is a separate dependency with its own license.
