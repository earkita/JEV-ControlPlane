# Attribution and provenance

The implementation in `src/openjev_semif/` was written for this repository. No upstream Python modules, tests, datasets, model weights, or browser assets were copied.

| Source | Revision reviewed | Concepts adapted | License |
|---|---|---|---|
| [daseinlabs/open-jev](https://github.com/daseinlabs/open-jev) | `4627a04a075bf714f4b60b69de19fa2de0a8a431` | Resident HTTP service, typed SystemOne questions, likelihood option scoring, CLI service workflow | MIT; copyright Dasein Labs, full text in `LICENSE` |
| [TheoLeeCJ/SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev) | `23cf1f39fc9534fe81437200959b6dfc7106e45a` | Final-position option logits, prompt hashing, answer-token/boundary checks, shared KV branching, temperature scaling, reproducibility fields | MIT; copyright TheoLeeCJ, full text in `LICENSE-SemIf` |

Qwen/Qwen3.5-4B model files are downloaded by the operator and are not included in this repository. Check the [model card](https://huggingface.co/Qwen/Qwen3.5-4B) for its own terms.
