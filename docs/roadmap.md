# Next steps

1. **Qwen 27B:** Keep the same scoring and API contracts. Add a quantized or multi-GPU Torch backend and explicit memory tests; a 27B BF16 checkpoint alone exceeds 24 GB. Recheck A–P token boundaries and direct/shared equivalence for the exact tokenizer and runtime.
2. **GLM and other causal LMs:** Implement the backend protocol, then run the answer-token and cache correctness tests on each tokenizer/model pair. If a letter is not one stable token, offer an explicitly named multi-token scorer instead of silently changing SemIf semantics.
3. **Remote inference:** Add a backend that can return full final-position vocabulary logits and identify model/tokenizer revisions. If an endpoint cannot expose exact logits or KV branches, disable those modes clearly.
4. **Hermes or Claude Code harness:** Wrap `/v1/systemone` as a decision tool, pass structured state and criteria, record the returned prompt hash and revision in agent traces, and test timeout/retry behavior with a local server.
5. **Performance:** Add batched CUDA KV branches after comparing logits to the current sequential branches. Benchmark longer state and larger question batches, and test optional accelerated Qwen kernels separately.
6. **Calibration:** Collect labelled validation data from the target workload, fit temperature with `fit_temperature`, save a `CalibrationProfile` for that model revision/workload, then assess calibration on held-out examples.
