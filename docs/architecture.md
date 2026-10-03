# Architecture and limits

A single `TorchHFBackend` owns one tokenizer and model for the server lifetime. `ModelBackend` is the minimal scoring protocol: full final logits, KV prefill, KV branch final logits, continuation log probability, and synchronization. The scorers are independent of Qwen; the backend handles the Qwen3.5 text-only model class when its Hugging Face config is multimodal.

`SemIfScorer` renders the evidence, question and labelled descriptions, then reads only the final-position logits for verified answer tokens A–P. The tokenizer must encode each letter as one exact token at the answer boundary. It softmaxes the selected logits after optional temperature scaling. `LikelihoodScorer` renders a separate context and sums `log P(option token | previous tokens)` for the text of each option. Their raw scores have different meanings and should not be compared directly.

Shared mode tokenizes every complete prompt, finds their exact common token prefix, verifies it includes the state, and prefills that prefix once. It copies the native KV cache for each question suffix and obtains each final-position vocabulary. No state tokens are forwarded again in the branches. This design favors correctness over peak throughput; CUDA batch branching is a future optimization. The server uses one worker to avoid loading multiple model copies.

Every request rejects a prompt longer than the configured or model context limit. No silent truncation occurs. Responses include the model/tokenizer revision, prompt hash, prompt version, token count, scorer, temperature, and separate timing fields. The default confidence is normalized entropy concentration; use a labelled validation set and a workload-specific temperature profile when calibrated probabilities are required.

The supplied benchmark runs all three methods on the same loaded model. Direct and likelihood count each decision as one request; shared counts a group of questions over one state as one request. Latency percentiles are request latencies. They are local measurements and depend on the exact prompt, CUDA build, and optional accelerated kernels.
