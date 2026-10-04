"""One resident Hugging Face causal model and native KV branches."""
from __future__ import annotations
import copy
import inspect
from pathlib import Path

class TorchHFBackend:
    def __init__(self, settings):
        import torch
        from huggingface_hub import model_info
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_name = settings.model
        self.device = settings.device
        self.dtype = settings.dtype
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        dtype = getattr(torch, settings.dtype, None)
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("dtype must be bfloat16, float16, or float32")
        if self.device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device does not support bfloat16")
        local = Path(settings.model).exists()
        if Path(settings.model).is_absolute() and not local:
            raise FileNotFoundError(f"Local model directory does not exist: {settings.model}")
        revision = settings.revision or ("local" if local else model_info(settings.model).sha)
        self.model_revision = revision
        self.tokenizer_revision = revision
        remote_revision = None if local else revision
        config = AutoConfig.from_pretrained(settings.model, revision=remote_revision, trust_remote_code=False)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model, revision=remote_revision, trust_remote_code=False)
        model_class = AutoModelForCausalLM
        if config.model_type in {"qwen3_5", "qwen3_5_text"}:
            from transformers import Qwen3_5ForCausalLM
            model_class = Qwen3_5ForCausalLM
            if config.model_type == "qwen3_5": config = config.get_text_config()
        self.model = model_class.from_pretrained(
            settings.model, revision=remote_revision, config=config, dtype=dtype,
            trust_remote_code=False, low_cpu_mem_usage=True,
        ).to(self.device).eval()
        model_context = getattr(config, "max_position_embeddings", None)
        self.max_context = min(settings.max_context, model_context) if model_context else settings.max_context
        import transformers
        self.metadata = {"torch": torch.__version__, "transformers": transformers.__version__,
                         "device": self.device, "dtype": self.dtype,
                         "gpu": torch.cuda.get_device_name(self.device) if str(self.device).startswith("cuda") else None}
        self._selective_logits = "logits_to_keep" in inspect.signature(self.model.forward).parameters

    def synchronize(self):
        if str(self.device).startswith("cuda"): self.torch.cuda.synchronize(self.device)

    def _tensor(self, ids):
        return self.torch.tensor([ids], dtype=self.torch.long, device=self.device)

    def forward(self, ids):
        with self.torch.inference_mode():
            kwargs = {"input_ids": self._tensor(ids), "use_cache": False, "return_dict": True}
            if self._selective_logits: kwargs["logits_to_keep"] = 1
            logits = self.model(**kwargs).logits[0, -1].float()
            self.synchronize()
            return logits.cpu().tolist()

    def prefill(self, ids):
        with self.torch.inference_mode():
            kwargs = {"input_ids": self._tensor(ids), "use_cache": True, "return_dict": True}
            if self._selective_logits: kwargs["logits_to_keep"] = 1
            out = self.model(**kwargs)
            cache = out.past_key_values
            last = out.logits[0, -1].float().clone()
            self.synchronize()
            if cache is None: raise RuntimeError("model did not return KV cache")
            if hasattr(cache, "get_seq_length") and cache.get_seq_length() != len(ids):
                raise RuntimeError("KV cache has incorrect prefix length")
            return cache, len(ids), last

    def get_last_logits(self, cache, suffix):
        base, prefix_length, _ = cache
        branch = copy.deepcopy(base)
        with self.torch.inference_mode():
            kwargs = {
                "input_ids": self._tensor(suffix),
                "attention_mask": self.torch.ones((1, prefix_length + len(suffix)), dtype=self.torch.long, device=self.device),
                "past_key_values": branch, "use_cache": True, "return_dict": True,
            }
            if self._selective_logits: kwargs["logits_to_keep"] = 1
            logits = self.model(**kwargs).logits[0, -1].float()
            self.synchronize()
            return logits.cpu().tolist()

    def score_continuations(self, cache, continuations):
        """Score all option texts from one context prefill and independent KV branches."""
        base, prefix_length, last = cache
        results = []
        with self.torch.inference_mode():
            first_log_probs = last.log_softmax(-1)
            for continuation in continuations:
                if not continuation: raise ValueError("empty option tokens")
                score = first_log_probs[continuation[0]]
                if len(continuation) > 1:
                    branch = copy.deepcopy(base)
                    inputs = self._tensor(continuation[:-1])
                    attention = self.torch.ones((1, prefix_length + len(continuation)-1),
                                                dtype=self.torch.long, device=self.device)
                    logits = self.model(input_ids=inputs, attention_mask=attention,
                                        past_key_values=branch, use_cache=True,
                                        return_dict=True).logits[0].float()
                    target = self.torch.tensor(continuation[1:], dtype=self.torch.long,
                                               device=self.device)
                    score = score + (logits.gather(-1, target[:, None]).squeeze(-1)
                                     - logits.logsumexp(-1)).sum()
                results.append(float(score.item()))
            self.synchronize()
        return results

    def sequence_logprob(self, prefix, continuation):
        if not prefix or not continuation: raise ValueError("prefix and continuation must contain tokens")
        ids = prefix + continuation
        if len(ids) > self.max_context: raise ValueError("likelihood prompt exceeds max_context; truncation disabled")
        with self.torch.inference_mode():
            out = self.model(input_ids=self._tensor(ids), use_cache=False, return_dict=True)
            pred = out.logits[0, len(prefix)-1:-1].float()
            target = self.torch.tensor(continuation, dtype=self.torch.long, device=self.device)
            score = (pred.gather(-1, target[:, None]).squeeze(-1) - pred.logsumexp(-1)).sum()
            self.synchronize()
            return float(score.item())

    def gpu_memory_bytes(self):
        return self.torch.cuda.max_memory_allocated(self.device) if str(self.device).startswith("cuda") else None
