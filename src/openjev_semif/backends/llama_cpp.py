"""Direct SemIf readout from a pinned GGUF served by llama.cpp.

This backend owns no second decision service: llama-server handles model inference,
while the existing OpenJEV API owns validation, scoring and response metadata.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import httpx

from ..scoring.base import Decision, Option, Result
from ..scoring.calibration import confidence, probabilities

PROMPT_VERSION = "openjev27b-gguf-text-v1"
LETTERS = "ABCDEFGHIJKLMNOP"


class BackendUnavailable(RuntimeError):
    """The inference server cannot provide a complete, verifiable readout."""


class LlamaCppBackend:
    def __init__(self, settings, *, client=None, tokenizer=None):
        model_path = Path(settings.model).expanduser().resolve()
        tokenizer_path = Path(settings.tokenizer_path or "").expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"GGUF model file does not exist: {model_path}")
        if not settings.tokenizer_path or not tokenizer_path.is_dir():
            raise FileNotFoundError(f"GGUF tokenizer directory does not exist: {tokenizer_path}")
        if not settings.revision or not settings.tokenizer_revision:
            raise ValueError("GGUF backend requires revision and tokenizer_revision")
        if not settings.backend_url:
            raise ValueError("GGUF backend requires backend_url")
        if settings.scorer != "semif":
            raise ValueError("GGUF backend supports semif only")
        self.model_name = "openjev/openjev-GGUF"
        self.model_revision = settings.revision
        self.tokenizer_revision = settings.tokenizer_revision
        self.device = settings.device
        self.dtype = settings.dtype
        self.max_context = settings.max_context
        self.noul_temperature = settings.noul_temperature
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path), local_files_only=True, trust_remote_code=False)
        self.tokenizer = tokenizer
        self._owned = client is None
        self.client = client or httpx.Client(base_url=settings.backend_url.rstrip("/"), timeout=180)
        health = self._request("GET", "/health")
        if health.get("status") != "ok":
            raise BackendUnavailable("llama-server is not ready")
        props = self._request("GET", "/props")
        loaded = props.get("model_path")
        if loaded and Path(loaded).resolve() != model_path:
            raise BackendUnavailable(f"llama-server loaded a different model: {loaded}")
        context = props.get("default_generation_settings", {}).get("n_ctx")
        if context and context < self.max_context:
            raise BackendUnavailable(
                f"llama-server context {context} is below configured max_context {self.max_context}")
        self.letter_ids = []
        for letter in LETTERS:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise BackendUnavailable(f"answer letter {letter!r} is not one tokenizer token")
            remote = self._request("POST", "/tokenize", {"content": letter, "add_special": False})["tokens"]
            if remote != ids:
                raise BackendUnavailable(f"tokenizer mismatch for answer letter {letter!r}")
            self.letter_ids.append(ids[0])
        if len(set(self.letter_ids)) != len(self.letter_ids):
            raise BackendUnavailable("answer letters do not map to distinct tokens")

    def close(self):
        if self._owned:
            self.client.close()

    def _request(self, method, path, payload=None):
        try:
            response = self.client.request(method, path, json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendUnavailable(f"llama-server {path} failed: {exc}") from exc

    def _prompt(self, decision: Decision, question_type: str | None = None):
        state = decision.state if isinstance(decision.state, str) else json.dumps(
            decision.state, ensure_ascii=False)
        options = decision.options
        question = decision.question
        if question_type == "noul":
            options = [
                Option("yes", options[0].description if options[0].description != "Yes" else "The statement is true."),
                Option("no", options[1].description if options[1].description != "No" else "The statement is false."),
            ]
        elif question_type == "score":
            question += " Rate along the ordered levels below (lowest first)."
        lines = "\n".join(
            f"[{LETTERS[i]}] {option.id}: {option.description}"
            for i, option in enumerate(options))
        user = (f"State:\n{state}\n\nQuestion: {question}\nOptions:\n{lines}"
                "\n\nAnswer with the letter of the best option only.")
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    def score_decision(self, decision: Decision, temperature: float = 1.0,
                       question_type: str | None = None) -> Result:
        started = time.perf_counter()
        if not 2 <= len(decision.options) <= len(LETTERS):
            raise ValueError("GGUF backend supports 2 to 16 options")
        prompt = self._prompt(decision, question_type)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        tokenization = time.perf_counter() - started
        if len(ids) + 1 > self.max_context:
            raise ValueError(f"prompt has {len(ids)} tokens; max context is {self.max_context}; truncation disabled")
        for letter, token_id in zip(LETTERS[:len(decision.options)], self.letter_ids):
            continuation = self.tokenizer.encode(prompt + letter, add_special_tokens=False)
            if continuation != ids + [token_id]:
                raise ValueError(f"answer letter {letter!r} changes the prompt token boundary")
        remote_ids = self._request("POST", "/tokenize", {"content": prompt, "add_special": False})["tokens"]
        if remote_ids != ids:
            raise BackendUnavailable("llama-server tokenization differs from the pinned tokenizer")
        response = self._request("POST", "/completion", {
            "prompt": prompt, "n_predict": 1, "n_probs": 256,
            "temperature": -1, "top_k": 0, "top_p": 1.0,
            "min_p": 0.0, "repeat_penalty": 1.0,
            "cache_prompt": True, "stream": False,
            "post_sampling_probs": False,
        })
        if response.get("truncated"):
            raise BackendUnavailable("llama-server truncated the prompt")
        token_probs = response.get("completion_probabilities") or []
        if len(token_probs) != 1:
            raise BackendUnavailable("llama-server did not return one-token probabilities")
        scores_by_id = {int(row["id"]): float(row["logprob"])
                        for row in token_probs[0].get("top_logprobs", [])}
        selected_ids = self.letter_ids[:len(decision.options)]
        if any(token_id not in scores_by_id for token_id in selected_ids):
            raise BackendUnavailable("an answer letter is absent from llama-server top-token probabilities")
        raw = [scores_by_id[token_id] for token_id in selected_ids]
        if not all(math.isfinite(x) for x in raw):
            raise BackendUnavailable("llama-server returned non-finite answer scores")
        effective_temperature = temperature * (self.noul_temperature if question_type == "noul" else 1.0)
        probs = probabilities(raw, effective_temperature)
        best = max(range(len(probs)), key=probs.__getitem__)
        timing = response.get("timings") or {}
        prefill = float(timing.get("prompt_ms", 0)) / 1000
        scoring = float(timing.get("predicted_ms", 0)) / 1000
        return Result(
            best=decision.options[best].id, best_index=best,
            options=[{"option": option.id, "description": option.description,
                      "raw_score": score, "probability": prob}
                     for option, score, prob in zip(decision.options, raw, probs)],
            confidence=confidence(probs), scorer="semif", model=self.model_name,
            model_revision=self.model_revision, tokenizer_revision=self.tokenizer_revision,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            prompt_version=PROMPT_VERSION, input_tokens=len(ids), truncated=False,
            temperature=effective_temperature,
            timings={"tokenization_s": tokenization, "prefill_s": prefill,
                     "scoring_s": scoring, "total_s": time.perf_counter() - started},
        )
