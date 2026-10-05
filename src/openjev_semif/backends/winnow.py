"""Adapter for Winnow's native typed-decision and vision server.

The native server owns the one resident GGUF model and its KV cache. This adapter
keeps the public OpenJEV response shape and records exact native token IDs.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import httpx

from ..scoring.base import Decision, Result
from ..scoring.calibration import confidence
from .llama_cpp import BackendUnavailable

PROMPT_VERSION = "winnow-native-token-ids-v1"


class WinnowBackend:
    def __init__(self, settings, *, client=None):
        model = Path(settings.model).expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(f"Winnow GGUF model does not exist: {model}")
        projector = Path(settings.projector_path).expanduser().resolve() if settings.projector_path else None
        if projector and not projector.is_file():
            raise FileNotFoundError(f"Winnow vision projector does not exist: {projector}")
        if not settings.backend_url or not settings.revision or not settings.tokenizer_revision:
            raise ValueError("Winnow backend needs backend_url, revision and tokenizer_revision")
        if settings.scorer != "semif":
            raise ValueError("Winnow backend supports semif only")
        self.model_name = "EldanRing/Winnow-12B"
        self.model_revision = settings.revision
        self.tokenizer_revision = settings.tokenizer_revision
        self.device = settings.device
        self.dtype = settings.dtype
        self.max_context = settings.max_context
        self.vision = projector is not None
        self._owned = client is None
        self.client = client or httpx.Client(base_url=settings.backend_url.rstrip("/"), timeout=300)
        health = self._request("GET", "/health")
        if health.get("status") != "ok":
            raise BackendUnavailable("Winnow server is not ready")
        props = self._request("GET", "/props")
        loaded = props.get("model_path")
        if loaded and Path(loaded).resolve() != model:
            raise BackendUnavailable(f"Winnow server loaded a different model: {loaded}")
        if self.vision and not props.get("modalities", {}).get("vision", False):
            raise BackendUnavailable("Winnow server has no active vision projector")
        context = props.get("default_generation_settings", {}).get("n_ctx")
        if context and context < self.max_context:
            raise BackendUnavailable(f"Winnow context {context} is below max_context {self.max_context}")

    def close(self):
        if self._owned:
            self.client.close()

    def _request(self, method, path, payload=None):
        try:
            response = self.client.request(method, path, json=payload)
            data = response.json()
            if response.status_code == 400:
                raise ValueError(data.get("error", {}).get("message", "invalid Winnow request"))
            response.raise_for_status()
            return data
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise BackendUnavailable(f"Winnow server {path} failed: {exc}") from exc

    @staticmethod
    def _native_question(question):
        return question.model_dump(exclude_none=True)

    def score_batch(self, state, questions, temperature=1.0, images=None, mode="direct"):
        images = images or []
        if images and not self.vision:
            raise ValueError("vision projector is not configured")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if mode == "direct" and len(questions) > 1:
            return [self.score_batch(state, {key: question}, temperature, images, mode)[0]
                    for key, question in questions.items()]
        started = time.perf_counter()
        request = {
            "model": "Winnow-12B",
            "state": state,
            "questions": {key: self._native_question(value) for key, value in questions.items()},
            "winnow": {"images": images, "temperature": temperature,
                       "diagnostics": True, "include_token_ids": True,
                       "reuse_prefix": mode == "shared"},
        }
        inspection = self._request("POST", "/v1/winnow/inspect", request)
        prefix_tokens = inspection["prefix_tokens"] + inspection.get("request_prefix_tokens", 0)
        suffix_counts = inspection["suffix_tokens"]
        if len(suffix_counts) != len(questions):
            raise BackendUnavailable("Winnow inspection returned incomplete question tokens")
        if any(prefix_tokens + suffix + 1 > self.max_context for suffix in suffix_counts):
            raise ValueError(f"prompt exceeds max context {self.max_context}; truncation disabled")
        tokenization = time.perf_counter() - started
        response = self._request("POST", "/v1/systemone", request)
        if set(response.get("answers", {})) != set(questions):
            raise BackendUnavailable("Winnow returned incomplete answers")
        metrics = response.get("winnow") or {}
        logits = []
        for key, question in questions.items():
            answer = response["answers"][key]
            row = answer.get("winnow", {}).get("logits")
            expected = 2 if question.type == "noul" else len(question.criteria)
            if not isinstance(row, list) or len(row) != expected or not all(math.isfinite(x) for x in row):
                raise BackendUnavailable(f"Winnow returned incomplete logits for {key}")
            logits.append(row)
        image_hashes = [hashlib.sha256(url.encode()).hexdigest() for url in images]
        results = []
        for index, (key, question) in enumerate(questions.items()):
            answer = response["answers"][key]
            if question.type == "noul":
                keys = ["true", "false"]
                raw = [logits[index][1], logits[index][0]]
                p_true = float(answer["noul"])
                probs = [p_true, 1.0 - p_true]
                descriptions = [str((question.criteria or {}).get("true") or "Yes"),
                                str((question.criteria or {}).get("false") or "No")]
            elif question.type == "choice":
                keys = list(question.criteria)
                raw = logits[index]
                probs = [float(answer["probabilities"][item]) for item in keys]
                descriptions = [str(question.criteria[item] or item) for item in keys]
            else:
                keys = [str(i) for i in range(len(question.criteria))]
                raw = logits[index]
                probs = [float(answer["probabilities"][item]) for item in keys]
                descriptions = [str(item) for item in question.criteria]
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in probs) or abs(sum(probs) - 1) > 1e-5:
                raise BackendUnavailable(f"Winnow returned invalid probabilities for {key}")
            best = max(range(len(keys)), key=probs.__getitem__)
            token_identity = {
                "prefix": inspection["prefix_token_ids"],
                "request_prefix": inspection["request_prefix_token_ids"],
                "suffix": inspection["suffix_token_ids"][index],
                "images_sha256": image_hashes,
            }
            digest = hashlib.sha256(json.dumps(token_identity, separators=(",", ":")).encode()).hexdigest()
            results.append(Result(
                best=keys[best], best_index=best,
                options=[{"option": item, "description": desc, "raw_score": logit,
                          "probability": prob}
                         for item, desc, logit, prob in zip(keys, descriptions, raw, probs)],
                confidence=confidence(probs), scorer="semif", model=self.model_name,
                model_revision=self.model_revision, tokenizer_revision=self.tokenizer_revision,
                prompt_hash=digest, prompt_version=PROMPT_VERSION,
                input_tokens=prefix_tokens + suffix_counts[index], truncated=False,
                temperature=temperature,
                timings={"tokenization_s": tokenization,
                         "prefill_s": float(metrics.get("prefill_ms", 0)) / 1000,
                         "shared_prefill_s": float(metrics.get("prefill_ms", 0)) / 1000 if mode == "shared" else 0,
                         "scoring_s": float(metrics.get("questions_ms", 0)) / 1000,
                         "total_s": time.perf_counter() - started},
                mode=mode,
            ))
        return results

    def score_decision(self, decision: Decision, temperature=1.0, images=None):
        from ..api.schemas import ChoiceQuestion

        question = ChoiceQuestion(type="choice", instructions=decision.question,
                                  criteria={option.id: option.description for option in decision.options})
        return self.score_batch(decision.state, {"score": question}, temperature, images)[0]
