import math

import httpx
import pytest
from fastapi.testclient import TestClient

from openjev_semif.api.server import create_app
from openjev_semif.backends.llama_cpp import BackendUnavailable, LlamaCppBackend
from openjev_semif.config import Settings
from openjev_semif.scoring.base import Decision, Option


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "<assistant>"


def setup_backend(tmp_path, *, missing_letter=False, max_context=8192):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"test-only")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    calls = []

    def handle(request):
        calls.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": str(model),
                "default_generation_settings": {"n_ctx": 8192}})
        payload = __import__("json").loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [ord(x) for x in payload["content"]]})
        if request.url.path == "/completion":
            rows = [{"id": ord("A"), "logprob": 2.0}]
            if not missing_letter:
                rows.append({"id": ord("B"), "logprob": 1.0})
            return httpx.Response(200, json={"completion_probabilities":
                [{"top_logprobs": rows}], "truncated": False,
                "timings": {"prompt_ms": 30, "predicted_ms": 2}})
        raise AssertionError(request.url.path)

    settings = Settings(model=str(model), revision="a" * 40, backend="llama_cpp",
        backend_url="http://127.0.0.1:8091", tokenizer_path=str(tokenizer_dir),
        tokenizer_revision="b" * 40, device="cuda", dtype="q4_k_m",
        max_context=max_context)
    client = httpx.Client(transport=httpx.MockTransport(handle),
                          base_url=settings.backend_url)
    return settings, LlamaCppBackend(settings, client=client, tokenizer=CharTokenizer()), calls


def test_gguf_readout_uses_exact_letter_ids_and_softmax(tmp_path):
    _, backend, calls = setup_backend(tmp_path)
    result = backend.score_decision(Decision("CI failed", "Next action?",
        [Option("inspect", "Inspect logs"), Option("ignore", "Ignore error")]))
    assert result.best == "inspect"
    assert result.options[0]["probability"] == pytest.approx(math.e / (math.e + 1))
    assert result.input_tokens > 0
    assert len(result.prompt_hash) == 64
    payload = __import__("json").loads(calls[-1].content)
    assert payload["n_predict"] == 1
    assert payload["n_probs"] == 256


def test_gguf_rejects_incomplete_logits_and_truncation(tmp_path):
    _, backend, _ = setup_backend(tmp_path, missing_letter=True)
    decision = Decision("state", "question", [Option("a", "first"), Option("b", "second")])
    with pytest.raises(BackendUnavailable, match="absent"):
        backend.score_decision(decision)
    backend.max_context = 4
    with pytest.raises(ValueError, match="truncation disabled"):
        backend.score_decision(decision)


def test_gguf_api_direct_and_unsupported_modes(tmp_path):
    settings, backend, _ = setup_backend(tmp_path)
    with TestClient(create_app(settings, backend)) as client:
        health = client.get("/health").json()
        assert health["backend"] == "llama_cpp"
        assert health["default_temperature"] == 1.0
        request = {"state": "CI failed", "question": "What next?",
                   "options": [{"id": "inspect", "description": "Inspect logs"},
                               {"id": "ignore", "description": "Ignore error"}]}
        assert client.post("/score", json=request).json()["best"] == "inspect"
        assert client.post("/score", json={**request, "scorer": "likelihood"}).status_code == 400
        batch = {"state": "CI failed", "questions": {"next": {"type": "choice",
                 "instructions": "What next?", "criteria": {"inspect": "Inspect logs",
                 "ignore": "Ignore error"}}}}
        assert client.post("/v1/systemone", json=batch).json()["answers"]["next"]["choice"] == "inspect"
        assert client.post("/v1/systemone", json={**batch, "mode": "shared"}).status_code == 400


def test_gguf_typed_noul_and_score_prompt_and_temperature(tmp_path):
    settings, backend, calls = setup_backend(tmp_path)
    with TestClient(create_app(settings, backend)) as client:
        batch = {"state": "The test failed", "questions": {
            "truth": {"type": "noul", "instructions": "Did the test fail?"},
            "rating": {"type": "score", "instructions": "How urgent?",
                       "criteria": ["low", "high"]}}}
        response = client.post("/v1/systemone", json=batch)
        assert response.status_code == 200
        answers = response.json()["answers"]
        assert answers["truth"]["temperature"] == pytest.approx(settings.noul_temperature)
        assert answers["truth"]["noul"] == pytest.approx(
            math.exp(1 / settings.noul_temperature) /
            (1 + math.exp(1 / settings.noul_temperature)))
        assert answers["rating"]["temperature"] == 1.0
    completions = [__import__("json").loads(call.content)["prompt"] for call in calls
                   if call.url.path == "/completion"]
    assert "[A] yes: The statement is true." in completions[0]
    assert "[B] no: The statement is false." in completions[0]
    assert "Rate along the ordered levels below (lowest first)." in completions[1]
