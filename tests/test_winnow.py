import base64
import json
import httpx
from fastapi.testclient import TestClient

from openjev_semif.api.server import create_app
from openjev_semif.backends.winnow import WinnowBackend
from openjev_semif.config import Settings


PNG = "data:image/png;base64," + base64.b64encode(
    b"\x89PNG\r\n\x1a\n" + b"test-image"
).decode()


def test_winnow_vision_decisions_and_shared_calls(tmp_path):
    model = tmp_path / "Winnow-Q8.gguf"
    projector = tmp_path / "mmproj.gguf"
    model.write_bytes(b"model")
    projector.write_bytes(b"projector")
    settings = Settings(model=str(model), revision="a" * 40, tokenizer_revision="b" * 40,
                        projector_path=str(projector), backend="winnow",
                        backend_url="http://localhost:8091", device="cuda", dtype="q8_0")
    calls = []

    def native(request):
        calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": str(model), "modalities": {"vision": True},
                                             "default_generation_settings": {"n_ctx": 4096}})
        body = json.loads(request.content)
        questions = body["questions"]
        if request.url.path == "/v1/winnow/inspect":
            assert body["winnow"]["include_token_ids"] is True
            return httpx.Response(200, json={"prefix_tokens": 100, "request_prefix_tokens": 0,
                "suffix_tokens": [20 for _ in questions], "prefix_token_ids": [1, 2],
                "request_prefix_token_ids": [],
                "suffix_token_ids": [[i + 3] for i in range(len(questions))]})
        assert request.url.path == "/v1/systemone"
        answers = {}
        for key, question in questions.items():
            if question["type"] == "noul":
                answers[key] = {"type": "noul", "noul": 0.8,
                                "winnow": {"logits": [0.0, 1.4]}}
            elif question["type"] == "score":
                answers[key] = {"type": "score", "score": 0.2,
                    "probabilities": {"0": 0.8, "1": 0.2},
                    "winnow": {"logits": [1.4, 0.0]}}
            else:
                keys = list(question["criteria"])
                answers[key] = {"type": "choice", "choice": keys[0],
                    "probabilities": {keys[0]: 0.8, keys[1]: 0.2},
                    "winnow": {"logits": [1.4, 0.0]}}
        return httpx.Response(200, json={"model": "Winnow-12B", "answers": answers,
            "usage": {"input_tokens": 120, "output_tokens": 0},
            "winnow": {"prefill_ms": 10, "questions_ms": 3}})

    transport = httpx.MockTransport(native)
    remote = httpx.Client(base_url="http://localhost:8091", transport=transport)
    backend = WinnowBackend(settings, client=remote)
    with TestClient(create_app(settings, backend)) as client:
        assert client.get("/health").json()["vision"] is True
        score = client.post("/score", json={"state": "What is shown?", "images": [PNG],
            "question": "What color?", "options": [
                {"id": "red", "description": "Red"}, {"id": "blue", "description": "Blue"}]})
        assert score.status_code == 200, score.text
        assert score.json()["best"] == "red"
        assert score.json()["options"][0]["raw_score"] == 1.4
        assert len(score.json()["prompt_hash"]) == 64
        request = {"state": "Look at the photo", "images": [PNG], "mode": "shared",
            "questions": {
                "color": {"type": "choice", "instructions": "What color?",
                          "criteria": {"red": "Red", "blue": "Blue"}},
                "visible": {"type": "noul", "instructions": "Is an object visible?"}}}
        before = calls.count("/v1/systemone")
        system = client.post("/v1/systemone", json=request)
        assert system.status_code == 200, system.text
        assert calls.count("/v1/systemone") == before + 1
        assert system.json()["answers"]["color"]["choice"] == "red"
        assert system.json()["answers"]["visible"]["noul"] == 0.8
        assert system.json()["answers"]["color"]["mode"] == "shared"
        assert system.json()["usage"]["output_tokens"] == 0
        cases = client.post("/v1/cases/evaluate", json={"cases": [{"id": "photo-red",
            "state": "Classify this photo", "images": [PNG], "questions": [{
                "id": "color", "question": "What color?", "options": [
                    {"id": "red", "description": "Red"},
                    {"id": "blue", "description": "Blue"}], "expected_option": "red"}]}]})
        assert cases.status_code == 200, cases.text
        assert cases.json()["summary"]["matched_count"] == 1
        invalid = client.post("/score", json={"state": "x", "images": ["file:///etc/passwd"],
            "question": "?", "options": ["yes", "no"]})
        assert invalid.status_code == 422
        backend.max_context = 100
        overflow = client.post("/score", json={"state": "x", "images": [PNG],
            "question": "?", "options": ["yes", "no"]})
        assert overflow.status_code == 400
        assert "truncation disabled" in overflow.text
