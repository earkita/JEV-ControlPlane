from fastapi.testclient import TestClient
from openjev_semif.api.server import create_app


def test_health_score_systemone(backend, settings):
    with TestClient(create_app(settings, backend)) as client:
        health = client.get('/health')
        assert health.status_code == 200
        assert health.json()["status"] == "ready"
        assert health.json()["model_revision"] == backend.model_revision
        score = client.post('/score', json={"state":"CI failed", "question":"What next?",
            "options":[{"id":"inspect","description":"Inspect logs"},
                       {"id":"ignore","description":"Ignore"}]})
        assert score.status_code == 200, score.text
        body = score.json()
        assert body["best"] == "inspect"
        assert len(body["prompt_hash"]) == 64
        assert sum(o["probability"] for o in body["options"]) == 1
        system = client.post('/v1/systemone', json={"state":"CI failed", "mode":"shared",
            "questions":{"next_action":{"type":"choice", "instructions":"What next?",
                "criteria":{"inspect":"Inspect logs","ignore":"Ignore"}},
                "severity":{"type":"score", "instructions":"How severe?",
                 "criteria":["low","high"]}}})
        assert system.status_code == 200, system.text
        data = system.json()
        assert data["answers"]["next_action"]["choice"] == "inspect"
        assert data["answers"]["severity"]["type"] == "score"
        assert backend.prefill_calls == 1


def test_api_rejects_overlong_prompt(backend, settings):
    backend.max_context = 4
    with TestClient(create_app(settings, backend)) as client:
        response = client.post('/score', json={"state":"too long", "question":"What?",
            "options":["A", "B"]})
        assert response.status_code == 400
        assert "truncation disabled" in response.text
