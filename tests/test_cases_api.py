from fastapi.testclient import TestClient

from openjev_semif.api.server import create_app
from openjev_semif.client import OpenJEVClient


def payload():
    return {"cases": [
        {"id": "ci-001", "state": {"build": "failed"},
         "provenance": {"generator_model": "fixture-generator-v1"},
         "questions": [
             {"id": "next", "question": "What next?", "options": [
                 {"id": "inspect", "description": "Inspect logs"},
                 {"id": "ignore", "description": "Ignore failure"}],
              "expected_option": "inspect"},
             {"id": "verify", "question": "How to verify?", "options": [
                 {"id": "rerun", "description": "Run tests again"},
                 {"id": "skip", "description": "Skip tests"}],
              "expected_option": "skip"}]},
        {"id": "ci-002", "state": "Another build failed", "questions": [
            {"id": "investigate", "question": "What next?", "options": [
                {"id": "inspect", "description": "Inspect logs"},
                {"id": "ignore", "description": "Ignore failure"}]}]},
    ]}


def test_validate_cases_without_inference(backend, settings):
    def fail(*args):
        raise AssertionError("validation invoked inference")
    backend.forward = fail
    with TestClient(create_app(settings, backend)) as app:
        response = app.post("/v1/cases/validate", json=payload())
    assert response.status_code == 200
    assert response.json() == {"valid": True, "case_count": 2,
                               "question_count": 3, "labelled_count": 2}


def test_evaluate_cases_reuses_systemone_and_reports_label_matches(backend, settings):
    with TestClient(create_app(settings, backend)) as app:
        with OpenJEVClient(session=app) as client:
            response = client.evaluate_cases(payload())
    assert response.model_revision == backend.model_revision
    assert response.summary.case_count == 2
    assert response.summary.question_count == 3
    assert response.summary.labelled_count == 2
    assert response.summary.matched_count == 1
    assert response.summary.accuracy == 0.5
    first = response.cases[0]
    assert first.provenance == {"generator_model": "fixture-generator-v1"}
    assert first.questions[0].matched is True
    assert first.questions[0].answer.choice == "inspect"
    assert first.questions[1].matched is False
    assert response.cases[1].questions[0].matched is None
    assert response.usage.input_tokens > 0


def test_case_schema_rejects_inconsistent_generated_fixtures(backend, settings):
    with TestClient(create_app(settings, backend)) as app:
        bad = payload()
        bad["cases"][0]["questions"][0]["expected_option"] = "missing"
        response = app.post("/v1/cases/validate", json=bad)
        assert response.status_code == 422
        assert "expected_option must match" in response.text

        bad = payload()
        bad["cases"][0]["questions"][0]["options"][1]["id"] = "inspect"
        response = app.post("/v1/cases/evaluate", json=bad)
        assert response.status_code == 422
        assert "option IDs must be unique" in response.text

        bad = payload()
        bad["cases"][1]["id"] = "ci-001"
        assert app.post("/v1/cases/validate", json=bad).status_code == 422


def test_case_endpoints_use_systemone_api_key(monkeypatch, backend, settings):
    monkeypatch.setenv("OPENJEV_API_KEY", "test-key")
    with TestClient(create_app(settings, backend)) as app:
        assert app.post("/v1/cases/validate", json=payload()).status_code == 401
        assert app.post("/v1/cases/evaluate", json=payload()).status_code == 401
        with OpenJEVClient(session=app, api_key="test-key") as client:
            assert client.validate_cases(payload()).valid is True
            assert client.evaluate_cases(payload()).summary.question_count == 3
