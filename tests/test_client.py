import json
import httpx
import pytest
from fastapi.testclient import TestClient
from openjev_semif.api.server import create_app
from openjev_semif.client import OpenJEVClient, OpenJEVHTTPError
from openjev_semif.cli import main


def test_client_calls_live_app_without_loading_model(backend, settings):
    with TestClient(create_app(settings, backend)) as session:
        with OpenJEVClient(session=session) as client:
            assert client.health().status == "ready"
            score = client.score({"state": "CI failed", "question": "What next?",
                "options": [{"id": "inspect", "description": "Inspect logs"},
                            {"id": "ignore", "description": "Ignore"}]})
            assert score.best == "inspect"
            assert score.options[0].probability > score.options[1].probability
            answer = client.systemone({"state": "CI failed", "questions": {
                "next": {"type": "choice", "instructions": "What next?",
                         "criteria": {"inspect": "Inspect logs", "ignore": "Ignore"}}}})
            assert answer.answers["next"].choice == "inspect"
            assert answer.model_revision == backend.model_revision
        assert not session.is_closed


def test_client_surfaces_http_error(backend, settings):
    backend.max_context = 4
    with TestClient(create_app(settings, backend)) as session:
        with OpenJEVClient(session=session) as client:
            with pytest.raises(OpenJEVHTTPError, match="truncation disabled") as error:
                client.score({"state": "too long", "question": "Why?", "options": ["A", "B"]})
            assert error.value.status_code == 400


def test_client_cli_does_not_create_backend(monkeypatch, capsys):
    def handle(request):
        assert request.url.path == "/health"
        return httpx.Response(200, json={"model":"fake", "backend":"torch", "device":"cpu",
            "dtype":"float32", "scorer":"semif", "status":"ready", "model_revision":"abc"})
    session = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle))
    monkeypatch.setattr("openjev_semif.client.httpx.Client", lambda **kwargs: session)
    def fail(*args): raise AssertionError("client CLI tried to load model")
    monkeypatch.setattr("openjev_semif.backends.registry.create_backend", fail)
    main(["client", "health", "--url", "http://test"])
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
    session.close()


def test_client_api_key(monkeypatch, backend, settings):
    monkeypatch.setenv("OPENJEV_API_KEY", "test-key")
    request = {"state": "CI failed", "questions": {"next": {
        "type": "choice", "instructions": "What next?",
        "criteria": {"inspect": "Inspect logs", "ignore": "Ignore"}}}}
    with TestClient(create_app(settings, backend)) as session:
        with OpenJEVClient(session=session) as anonymous:
            with pytest.raises(OpenJEVHTTPError) as error:
                anonymous.systemone(request)
            assert error.value.status_code == 401
        with OpenJEVClient(session=session, api_key="test-key") as authorized:
            assert authorized.systemone(request).answers["next"].choice == "inspect"
