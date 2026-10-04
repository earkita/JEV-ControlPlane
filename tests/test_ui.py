from hashlib import sha256
from fastapi.testclient import TestClient
from openjev_semif.api.server import create_app


def test_ui_is_served_with_api(backend, settings):
    with TestClient(create_app(settings, backend)) as client:
        home = client.get("/", follow_redirects=False)
        assert home.status_code == 302
        assert home.headers["location"] == "/ui"
        page = client.get("/ui")
        assert page.status_code == 200
        assert "text/html" in page.headers["content-type"]
        assert 'id="decision-form"' in page.text
        assert 'id="panel-batch"' in page.text
        assert 'id="panel-json"' in page.text
        assert 'id="json-input"' in page.text
        assert page.headers["cache-control"] == "no-store"
        css = client.get("/ui/assets/style.css")
        script = client.get("/ui/assets/app.js")
        assert css.status_code == script.status_code == 200
        assert "text/css" in css.headers["content-type"]
        assert "javascript" in script.headers["content-type"]
        assert f'href="/ui/assets/style.css?v={sha256(css.content).hexdigest()[:12]}"' in page.text
        assert f'src="/ui/assets/app.js?v={sha256(script.content).hexdigest()[:12]}"' in page.text
        assert 'fetch(request.path' in script.text
        assert 'renderCases(body, request.body)' in script.text
        assert client.get("/docs").status_code == 200
        assert client.get("/health").json()["status"] == "ready"
