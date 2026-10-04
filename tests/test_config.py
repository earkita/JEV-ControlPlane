from openjev_semif.cli import load_settings, parser


def test_launcher_config_precedence_and_home_expansion(monkeypatch, tmp_path):
    config = tmp_path / "serve.json"
    config.write_text('{"model":"~/models/qwen", "port":8000, "temperature":1.0}')
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PORT", "8001")
    monkeypatch.setenv("TEMPERATURE", "1.5")

    args = parser().parse_args(["serve", "--config", str(config), "--port", "8002"])
    settings = load_settings(args)

    assert settings.model == str(tmp_path / "models/qwen")
    assert settings.port == 8002
    assert settings.temperature == 1.5


def test_launcher_config_can_be_overridden_with_cli(monkeypatch, tmp_path):
    config = tmp_path / "serve.json"
    config.write_text('{"model":"~/models/qwen", "device":"cuda"}')
    monkeypatch.setenv("MODEL", "env/model")

    args = parser().parse_args(["serve", "--config", str(config),
                                "--model", "cli/model", "--device", "cpu"])
    settings = load_settings(args)

    assert settings.model == "cli/model"
    assert settings.device == "cpu"
