import pytest
from openjev_semif.scoring.calibration import (
    CalibrationProfile, effective_temperature, fit_temperature, probabilities,
)


def test_profile_round_trip_and_match(tmp_path, backend, settings):
    path = tmp_path / "profile.json"
    CalibrationProfile(backend.model_name, backend.model_revision, "ci", "semif", 2.0).save(path)
    configured = settings.__class__(**{**settings.__dict__, "calibration_profile": str(path)})
    assert effective_temperature(configured, backend) == 2.0
    wrong = settings.__class__(**{**configured.__dict__, "scorer": "likelihood"})
    with pytest.raises(ValueError, match="does not match"):
        effective_temperature(wrong, backend)


def test_fit_temperature_softens_overconfident_logits():
    rows = [([8.0, 0.0], 0), ([8.0, 0.0], 0), ([8.0, 0.0], 1)]
    temperature = fit_temperature(rows)
    assert temperature > 1
    assert probabilities(rows[0][0], temperature)[0] < probabilities(rows[0][0])[0]
    assert fit_temperature([([2.0, 0.0], 0)]) > 0
