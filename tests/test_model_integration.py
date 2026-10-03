"""Opt-in CUDA integration; requires the Qwen3.5-4B weights and GPU."""
import os
import pytest
from openjev_semif.config import Settings
from openjev_semif.backends.registry import create_backend
from openjev_semif.scoring.base import Decision, Option
from openjev_semif.scoring.semif import SemIfScorer
from openjev_semif.scoring.likelihood import LikelihoodScorer
from openjev_semif.runtime.shared_state import score_shared
from openjev_semif.prompting.semif import encode, prompt
from openjev_semif.api.server import create_app
from fastapi.testclient import TestClient
from openjev_semif.client import OpenJEVClient

@pytest.mark.skipif(os.getenv("RUN_MODEL_INTEGRATION") != "1", reason="set RUN_MODEL_INTEGRATION=1")
def test_qwen_sanity_and_shared():
    model = create_backend(Settings())
    text = prompt(model.tokenizer, "evidence", "Choose", ["one", "two", "three", "four"])
    _, slots, _ = encode(model.tokenizer, text, 4, model.max_context)
    assert len(set(slots)) == 4
    assert [model.tokenizer.decode([i]) for i in slots] == list("ABCD")
    rows = [
        Decision("A CUDA kernel test failed after a code change.", "What should happen next?",
                 [Option("logs", "Inspect the error logs"), Option("ignore", "Ignore the test failure")]),
        Decision("A CUDA kernel test failed after a code change.", "What validates a repair?",
                 [Option("retest", "Rerun the failing test"), Option("delete", "Delete the test")]),
        Decision("A CUDA kernel test failed after a code change.", "Which action limits risk until fixed?",
                 [Option("rollback", "Disable the new kernel"), Option("ship", "Ship the failing kernel")]),
    ]
    likelihood = LikelihoodScorer(model).score(rows[0])
    text = f"State:\n{rows[0].state}\n\nQuestion:\n{rows[0].question}\n\nAnswer:\n"
    prefix = model.tokenizer.encode(text, add_special_tokens=False)
    for option, item in zip(rows[0].options, likelihood.options):
        continuation = model.tokenizer.encode(option.description, add_special_tokens=False)
        assert item["raw_score"] == pytest.approx(model.sequence_logprob(prefix, continuation), abs=0.1)
    direct = [SemIfScorer(model).score(x) for x in rows]
    shared = score_shared(model, rows)
    assert direct[0].best == "logs"
    assert direct[1].best == "retest"
    assert direct[2].best == "rollback"
    with TestClient(create_app(Settings(), model)) as client:
        with OpenJEVClient(session=client) as remote:
            assert remote.health().status == "ready"
        assert client.post("/score", json={"state": rows[0].state, "question": rows[0].question,
            "options": [{"id": x.id, "description": x.description} for x in rows[0].options]}).json()["best"] == "logs"
        system = client.post("/v1/systemone", json={"state": rows[0].state,
            "questions": {"next": {"type": "choice", "instructions": rows[0].question,
                "criteria": {x.id: x.description for x in rows[0].options}}}})
        assert system.status_code == 200, system.text
        assert system.json()["answers"]["next"]["choice"] == "logs"
    for a, b in zip(direct, shared):
        assert [x["probability"] for x in a.options] == pytest.approx(
            [x["probability"] for x in b.options], abs=0.02)
