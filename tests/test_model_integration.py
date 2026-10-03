"""Opt-in CUDA integration; requires the Qwen3.5-4B weights and GPU."""
import os
import pytest
from openjev_semif.config import Settings
from openjev_semif.backends.registry import create_backend
from openjev_semif.scoring.base import Decision, Option
from openjev_semif.scoring.semif import SemIfScorer
from openjev_semif.runtime.shared_state import score_shared

@pytest.mark.skipif(os.getenv("RUN_MODEL_INTEGRATION") != "1", reason="set RUN_MODEL_INTEGRATION=1")
def test_qwen_sanity_and_shared():
    model = create_backend(Settings())
    rows = [
        Decision("A CUDA kernel test failed after a code change.", "What should happen next?",
                 [Option("logs", "Inspect the error logs"), Option("ignore", "Ignore the test failure")]),
        Decision("A CUDA kernel test failed after a code change.", "What validates a repair?",
                 [Option("retest", "Rerun the failing test"), Option("delete", "Delete the test")]),
    ]
    direct = [SemIfScorer(model).score(x) for x in rows]
    shared = score_shared(model, rows)
    assert direct[0].best == "logs"
    assert direct[1].best == "retest"
    for a, b in zip(direct, shared):
        assert [x["probability"] for x in a.options] == pytest.approx(
            [x["probability"] for x in b.options], abs=0.02)
