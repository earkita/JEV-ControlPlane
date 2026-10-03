import hashlib
import math
import pytest
from openjev_semif.scoring.base import Decision, Option
from openjev_semif.scoring.semif import SemIfScorer
from openjev_semif.scoring.likelihood import LikelihoodScorer
from openjev_semif.scoring.calibration import probabilities
from openjev_semif.prompting.semif import encode, prompt
from openjev_semif.runtime.shared_state import score_shared


def decision(question="Choose"):
    return Decision("Long shared state " * 20, question,
                    [Option("a", "Alpha"), Option("b", "Beta"), Option("c", "Gamma")])


def test_manual_logits_softmax():
    p = probabilities([2.0, 1.0, 0.0])
    z = math.exp(2) + math.exp(1) + 1
    assert p == pytest.approx([math.exp(2)/z, math.exp(1)/z, 1/z])
    assert probabilities([2.0, 1.0], 2.0)[0] < probabilities([2.0, 1.0])[0]


def test_answer_slots_and_boundary(backend):
    text = prompt(backend.tokenizer, "state", "question", ["one", "two", "three", "four"])
    ids, slots, digest = encode(backend.tokenizer, text, 4, 1000)
    assert slots == [ord(x) for x in "ABCD"]
    assert digest == hashlib.sha256(text.encode()).hexdigest()
    assert len(ids) == len(text)


def test_boundary_rejection(backend):
    class MergingTokenizer:
        def encode(self, text, add_special_tokens=False):
            if text.endswith("A"): return [1]
            return [ord(c) for c in text]
        def decode(self, ids): return ''.join(chr(i) for i in ids)
    with pytest.raises(ValueError, match="answer"):
        encode(MergingTokenizer(), "prompt", 2, 100)


def test_no_truncation(backend):
    backend.max_context = 5
    with pytest.raises(ValueError, match="truncation disabled"):
        SemIfScorer(backend).score(decision())


def test_direct_and_shared_match(backend):
    rows = [decision("First?"), decision("Second?"), decision("Third?")]
    direct = [SemIfScorer(backend).score(row) for row in rows]
    shared = score_shared(backend, rows)
    assert backend.prefill_calls == 1
    for a, b in zip(direct, shared):
        assert a.best == b.best
        assert a.prompt_hash == b.prompt_hash
        assert [o["probability"] for o in a.options] == pytest.approx(
            [o["probability"] for o in b.options])
        assert b.mode == "shared"
        assert b.timings["shared_prefill_s"] >= 0


def test_distinct_likelihood_semantics(backend):
    result = LikelihoodScorer(backend).score(decision())
    assert result.scorer == "likelihood"
    assert result.best == "b"
    assert result.options[0]["raw_score"] == -5
