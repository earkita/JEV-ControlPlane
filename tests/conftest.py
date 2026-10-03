import pytest
from openjev_semif.config import Settings

class CharTokenizer:
    chat_template = None
    def encode(self, text, add_special_tokens=False): return [ord(c) for c in text]
    def decode(self, ids): return ''.join(chr(i) for i in ids)

class FakeBackend:
    model_name = "fake/model"
    model_revision = "a" * 40
    tokenizer_revision = "a" * 40
    device = "cpu"
    dtype = "float32"
    max_context = 8192
    tokenizer = CharTokenizer()
    prefill_calls = 0
    def forward(self, ids):
        scores = [0.0] * 256
        scores[ord('A')] = 2.0
        scores[ord('B')] = 1.0
        scores[ord('C')] = 0.0
        return scores
    def prefill(self, ids):
        self.prefill_calls += 1
        return ids, len(ids)
    def get_last_logits(self, cache, suffix): return self.forward(cache[0] + suffix)
    def sequence_logprob(self, prefix, continuation): return -len(continuation)

@pytest.fixture
def backend(): return FakeBackend()

@pytest.fixture
def settings(): return Settings(model="fake/model", device="cpu", dtype="float32")
