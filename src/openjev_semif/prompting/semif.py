"""Stable SemIf decision prompt and exact answer-token validation."""
from __future__ import annotations
import hashlib
import json

LETTERS = "ABCDEFGHIJKLMNOP"
PROMPT_VERSION = "semif-chat-v1"
SYSTEM = "Choose exactly one listed option using the evidence and criterion. Reply with only its uppercase letter."


def state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, allow_nan=False, sort_keys=True)


def render(state, question: str, descriptions: list[str]) -> str:
    if not question.strip(): raise ValueError("question must be nonempty")
    if not 2 <= len(descriptions) <= len(LETTERS): raise ValueError("SemIf supports 2-16 options")
    lines = "\n".join(f"{LETTERS[i]}) {d}" for i, d in enumerate(descriptions))
    return f"State:\n{state_text(state)}\n\nQuestion:\n{question}\n\nOptions:\n{lines}\n\nAnswer:"


def prompt(tokenizer, state, question: str, descriptions: list[str]) -> str:
    user = render(state, question, descriptions)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    return SYSTEM + "\n\n" + user + "\n"


def encode(tokenizer, text: str, count: int, max_context: int):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids: raise ValueError("prompt tokenized to no input")
    if len(ids) > max_context:
        raise ValueError(f"prompt has {len(ids)} tokens; max_context={max_context}; truncation disabled")
    slots = []
    for letter in LETTERS[:count]:
        token = tokenizer.encode(letter, add_special_tokens=False)
        if len(token) != 1 or tokenizer.decode(token) != letter:
            raise ValueError(f"answer letter {letter} is not one exact token")
        if tokenizer.encode(text + letter, add_special_tokens=False) != ids + token:
            raise ValueError(f"answer boundary changes tokenization for {letter}")
        slots.append(token[0])
    if len(set(slots)) != len(slots): raise ValueError("answer tokens collide")
    return ids, slots, hashlib.sha256(text.encode()).hexdigest()
