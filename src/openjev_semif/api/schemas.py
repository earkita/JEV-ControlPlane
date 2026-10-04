"""HTTP contracts for direct scores and typed SystemOne questions."""
from __future__ import annotations
from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field, model_validator

class OptionIn(BaseModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)

class ScoreRequest(BaseModel):
    state: str | dict | list
    question: str = Field(min_length=1)
    options: list[str | OptionIn] = Field(min_length=2, max_length=16)
    scorer: Literal["semif", "likelihood"] | None = None
    temperature: float | None = Field(default=None, gt=0)

class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Any = None
    criteria: dict[str, Any] = Field(min_length=2, max_length=16)

class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Any = None
    criteria: list[Any] = Field(min_length=2, max_length=16)

class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Any = None
    criteria: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_keys(self):
        if self.criteria and not set(self.criteria) <= {"true", "false"}:
            raise ValueError("noul criteria only supports true and false")
        return self

Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]

class SystemOneRequest(BaseModel):
    state: str | dict | list
    questions: dict[str, Question] = Field(min_length=1)
    scorer: Literal["semif", "likelihood"] | None = None
    mode: Literal["direct", "shared"] = "direct"
    temperature: float | None = Field(default=None, gt=0)

class HealthResponse(BaseModel):
    model: str
    backend: str
    device: str
    dtype: str
    scorer: str
    status: Literal["ready", "loading"]
    model_revision: str | None = None
    default_temperature: float = 1.0

class OptionScoreOut(BaseModel):
    option: str
    description: str
    raw_score: float
    probability: float
    n_tokens: int | None = None

class ScoreResponse(BaseModel):
    best: str
    best_index: int
    options: list[OptionScoreOut]
    confidence: float
    scorer: Literal["semif", "likelihood"]
    model: str
    model_revision: str
    tokenizer_revision: str
    prompt_hash: str
    prompt_version: str
    input_tokens: int
    truncated: bool
    temperature: float
    timings: dict[str, float]
    mode: Literal["direct", "shared"]

class ChoiceAnswerOut(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float
    timings: dict[str, float]
    scorer: str
    model_revision: str
    tokenizer_revision: str
    prompt_hash: str
    input_tokens: int
    temperature: float
    truncated: bool
    mode: str

class ScoreAnswerOut(BaseModel):
    type: Literal["score"]
    score: float
    legend: dict[str, Any]
    probabilities: dict[str, float]
    confidence: float
    timings: dict[str, float]
    scorer: str
    model_revision: str
    tokenizer_revision: str
    prompt_hash: str
    input_tokens: int
    temperature: float
    truncated: bool
    mode: str

class NoulAnswerOut(BaseModel):
    type: Literal["noul"]
    noul: float
    probabilities: dict[str, float]
    confidence: float
    timings: dict[str, float]
    scorer: str
    model_revision: str
    tokenizer_revision: str
    prompt_hash: str
    input_tokens: int
    temperature: float
    truncated: bool
    mode: str

AnswerOut = Annotated[ChoiceAnswerOut | ScoreAnswerOut | NoulAnswerOut, Field(discriminator="type")]

class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int

class SystemOneResponse(BaseModel):
    model: str
    model_revision: str
    answers: dict[str, AnswerOut]
    usage: UsageOut
