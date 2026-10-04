"""HTTP contracts for direct scores and typed SystemOne questions."""
from __future__ import annotations
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class CaseOption(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class CaseQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    options: list[CaseOption] = Field(min_length=2, max_length=16)
    expected_option: str | None = None

    @model_validator(mode="after")
    def validate_options(self):
        ids = [option.id for option in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError("option IDs must be unique within a question")
        if self.expected_option is not None and self.expected_option not in ids:
            raise ValueError("expected_option must match an option ID")
        return self


class DecisionCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    state: str | dict | list
    questions: list[CaseQuestion] = Field(min_length=1, max_length=16)
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_questions(self):
        ids = [question.id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question IDs must be unique within a case")
        return self


class CaseBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cases: list[DecisionCase] = Field(min_length=1, max_length=16)
    scorer: Literal["semif", "likelihood"] | None = None
    mode: Literal["direct", "shared"] = "direct"
    temperature: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_batch(self):
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case IDs must be unique")
        if sum(len(case.questions) for case in self.cases) > 64:
            raise ValueError("a batch may contain at most 64 questions")
        if self.mode == "shared" and any(len(case.questions) < 2 for case in self.cases):
            raise ValueError("shared mode requires at least two questions per case")
        return self


class CaseValidationResponse(BaseModel):
    valid: Literal[True] = True
    case_count: int
    question_count: int
    labelled_count: int


class CaseQuestionResult(BaseModel):
    id: str
    expected_option: str | None
    matched: bool | None
    answer: ChoiceAnswerOut


class CaseResult(BaseModel):
    id: str
    provenance: dict[str, str]
    questions: list[CaseQuestionResult]


class CaseSummary(BaseModel):
    case_count: int
    question_count: int
    labelled_count: int
    matched_count: int
    accuracy: float | None


class CaseBatchResponse(BaseModel):
    model: str
    model_revision: str
    cases: list[CaseResult]
    summary: CaseSummary
    usage: UsageOut
