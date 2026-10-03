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
