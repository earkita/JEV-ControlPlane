"""Single-model HTTP service with startup/shutdown lifecycle."""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header
from .schemas import ScoreRequest, SystemOneRequest, HealthResponse, OptionIn
from .systemone import decision_for, answer_for
from ..backends.registry import create_backend
from ..config import Settings
from ..scoring.base import Decision, Option
from ..scoring.likelihood import LikelihoodScorer
from ..scoring.semif import SemIfScorer
from ..scoring.calibration import effective_temperature
from ..runtime.shared_state import score_shared


def create_app(settings: Settings | None = None, backend=None) -> FastAPI:
    settings = settings or Settings.from_env()
    state = {"backend": backend, "default_temperature": settings.temperature}

    @asynccontextmanager
    async def lifespan(app):
        if state["backend"] is None: state["backend"] = create_backend(settings)
        state["default_temperature"] = effective_temperature(settings, state["backend"])
        yield
        state["backend"] = None

    app = FastAPI(title="OpenJEV-SemIf", version="0.1.0", lifespan=lifespan)

    def get_backend():
        if state["backend"] is None: raise HTTPException(503, "model loading")
        return state["backend"]

    def scorer_for(name, model):
        return SemIfScorer(model) if name == "semif" else LikelihoodScorer(model)

    @app.get("/health", response_model=HealthResponse)
    def health():
        model = state["backend"]
        return HealthResponse(model=settings.model, backend=settings.backend,
            device=model.device if model else settings.device, dtype=model.dtype if model else settings.dtype,
            scorer=settings.scorer, status="ready" if model else "loading",
            model_revision=model.model_revision if model else None)

    @app.post("/score")
    def score(req: ScoreRequest):
        model = get_backend()
        options = [Option(x.id, x.description) if isinstance(x, OptionIn)
                   else Option(x, x) for x in req.options]
        if len({x.id for x in options}) != len(options): raise HTTPException(400, "option ids must be unique")
        try:
            result = scorer_for(req.scorer or settings.scorer, model).score(
                Decision(req.state, req.question, options), req.temperature or state["default_temperature"])
            return result.as_dict()
        except ValueError as exc: raise HTTPException(400, str(exc)) from exc

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest, authorization: str | None = Header(default=None)):
        api_key = os.environ.get("OPENJEV_API_KEY")
        if api_key and authorization != f"Bearer {api_key}": raise HTTPException(401, "invalid API key")
        model = get_backend()
        scorer = req.scorer or settings.scorer
        if req.mode == "shared" and scorer != "semif":
            raise HTTPException(400, "shared mode currently supports semif only")
        questions = list(req.questions.items())
        decisions = [decision_for(req.state, q) for _, q in questions]
        try:
            temperature = req.temperature or state["default_temperature"]
            if req.mode == "shared": results = score_shared(model, decisions, temperature)
            else: results = [scorer_for(scorer, model).score(d, temperature) for d in decisions]
        except ValueError as exc: raise HTTPException(400, str(exc)) from exc
        answers = {qid: answer_for(q, result) for (qid, q), result in zip(questions, results)}
        return {"model": model.model_name, "model_revision": model.model_revision,
                "answers": answers, "usage": {"input_tokens": sum(r.input_tokens for r in results),
                                              "output_tokens": 0}}
    return app
