"""Single-model HTTP service with startup/shutdown lifecycle."""
from __future__ import annotations
import os
from hashlib import sha256
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from .schemas import (ScoreRequest, ScoreResponse, SystemOneRequest, SystemOneResponse,
                      HealthResponse, OptionIn, CaseBatchRequest, CaseBatchResponse,
                      CaseValidationResponse, ChoiceQuestion)
from .systemone import decision_for, answer_for
from ..backends.registry import create_backend
from ..backends.llama_cpp import BackendUnavailable
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
        if hasattr(state["backend"], "close"): state["backend"].close()
        state["backend"] = None

    app = FastAPI(title="OpenJEV-SemIf", version="0.1.0", lifespan=lifespan)
    web_dir = Path(__file__).resolve().parent.parent / "web"
    app.mount("/ui/assets", StaticFiles(directory=web_dir), name="ui-assets")

    @app.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/ui", status_code=302)

    @app.get("/ui", include_in_schema=False)
    def ui():
        markup = (web_dir / "index.html").read_text(encoding="utf-8")
        for asset in ("style.css", "app.js"):
            version = sha256((web_dir / asset).read_bytes()).hexdigest()[:12]
            markup = markup.replace(f'/ui/assets/{asset}"', f'/ui/assets/{asset}?v={version}"')
        return HTMLResponse(markup, headers={"Cache-Control": "no-store"})

    def get_backend():
        if state["backend"] is None: raise HTTPException(503, "model loading")
        return state["backend"]

    def scorer_for(name, model):
        return SemIfScorer(model) if name == "semif" else LikelihoodScorer(model)

    def require_api_key(authorization):
        api_key = os.environ.get("OPENJEV_API_KEY")
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(401, "invalid API key")

    @app.get("/health", response_model=HealthResponse)
    def health():
        model = state["backend"]
        return HealthResponse(model=settings.model, backend=settings.backend,
            device=model.device if model else settings.device, dtype=model.dtype if model else settings.dtype,
            scorer=settings.scorer, status="ready" if model else "loading",
            model_revision=model.model_revision if model else None,
            default_temperature=state["default_temperature"])

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest):
        model = get_backend()
        options = [Option(x.id, x.description) if isinstance(x, OptionIn)
                   else Option(x, x) for x in req.options]
        if len({x.id for x in options}) != len(options): raise HTTPException(400, "option ids must be unique")
        try:
            scorer = req.scorer or settings.scorer
            decision = Decision(req.state, req.question, options)
            temperature = req.temperature or state["default_temperature"]
            if hasattr(model, "score_decision"):
                if scorer != "semif": raise ValueError("llama_cpp backend supports semif only")
                result = model.score_decision(decision, temperature)
            else:
                result = scorer_for(scorer, model).score(decision, temperature)
            return result.as_dict()
        except ValueError as exc: raise HTTPException(400, str(exc)) from exc
        except BackendUnavailable as exc: raise HTTPException(502, str(exc)) from exc

    def run_systemone(req: SystemOneRequest):
        model = get_backend()
        scorer = req.scorer or settings.scorer
        if hasattr(model, "score_decision") and scorer != "semif":
            raise HTTPException(400, "llama_cpp backend supports semif only")
        if hasattr(model, "score_decision") and req.mode == "shared":
            raise HTTPException(400, "shared mode is not yet verified for llama_cpp backend")
        if req.mode == "shared" and scorer != "semif":
            raise HTTPException(400, "shared mode currently supports semif only")
        questions = list(req.questions.items())
        decisions = [decision_for(req.state, q) for _, q in questions]
        try:
            temperature = req.temperature or state["default_temperature"]
            if req.mode == "shared": results = score_shared(model, decisions, temperature)
            elif hasattr(model, "score_decision"):
                results = [model.score_decision(d, temperature, q.type)
                           for (_, q), d in zip(questions, decisions)]
            else: results = [scorer_for(scorer, model).score(d, temperature) for d in decisions]
        except ValueError as exc: raise HTTPException(400, str(exc)) from exc
        except BackendUnavailable as exc: raise HTTPException(502, str(exc)) from exc
        answers = {qid: answer_for(q, result) for (qid, q), result in zip(questions, results)}
        return {"model": model.model_name, "model_revision": model.model_revision,
                "answers": answers, "usage": {"input_tokens": sum(r.input_tokens for r in results),
                                              "output_tokens": 0}}

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def systemone(req: SystemOneRequest, authorization: str | None = Header(default=None)):
        require_api_key(authorization)
        return run_systemone(req)

    @app.post("/v1/cases/validate", response_model=CaseValidationResponse)
    def validate_cases(req: CaseBatchRequest, authorization: str | None = Header(default=None)):
        require_api_key(authorization)
        return {"valid": True, "case_count": len(req.cases),
                "question_count": sum(len(case.questions) for case in req.cases),
                "labelled_count": sum(q.expected_option is not None
                                      for case in req.cases for q in case.questions)}

    @app.post("/v1/cases/evaluate", response_model=CaseBatchResponse)
    def evaluate_cases(req: CaseBatchRequest, authorization: str | None = Header(default=None)):
        require_api_key(authorization)
        get_backend()
        cases = []
        input_tokens = 0
        model_name = None
        model_revision = None
        labelled_count = 0
        matched_count = 0
        for case in req.cases:
            questions = {q.id: ChoiceQuestion(type="choice", instructions=q.question,
                criteria={option.id: option.description for option in q.options})
                for q in case.questions}
            result = run_systemone(SystemOneRequest(state=case.state, questions=questions,
                scorer=req.scorer, mode=req.mode, temperature=req.temperature))
            model_name = result["model"]
            model_revision = result["model_revision"]
            input_tokens += result["usage"]["input_tokens"]
            evaluations = []
            for q in case.questions:
                answer = result["answers"][q.id]
                matched = None if q.expected_option is None else answer["choice"] == q.expected_option
                if matched is not None:
                    labelled_count += 1
                    matched_count += int(matched)
                evaluations.append({"id": q.id, "expected_option": q.expected_option,
                                    "matched": matched, "answer": answer})
            cases.append({"id": case.id, "provenance": case.provenance,
                          "questions": evaluations})
        question_count = sum(len(case.questions) for case in req.cases)
        return {"model": model_name, "model_revision": model_revision, "cases": cases,
                "summary": {"case_count": len(cases), "question_count": question_count,
                            "labelled_count": labelled_count, "matched_count": matched_count,
                            "accuracy": matched_count / labelled_count if labelled_count else None},
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
    return app
