"""Typed synchronous HTTP client for a running OpenJEV-SemIf service."""
from __future__ import annotations
import os
from typing import Any
import httpx
from .api.schemas import (
    HealthResponse, ScoreRequest, ScoreResponse, SystemOneRequest, SystemOneResponse,
)

class OpenJEVHTTPError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")

class OpenJEVClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        session: httpx.Client | None = None,
    ):
        if timeout <= 0: raise ValueError("timeout must be positive")
        self._owned = session is None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._session = session or httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout,
        )
        self._headers = headers if session is not None else None

    def close(self) -> None:
        if self._owned: self._session.close()

    def __enter__(self) -> "OpenJEVClient": return self

    def __exit__(self, exc_type, exc, tb) -> None: self.close()

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self._session.request(method, path, json=payload, headers=self._headers)
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise OpenJEVHTTPError(response.status_code, str(detail))
        return response.json()

    def health(self) -> HealthResponse:
        return HealthResponse.model_validate(self._request("GET", "/health"))

    def score(self, request: ScoreRequest | dict[str, Any]) -> ScoreResponse:
        request = ScoreRequest.model_validate(request)
        data = self._request("POST", "/score", request.model_dump(exclude_none=True))
        return ScoreResponse.model_validate(data)

    def systemone(self, request: SystemOneRequest | dict[str, Any]) -> SystemOneResponse:
        request = SystemOneRequest.model_validate(request)
        data = self._request("POST", "/v1/systemone", request.model_dump(exclude_none=True))
        return SystemOneResponse.model_validate(data)
