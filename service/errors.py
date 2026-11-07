"""RFC 7807 problem+json errors."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, status: int, title: str, detail: str | None = None, type_: str = "about:blank", **extra):
        self.status, self.title, self.detail, self.type, self.extra = status, title, detail, type_, extra


def problem(status: int, title: str, detail: str | None = None, type_: str = "about:blank", **extra) -> JSONResponse:
    body = {"type": type_, "title": title, "status": status}
    if detail:
        body["detail"] = detail
    body.update(extra)
    return JSONResponse(status_code=status, content=body, media_type="application/problem+json")


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return problem(exc.status, exc.title, exc.detail, exc.type, **exc.extra)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        errors = [{"loc": [str(x) for x in e.get("loc", [])], "msg": str(e.get("msg", "")), "type": e.get("type", "")}
                  for e in exc.errors()]
        return problem(422, "Validation failed", "Request body or parameters are invalid.", errors=errors)
