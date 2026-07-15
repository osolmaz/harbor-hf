from __future__ import annotations

import hashlib
import os
from base64 import urlsafe_b64decode, urlsafe_b64encode
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from harbor_hf.presentation.config import PresentationConfig
from harbor_hf.presentation.repository import PresentationError, ResultRepository
from harbor_hf.presentation.service import ResultNotFound, ResultService


class ServiceHolder:
    def __init__(
        self,
        service: ResultService | None = None,
        config: PresentationConfig | None = None,
    ) -> None:
        self._service = service
        self._config = config
        self._injected = service is not None
        self._loaded_at = monotonic() if service is not None else 0.0
        self._lock = Lock()

    def get(self) -> ResultService:
        if self._injected:
            assert self._service is not None
            return self._service
        config = self._config or PresentationConfig.from_env()
        now = monotonic()
        immutable = _is_commit(config.index_revision)
        if self._service is not None and (
            immutable or now - self._loaded_at < config.refresh_seconds
        ):
            return self._service
        with self._lock:
            now = monotonic()
            if self._service is not None and (
                immutable or now - self._loaded_at < config.refresh_seconds
            ):
                return self._service
            repository = ResultRepository(config)
            snapshot = repository.load()
            if (
                self._service is None
                or self._service.snapshot.index_revision != snapshot.index_revision
            ):
                self._service = ResultService(snapshot, config.title, repository)
            self._loaded_at = now
        assert self._service is not None
        return self._service


def create_app(
    service: ResultService | None = None, *, web_dir: Path | None = None
) -> FastAPI:
    holder = ServiceHolder(service)
    app = FastAPI(
        title="Harbor Results API",
        version="1.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    _register_error_handlers(app)
    _register_cache_middleware(app, holder)
    _register_api_routes(app, holder)
    _register_frontend(app, web_dir)
    return app


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ResultNotFound)
    async def not_found(_request: Request, error: ResultNotFound) -> JSONResponse:
        return JSONResponse(
            content=_error("not_found", f"Result entity {error.args[0]} was not found"),
            status_code=404,
        )

    @app.exception_handler(PresentationError)
    async def unavailable(_request: Request, error: PresentationError) -> JSONResponse:
        return JSONResponse(
            content=_error(
                "results_unavailable",
                f"Published results are unavailable: {error}",
            ),
            status_code=503,
        )

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, error: HTTPException) -> JSONResponse:
        return JSONResponse(
            content=_error("request_rejected", str(error.detail)),
            status_code=error.status_code,
            headers=error.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            content=_error("invalid_request", str(error)), status_code=422
        )


def _register_cache_middleware(app: FastAPI, holder: ServiceHolder) -> None:
    @app.middleware("http")
    async def immutable_etag(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        if request.method != "GET" or not request.url.path.startswith("/api/v1/"):
            return response
        try:
            revision = holder.get().snapshot.index_revision
        except (PresentationError, ValueError):
            return response
        digest = hashlib.sha256(
            f"{revision}:{request.url.path}?{request.url.query}".encode()
        ).hexdigest()
        etag = f'"{digest}"'
        if request.headers.get("if-none-match") == etag and response.status_code == 200:
            return Response(status_code=304, headers={"ETag": etag})
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = (
            "public, max-age=60, stale-while-revalidate=3600"
        )
        return response


def _register_api_routes(app: FastAPI, holder: ServiceHolder) -> None:
    _register_catalog_routes(app, holder)
    _register_run_routes(app, holder)
    _register_campaign_routes(app, holder)
    _register_entity_routes(app, holder)


def _register_catalog_routes(app: FastAPI, holder: ServiceHolder) -> None:
    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return holder.get().health()

    @app.get("/api/v1/ready")
    def ready() -> dict[str, Any]:
        return holder.get().health()

    @app.get("/api/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        return holder.get().capabilities()

    @app.get("/api/v1/runs")
    def runs(
        search: Annotated[str, Query(max_length=200)] = "",
        benchmark: Annotated[str, Query(max_length=200)] = "",
        model: Annotated[str, Query(max_length=300)] = "",
        hardware: Annotated[str, Query(max_length=200)] = "",
        cursor: Annotated[str, Query(max_length=128)] = "",
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> dict[str, Any]:
        result = holder.get().list_runs(
            search=search, benchmark=benchmark, model=model, hardware=hardware
        )
        return _paginate(result, cursor=cursor, limit=limit)


def _register_run_routes(app: FastAPI, holder: ServiceHolder) -> None:
    @app.get("/api/v1/runs/{run_id}")
    def run(run_id: str) -> dict[str, Any]:
        return holder.get().run(run_id)

    @app.get("/api/v1/runs/{run_id}/compare/{other_run_id}")
    def compare(run_id: str, other_run_id: str) -> dict[str, Any]:
        return holder.get().compare(run_id, other_run_id)

    @app.get("/api/v1/compare")
    def compare_query(
        run_id: Annotated[list[str], Query(min_length=2, max_length=2)],
    ) -> dict[str, Any]:
        return holder.get().compare(run_id[0], run_id[1])

    @app.get("/api/v1/runs/{run_id}/trials")
    def run_trials(run_id: str) -> dict[str, Any]:
        detail = holder.get().run(run_id)
        return {"items": detail["trials"], "total": len(detail["trials"])}

    @app.get("/api/v1/runs/{run_id}/metrics")
    def run_metrics(run_id: str) -> dict[str, Any]:
        detail = holder.get().run(run_id)
        return {"items": detail["metrics"], "total": len(detail["metrics"])}


def _register_campaign_routes(app: FastAPI, holder: ServiceHolder) -> None:
    @app.get("/api/v1/campaigns")
    def campaigns(
        cursor: Annotated[str, Query(max_length=128)] = "",
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> dict[str, Any]:
        return _paginate(holder.get().list_campaigns(), cursor=cursor, limit=limit)

    @app.get("/api/v1/campaigns/{campaign_id}")
    def campaign(campaign_id: str) -> dict[str, Any]:
        return holder.get().campaign(campaign_id)


def _register_entity_routes(app: FastAPI, holder: ServiceHolder) -> None:
    @app.get("/api/v1/trials/{trial_id}")
    def trial(trial_id: str) -> dict[str, Any]:
        return holder.get().trial(trial_id)

    @app.get("/api/v1/trials/{trial_id}/executions")
    def trial_executions(trial_id: str) -> dict[str, Any]:
        detail = holder.get().trial(trial_id)
        return {
            "items": detail["executions"],
            "total": len(detail["executions"]),
        }

    @app.get("/api/v1/executions/{execution_id}")
    def execution(execution_id: str) -> dict[str, Any]:
        return holder.get().execution(execution_id)

    @app.get("/api/v1/executions/{execution_id}/trajectory")
    def trajectory(execution_id: str) -> None:
        holder.get().execution(execution_id)
        raise HTTPException(
            status_code=403,
            detail="Raw trajectories remain in private canonical evidence",
        )

    @app.get("/api/v1/artifacts/{artifact_id}")
    def artifact(artifact_id: str) -> dict[str, Any]:
        return holder.get().artifact(artifact_id)

    @app.get("/api/v1/artifacts/{artifact_id}/content")
    def artifact_content(artifact_id: str) -> None:
        holder.get().artifact(artifact_id)
        raise HTTPException(
            status_code=403,
            detail="Artifact content is unavailable in public mode",
        )


def _register_frontend(app: FastAPI, web_dir: Path | None) -> None:
    static_root = web_dir or Path(
        os.environ.get("HARBOR_HF_WEB_DIR", "apps/results-web/dist")
    )

    @app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        candidate = (static_root / path).resolve()
        root = static_root.resolve()
        if path and candidate.is_relative_to(root) and candidate.is_file():
            return FileResponse(candidate)
        index = root / "index.html"
        if index.is_file():
            return FileResponse(index)
        raise HTTPException(status_code=404, detail="Results frontend is not built")


def _error(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def _paginate(result: dict[str, Any], *, cursor: str, limit: int) -> dict[str, Any]:
    offset = _decode_cursor(cursor)
    items = result["items"]
    page = items[offset : offset + limit]
    next_offset = offset + len(page)
    return {
        **result,
        "items": page,
        "next_cursor": _encode_cursor(next_offset)
        if next_offset < len(items)
        else None,
    }


def _decode_cursor(cursor: str) -> int:
    if not cursor:
        return 0
    try:
        value = urlsafe_b64decode(cursor.encode()).decode()
        version, offset = value.split(":", maxsplit=1)
        if version != "v1" or int(offset) < 0:
            raise ValueError
        return int(offset)
    except (ValueError, UnicodeDecodeError) as error:
        raise HTTPException(
            status_code=400, detail="Invalid pagination cursor"
        ) from error


def _encode_cursor(offset: int) -> str:
    return urlsafe_b64encode(f"v1:{offset}".encode()).decode()


def _is_commit(value: str) -> bool:
    return len(value) in {40, 64} and all(
        character in "0123456789abcdef" for character in value
    )


app = create_app()
