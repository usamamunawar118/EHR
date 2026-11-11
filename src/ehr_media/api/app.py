"""FastAPI application factory.

Run with ``ehr serve`` (or ``uvicorn ehr_media.api.app:create_app --factory``).

The same app serves the static front-end, so the browser talks to one origin (no CORS needed).
There is intentionally **no authentication**: this is a demo over synthetic data. A real
deployment must add SSO/OAuth, audit logging and TLS before any PHI touches it.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ehr_media import __version__
from ehr_media.api.deps import Container, build_container
from ehr_media.api.routes import health, patients, search
from ehr_media.config import Settings, get_settings
from ehr_media.logging import configure_logging

logger = logging.getLogger(__name__)


class RevalidatingStaticFiles(StaticFiles):
    """Static files that browsers must revalidate on every load.

    Starlette sends no ``Cache-Control``, so browsers cache the JS/CSS heuristically and users can
    keep running a stale UI after an update. ``no-cache`` still allows a cheap ``304`` via the
    ETag/Last-Modified validators the base class already sends.
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response: Response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


DESCRIPTION = """
Semantic search over FHIR R4 patient records and AI-generated summaries.

**All data is synthetic. Summaries are AI-generated and are not clinical decisions.**
"""


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """Build the app. Tests pass a ready-made ``container``; production builds it in lifespan."""
    settings = settings or (container.settings if container else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        app.state.container = container or build_container(settings)
        # The embedder is warmed in build_container; this also pages the vector index into
        # memory, so the first user query is as fast as the rest (it was ~1 s cold).
        try:
            app.state.container.search.search("warm-up")
        except Exception as exc:  # an optimisation only: never block start-up on it
            logger.warning(
                "search warm-up failed (%s); first query may be slower", type(exc).__name__
            )
        logger.info(
            "ready: %d patients, %d indexed entries",
            app.state.container.repo.count()["patients"],
            app.state.container.store.count(),
        )
        yield

    app = FastAPI(
        title="EHR Media Intelligence API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.include_router(search.router)
    app.include_router(patients.router)
    app.include_router(health.router)

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception) -> JSONResponse:
        # Never leak internals (or record text) to the client; details go to the server log.
        logger.exception("unhandled error: %s", type(exc).__name__)
        return JSONResponse({"detail": "Internal server error"}, status_code=500)

    if settings.frontend_dir.is_dir():
        # Mounted last so API routes take precedence; html=True serves index.html at "/".
        app.mount(
            "/", RevalidatingStaticFiles(directory=settings.frontend_dir, html=True), name="ui"
        )
    else:
        logger.warning("frontend directory %s not found; serving API only", settings.frontend_dir)
    return app
