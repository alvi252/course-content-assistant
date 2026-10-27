"""HTTP API and the single-page UI."""

from __future__ import annotations

import logging
import secrets
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .config import Settings
from .errors import (
    GenerationError,
    IdeaNotFoundError,
    IndexModelMismatchError,
    PassageUnavailableError,
    ProviderError,
    ProviderNotConfiguredError,
)
from .history import ChunkResolver, History
from .models import FORMATS, Idea, IdeaStatus, course_name, format_label
from .services import Services, build_services
from .voice import load_samples, load_style

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
MAX_TOPIC_CHARS = 200
MAX_SKIP_USED_DAYS = 365

_DOCS_PATHS = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc"})
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
        "font-src 'self'; connect-src 'self'; frame-ancestors 'none'"
    ),
}

# --- response schemas ---------------------------------------------------------------------


class SourceOut(BaseModel):
    chunk_id: str
    doc_path: str
    course: str
    lesson: str
    section: str
    text: str | None


class IdeaOut(BaseModel):
    id: int
    status: IdeaStatus
    format: str
    format_label: str
    title: str
    hook: str
    outline: list[str]
    quote: str
    source: SourceOut
    created_at: str
    used_at: str | None
    script: str | None
    caption: str | None
    warnings: list[str] = []


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    embedding_tokens: int
    cost_usd: float | None


class DroppedOut(BaseModel):
    reason: str
    title: str


class IdeasResponse(BaseModel):
    ideas: list[IdeaOut]
    dropped: list[DroppedOut]
    requested: int
    candidates: int
    fresh_passages_left: int
    empty_reason: str | None
    model: str | None
    latency_ms: dict[str, float]
    usage: UsageOut


class IdeaListResponse(BaseModel):
    ideas: list[IdeaOut]


class ScriptResponse(BaseModel):
    idea: IdeaOut
    script: str
    caption: str
    warnings: list[str]
    model: str
    latency_ms: dict[str, float]
    usage: UsageOut


class LessonOut(BaseModel):
    doc_path: str
    course: str
    lesson: str
    passages: int
    suggested: int
    used: int
    last_used_at: str | None


class CoverageTotals(BaseModel):
    lessons: int
    never_used: int
    passages: int
    used_passages: int


class CoverageResponse(BaseModel):
    lessons: list[LessonOut]
    totals: CoverageTotals


class SampleOut(BaseModel):
    path: str
    title: str
    format: str
    format_label: str
    text: str


class VoiceResponse(BaseModel):
    author: str
    style: str
    phrases: list[str]
    never_say: list[str]
    samples: list[SampleOut]


class FormatOut(BaseModel):
    key: str
    label: str


class IngestRequest(BaseModel):
    rebuild: bool = False


class IngestResponse(BaseModel):
    added: list[str]
    updated: list[str]
    unchanged: list[str]
    removed: list[str]
    skipped: list[str]
    chunks_written: int
    embeddings_computed: int
    embeddings_from_cache: int
    embedding_tokens: int
    changed: bool


class DocumentOut(BaseModel):
    path: str
    title: str
    ingested_at: str
    chunks: int


class HealthResponse(BaseModel):
    status: str
    version: str
    documents: int
    chunks: int
    ideas_used: int
    ideas_default_count: int
    ideas_max_count: int
    skip_used_days_default: int
    skip_used_days_max: int
    llm_model: str
    embedding_model: str


class IdeasRequest(BaseModel):
    """Omitted limits fall back to the settings; the upper bound of ``count`` is checked in the
    endpoint because it comes from the settings (IDEAS_MAX_COUNT)."""

    count: int | None = Field(default=None, ge=1)
    format: str | None = Field(default=None, examples=list(FORMATS))
    topic: str | None = Field(default=None, examples=["leftovers"])
    skip_used_days: int | None = Field(default=None, ge=0, le=MAX_SKIP_USED_DAYS)

    @field_validator("format")
    @classmethod
    def known_format(cls, value: str | None) -> str | None:
        if value is not None and value not in FORMATS:
            raise ValueError(f"format must be one of {', '.join(FORMATS)} or null")
        return value

    @field_validator("topic")
    @classmethod
    def clean_topic(cls, value: str | None) -> str | None:
        value = (value or "").strip()
        if len(value) > MAX_TOPIC_CHARS:
            raise ValueError(f"topic must be at most {MAX_TOPIC_CHARS} characters")
        return value or None


# --- application --------------------------------------------------------------------------


def create_app(settings: Settings | None = None, *, services: Services | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    svc = services or build_services(settings)
    ingest_lock = threading.Lock()

    app = FastAPI(
        title="Course Content Assistant",
        version=__version__,
        description="Content ideas and scripts built from your own course material.",
    )
    app.state.services = svc

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        for name, value in _SECURITY_HEADERS.items():
            # FastAPI's interactive docs load Swagger UI / ReDoc from a CDN with inline
            # scripts; the strict CSP applies to everything else.
            if name == "Content-Security-Policy" and request.url.path in _DOCS_PATHS:
                continue
            response.headers.setdefault(name, value)
        return response

    def require_admin(x_admin_token: Annotated[str | None, Header()] = None) -> None:
        if not settings.admin_token:
            return
        if x_admin_token is None or not secrets.compare_digest(
            x_admin_token.encode(), settings.admin_token.encode()
        ):
            raise HTTPException(status_code=401, detail="Missing or invalid X-Admin-Token")

    def provider_failure(where: str, exc: ProviderError) -> HTTPException:
        logger.error("Provider failure during %s: %s", where, exc)
        if isinstance(exc, ProviderNotConfiguredError):
            return HTTPException(status_code=503, detail=str(exc))  # the message holds no key
        return HTTPException(
            status_code=502, detail="The AI provider is unavailable. Please retry shortly."
        )

    def ideas_json(ideas: list[Idea]) -> list[IdeaOut]:
        resolver = ChunkResolver(svc.retriever.all_chunks())
        return [IdeaOut.model_validate(i.to_dict(None, resolver.resolve(i))) for i in ideas]

    def change_status(idea_id: int, status: IdeaStatus) -> IdeaOut:
        try:
            idea = svc.ideas.set_status(idea_id, status)
        except IdeaNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Idea not found") from exc
        return ideas_json([idea])[0]

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        stats = svc.store.stats()
        return HealthResponse(
            status="ok",
            version=__version__,
            documents=stats["documents"],
            chunks=stats["chunks"],
            ideas_used=svc.store.idea_counts()["used"],
            ideas_default_count=settings.ideas_default_count,
            ideas_max_count=settings.ideas_max_count,
            skip_used_days_default=settings.skip_used_days,
            skip_used_days_max=MAX_SKIP_USED_DAYS,
            llm_model=svc.ideas.llm.model,
            embedding_model=svc.embedder.model,
        )

    @app.get("/formats", response_model=list[FormatOut])
    def formats() -> list[FormatOut]:
        return [FormatOut(key=key, label=label) for key, label in FORMATS.items()]

    @app.get("/documents", response_model=list[DocumentOut])
    def documents() -> list[DocumentOut]:
        return [DocumentOut(**row) for row in svc.store.list_documents()]

    @app.post("/ideas", response_model=IdeasResponse)
    def create_ideas(body: IdeasRequest) -> IdeasResponse:
        if body.count is not None and body.count > settings.ideas_max_count:
            limit = settings.ideas_max_count
            raise RequestValidationError(
                [
                    {
                        "type": "less_than_equal",
                        "loc": ("body", "count"),
                        "msg": f"Input should be less than or equal to {limit}",
                        "input": body.count,
                        "ctx": {"le": limit},
                    }
                ]
            )
        try:
            result = svc.ideas.suggest(
                body.count,
                fmt=body.format,
                topic=body.topic,
                skip_used_days=body.skip_used_days,
            )
        except ProviderError as exc:
            raise provider_failure("POST /ideas", exc) from exc
        except IndexModelMismatchError as exc:
            logger.error("POST /ideas: %s", exc)
            raise HTTPException(
                status_code=409,
                detail="The index was built with a different embedding model; "
                "run ingest --rebuild.",
            ) from exc
        except GenerationError as exc:
            logger.error("Unusable model reply during POST /ideas: %s", exc)
            raise HTTPException(
                status_code=502, detail="The AI reply could not be used. Please retry."
            ) from exc
        return IdeasResponse.model_validate(result.to_dict())

    @app.get("/ideas", response_model=IdeaListResponse)
    def list_ideas(
        status: Annotated[IdeaStatus | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> IdeaListResponse:
        return IdeaListResponse(ideas=ideas_json(svc.store.list_ideas(status=status, limit=limit)))

    @app.post("/ideas/{idea_id}/use", response_model=IdeaOut)
    def use_idea(idea_id: int) -> IdeaOut:
        return change_status(idea_id, "used")

    @app.post("/ideas/{idea_id}/dismiss", response_model=IdeaOut)
    def dismiss_idea(idea_id: int) -> IdeaOut:
        return change_status(idea_id, "dismissed")

    @app.post("/ideas/{idea_id}/reset", response_model=IdeaOut)
    def reset_idea(idea_id: int) -> IdeaOut:
        return change_status(idea_id, "suggested")

    @app.post("/ideas/{idea_id}/script", response_model=ScriptResponse)
    def write_script(idea_id: int) -> ScriptResponse:
        try:
            result = svc.ideas.write_script(idea_id)
        except IdeaNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Idea not found") from exc
        except PassageUnavailableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ProviderError as exc:
            raise provider_failure("POST /ideas/{id}/script", exc) from exc
        except GenerationError as exc:
            logger.error("Unusable model reply during script: %s", exc)
            raise HTTPException(
                status_code=502, detail="The AI reply could not be used. Please retry."
            ) from exc
        return ScriptResponse.model_validate(result.to_dict())

    @app.get("/coverage", response_model=CoverageResponse)
    def coverage() -> CoverageResponse:
        rows = History.load(svc.store, svc.retriever.all_chunks()).coverage()
        lessons = [
            LessonOut(
                doc_path=r.doc_path,
                course=course_name(r.doc_path),
                lesson=r.title,
                passages=r.passages,
                suggested=r.suggested,
                used=r.used,
                last_used_at=r.last_used_at,
            )
            for r in rows
        ]
        totals = CoverageTotals(
            lessons=len(rows),
            never_used=sum(1 for r in rows if r.used == 0),
            passages=sum(r.passages for r in rows),
            used_passages=sum(r.used_passages for r in rows),
        )
        return CoverageResponse(lessons=lessons, totals=totals)

    @app.get("/voice", response_model=VoiceResponse)
    def voice() -> VoiceResponse:
        style = load_style(settings.voice_dir)
        samples = [
            SampleOut(
                path=s.path,
                title=s.title,
                format=s.format,
                format_label=format_label(s.format),
                text=s.text,
            )
            for s in load_samples(settings.voice_dir)
        ]
        return VoiceResponse(
            author=settings.author_name,
            style=style.raw,
            phrases=style.phrases,
            never_say=style.never_say,
            samples=samples,
        )

    @app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_admin)])
    def ingest(body: IngestRequest | None = None) -> IngestResponse:
        if not ingest_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="An ingestion is already running")
        try:
            report = svc.ingestor.run(rebuild=bool(body and body.rebuild))
        except ProviderError as exc:
            raise provider_failure("POST /ingest", exc) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            ingest_lock.release()
        return IngestResponse.model_validate(report.to_dict())

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
