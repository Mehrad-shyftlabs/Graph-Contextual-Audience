"""FastAPI application for the Audience Targeting API."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from audience_targeting.api_models import (
    HealthResponse,
    MatchedSubCategoryResponse,
    RecommendationResponse,
    SearchMetadata,
    SearchRequest,
    SearchResponse,
    SegmentDetailResponse,
    SegmentResponse,
    SubCategoryResponse,
    SuperCategoryResponse,
    SystemStatsResponse,
)
from audience_targeting.logging_config import setup_logging
from audience_targeting.search_engine import AudienceSearchEngine, create_engine
from audience_targeting.settings import Settings

logger = logging.getLogger("audience_targeting.api")

# Global references set during lifespan
_engine: AudienceSearchEngine | None = None
_settings: Settings | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup/shutdown lifecycle.

    On startup: load settings, configure logging, initialize the search engine
    (embedding model + Qdrant client), and verify that all three required
    Qdrant collections exist and contain data. Fails fast with RuntimeError
    if any collection is missing.
    """
    global _engine, _settings
    _settings = Settings()
    setup_logging(_settings.log_level)
    _engine = create_engine(_settings)

    # Verify required Qdrant collections exist and have data
    for base in ("supercategories", "subcategories", "segments"):
        coll = _settings.collection_name(base)
        try:
            info = _engine.client.get_collection(coll)
            if info.points_count == 0:
                logger.warning("Collection '%s' exists but is empty", coll)
            else:
                logger.info("Collection '%s': %d points", coll, info.points_count)
        except Exception as e:
            logger.error("Required collection '%s' not found: %s", coll, e)
            raise RuntimeError(f"Missing required Qdrant collection: {coll}") from e

    yield
    _engine = None


app = FastAPI(
    title="Audience Targeting API",
    version="2.0.0",
    description="AI-powered cross-platform audience segment matching",
    lifespan=lifespan,
)

# ── Rate limiting ────────────────────────────────────────────────────────

_init_settings = Settings()
limiter = Limiter(key_func=get_remote_address, default_limits=[])
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


# ── Middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=_init_settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with method, path, status, duration, and request ID.

    Reads X-Request-ID from the incoming request header (for distributed
    tracing); generates a UUID if absent. Returns the ID in the response
    X-Request-ID header so callers can correlate.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.time()
        response = await call_next(request)
        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(elapsed_ms, 2),
                "request_id": request_id,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestLoggingMiddleware)


# ── Global exception handler ────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── Auth ────────────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(_api_key_header)):
    """FastAPI dependency that validates the X-API-Key header.

    When AT_API_KEY is not set (None), authentication is disabled — all
    requests pass through. When set, the header must match exactly or a
    401 is returned. Applied to all /v1/* endpoints; /health and /ready
    are exempt so orchestrators can probe without credentials.
    """
    settings = _get_settings()
    if settings.api_key is None:
        return  # Auth disabled when no key configured
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Helpers ─────────────────────────────────────────────────────────────

def _get_engine() -> AudienceSearchEngine:
    """Return the initialized search engine or 503 if startup hasn't completed."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine


def _get_settings() -> Settings:
    """Return the initialized settings or 503 if startup hasn't completed."""
    if _settings is None:
        raise HTTPException(status_code=503, detail="Settings not initialized")
    return _settings


# ── Health ───────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe. Always returns 200; status is 'ok' or 'degraded'."""
    qdrant_ok = False
    try:
        engine = _get_engine()
        engine.client.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if qdrant_ok and _engine is not None else "degraded",
        qdrant_connected=qdrant_ok,
        model_loaded=_engine is not None,
    )


@app.get("/ready")
async def ready():
    """Readiness probe. Returns 200 if Qdrant is reachable, 503 otherwise."""
    engine = _get_engine()
    try:
        engine.client.get_collections()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant not reachable: {e}")
    return {"status": "ready"}


# ── Core search ──────────────────────────────────────────────────────────


@app.post("/v1/search", response_model=SearchResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit(lambda: _init_settings.rate_limit)
async def search(request: Request, body: SearchRequest):
    """Execute a semantic audience search across all platforms.

    Pipeline: embed query → Layer 0 (super-categories) → Layer 1
    (sub-categories) → Layer 2 (segments) → re-rank → threshold filter.

    If body.match_threshold or body.partial_match_threshold are provided,
    they override the server/platform defaults for this request (account-level
    control). Segments below the partial threshold are excluded entirely.
    """
    engine = _get_engine()
    t0 = time.time()

    try:
        result = engine.search(
            query=body.query,
            platforms=body.platforms,
            top_k_segments=body.top_k,
        )
    except (ConnectionError, TimeoutError) as e:
        logger.error("Qdrant search failed: %s", e)
        raise HTTPException(status_code=503, detail="Search backend unavailable")
    except Exception as e:
        logger.exception("Unexpected error during search")
        raise HTTPException(status_code=500, detail="Search failed")

    elapsed_ms = (time.time() - t0) * 1000

    # Build response — filter by match threshold and assign labels
    # Use request-level overrides (account-level) if provided, else server defaults
    settings = _get_settings()

    def _classify(score: float, platform: str) -> str | None:
        match_thr, partial_thr = settings.get_match_thresholds(platform)
        if body.match_threshold is not None:
            match_thr = body.match_threshold
        if body.partial_match_threshold is not None:
            partial_thr = body.partial_match_threshold
        if score >= match_thr:
            return "match"
        if score >= partial_thr:
            return "partial_match"
        return None

    segments_response: dict[str, list[SegmentResponse]] = {}
    total_segments = 0
    for platform, seg_scores in result.segments_by_platform.items():
        filtered: list[SegmentResponse] = []
        for seg, score in seg_scores:
            label = _classify(score, platform)
            if label is None:
                continue
            filtered.append(SegmentResponse(
                segment_id=seg.id,
                name=seg.name,
                platform=seg.platform,
                score=score,
                match_label=label,
                hierarchy=seg.hierarchy,
                segment_type=seg.segment_type,
                audience_size=seg.audience_size,
                description=seg.description,
            ))
        if filtered:
            segments_response[platform] = filtered
            total_segments += len(filtered)

    matched_subs = [
        MatchedSubCategoryResponse(
            sub_id=ms.sub_category.id,
            name=ms.sub_category.name,
            score=ms.score,
            platforms=list(ms.sub_category.platforms),
            member_count=ms.sub_category.member_count,
            source_sentence=ms.source_sentence,
        )
        for ms in result.matched_subcategories
    ]

    recs = [
        RecommendationResponse(
            sub_id=r.sub_id, name=r.name, relation=r.relation,
            score=r.score, member_count=r.member_count, platforms=r.platforms,
        )
        for r in result.recommendations
    ] if body.include_recommendations else []

    broadening = [
        RecommendationResponse(
            sub_id=r.sub_id, name=r.name, relation=r.relation,
            score=r.score, member_count=r.member_count, platforms=r.platforms,
        )
        for r in result.broadening_options
    ] if body.include_scope_options else []

    narrowing = [
        RecommendationResponse(
            sub_id=r.sub_id, name=r.name, relation=r.relation,
            score=r.score, member_count=r.member_count, platforms=r.platforms,
        )
        for r in result.narrowing_options
    ] if body.include_scope_options else []

    return SearchResponse(
        query=result.query,
        sentence_topics=result.sentence_topics,
        matched_subcategories=matched_subs,
        segments_by_platform=segments_response,
        recommendations=recs,
        broadening_options=broadening,
        narrowing_options=narrowing,
        metadata=SearchMetadata(
            total_segments=total_segments,
            platforms_matched=len(result.segments_by_platform),
            sentences_processed=len(result.sentence_topics),
            search_time_ms=round(elapsed_ms, 2),
        ),
    )


# ── Browse taxonomy ──────────────────────────────────────────────────────


@app.get("/v1/supercategories", response_model=list[SuperCategoryResponse], dependencies=[Depends(verify_api_key)])
async def list_supercategories():
    """Return all Layer 0 super-categories from the taxonomy."""
    engine = _get_engine()
    settings = _get_settings()

    try:
        results, _ = engine.client.scroll(
            collection_name=settings.collection_name("supercategories"),
            limit=100,
        )
    except Exception as e:
        logger.error("Failed to fetch supercategories: %s", e)
        raise HTTPException(status_code=503, detail="Search backend unavailable")

    return [
        SuperCategoryResponse(
            super_id=r.payload["super_id"],
            name=r.payload["name"],
            subcategory_count=len(r.payload.get("subcategory_ids", [])),
            platforms=r.payload.get("platforms", []),
            member_count=r.payload.get("member_count", 0),
        )
        for r in results
    ]


@app.get("/v1/supercategories/{super_id}/subcategories", response_model=list[SubCategoryResponse], dependencies=[Depends(verify_api_key)])
async def list_subcategories(super_id: int):
    """Return all Layer 1 sub-categories belonging to the given super-category."""
    engine = _get_engine()
    settings = _get_settings()
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    try:
        results, _ = engine.client.scroll(
            collection_name=settings.collection_name("subcategories"),
            scroll_filter=Filter(must=[
                FieldCondition(key="parent_super_id", match=MatchValue(value=super_id)),
            ]),
            limit=100,
        )
    except Exception as e:
        logger.error("Failed to fetch subcategories: %s", e)
        raise HTTPException(status_code=503, detail="Search backend unavailable")

    return [
        SubCategoryResponse(
            sub_id=r.payload["sub_id"],
            name=r.payload["name"],
            parent_super_id=r.payload["parent_super_id"],
            platforms=r.payload.get("platforms", []),
            member_count=r.payload.get("member_count", 0),
        )
        for r in results
    ]


@app.get("/v1/segments/{segment_id}", response_model=SegmentDetailResponse, dependencies=[Depends(verify_api_key)])
async def get_segment(segment_id: str):
    """Return full detail for a single segment by its DSP-specific ID. 404 if not found."""
    engine = _get_engine()
    settings = _get_settings()
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    try:
        results, _ = engine.client.scroll(
            collection_name=settings.collection_name("segments"),
            scroll_filter=Filter(must=[
                FieldCondition(key="segment_id", match=MatchValue(value=segment_id)),
            ]),
            limit=1,
        )
    except Exception as e:
        logger.error("Failed to fetch segment: %s", e)
        raise HTTPException(status_code=503, detail="Search backend unavailable")

    if not results:
        raise HTTPException(status_code=404, detail=f"Segment '{segment_id}' not found")

    p = results[0].payload
    return SegmentDetailResponse(
        segment_id=p["segment_id"],
        name=p["name"],
        platform=p["platform"],
        hierarchy=p.get("hierarchy", []),
        segment_type=p.get("segment_type", ""),
        audience_size=p.get("audience_size"),
        description=p.get("description"),
        subcategory_id=p.get("subcategory_id", -1),
        super_category_id=p.get("super_category_id", -1),
        parent_segment_id=p.get("parent_segment_id"),
    )


@app.get("/v1/segments/{segment_id}/equivalents", response_model=list[SegmentResponse], dependencies=[Depends(verify_api_key)])
async def get_equivalents(segment_id: str):
    """Find cross-platform equivalent segments using BGE vector similarity.

    Looks up the segment's embedding and searches for near-identical segments
    on other platforms (similarity >= 0.85). Results are filtered by match
    quality thresholds, same as /v1/search. 404 if the source segment is not found.
    """
    engine = _get_engine()
    settings = _get_settings()
    from audience_targeting import qdrant_store
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    # Find the segment first
    try:
        results, _ = engine.client.scroll(
            collection_name=settings.collection_name("segments"),
            scroll_filter=Filter(must=[
                FieldCondition(key="segment_id", match=MatchValue(value=segment_id)),
            ]),
            limit=1,
            with_vectors=True,
        )
    except Exception as e:
        logger.error("Failed to fetch segment for equivalents: %s", e)
        raise HTTPException(status_code=503, detail="Search backend unavailable")

    if not results:
        raise HTTPException(status_code=404, detail=f"Segment '{segment_id}' not found")

    point = results[0]
    vec = point.vector
    bge_vec = vec.get("bge", []) if isinstance(vec, dict) else vec

    equivalents = qdrant_store.get_segment_equivalents(
        engine.client,
        segment_vector=bge_vec,
        subcategory_id=point.payload.get("subcategory_id", -1),
        exclude_platform=point.payload["platform"],
        settings=settings,
    )

    filtered = []
    for e in equivalents:
        label = settings.classify_match(e["score"], e["platform"])
        if label is None:
            continue
        filtered.append(SegmentResponse(
            segment_id=e["segment_id"],
            name=e["name"],
            platform=e["platform"],
            score=e["score"],
            match_label=label,
            hierarchy=e.get("hierarchy", []),
            segment_type=e.get("segment_type", ""),
            audience_size=e.get("audience_size"),
            description=e.get("description"),
        ))
    return filtered


# ── System info ──────────────────────────────────────────────────────────


@app.get("/v1/platforms", response_model=list[str], dependencies=[Depends(verify_api_key)])
async def list_platforms():
    """Return the list of configured platform names (e.g. meta, tiktok, ttd)."""
    return _get_settings().platforms


@app.get("/v1/stats", response_model=SystemStatsResponse, dependencies=[Depends(verify_api_key)])
async def system_stats():
    """Return point counts for each Qdrant collection and configured platforms."""
    engine = _get_engine()
    settings = _get_settings()

    try:
        collections = engine.client.get_collections().collections
        counts = {}
        for c in collections:
            info = engine.client.get_collection(c.name)
            counts[c.name] = info.points_count
    except Exception as e:
        logger.error("Failed to fetch stats: %s", e)
        raise HTTPException(status_code=503, detail="Search backend unavailable")

    return SystemStatsResponse(
        supercategories=counts.get(settings.collection_name("supercategories"), 0),
        subcategories=counts.get(settings.collection_name("subcategories"), 0),
        segments=counts.get(settings.collection_name("segments"), 0),
        platforms=settings.platforms,
    )
