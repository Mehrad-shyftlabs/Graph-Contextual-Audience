"""FastAPI application for the Audience Targeting API."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

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

# Global references set during lifespan
_engine: AudienceSearchEngine | None = None
_settings: Settings | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _settings
    _settings = Settings()
    setup_logging(_settings.log_level)
    _engine = create_engine(_settings)
    yield
    _engine = None


app = FastAPI(
    title="Audience Targeting API",
    version="2.0.0",
    description="AI-powered cross-platform audience segment matching",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_engine() -> AudienceSearchEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine


def _get_settings() -> Settings:
    if _settings is None:
        raise HTTPException(status_code=503, detail="Settings not initialized")
    return _settings


# ── Health ───────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
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
    engine = _get_engine()
    try:
        engine.client.get_collections()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant not reachable: {e}")
    return {"status": "ready"}


# ── Core search ──────────────────────────────────────────────────────────


@app.post("/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    engine = _get_engine()
    t0 = time.time()

    result = engine.search(
        query=request.query,
        platforms=request.platforms,
        top_k_segments=request.top_k,
    )

    elapsed_ms = (time.time() - t0) * 1000

    # Build response — filter by match threshold and assign labels
    settings = _get_settings()
    segments_response: dict[str, list[SegmentResponse]] = {}
    total_segments = 0
    for platform, seg_scores in result.segments_by_platform.items():
        filtered: list[SegmentResponse] = []
        for seg, score in seg_scores:
            label = settings.classify_match(score, platform)
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
    ] if request.include_recommendations else []

    broadening = [
        RecommendationResponse(
            sub_id=r.sub_id, name=r.name, relation=r.relation,
            score=r.score, member_count=r.member_count, platforms=r.platforms,
        )
        for r in result.broadening_options
    ] if request.include_scope_options else []

    narrowing = [
        RecommendationResponse(
            sub_id=r.sub_id, name=r.name, relation=r.relation,
            score=r.score, member_count=r.member_count, platforms=r.platforms,
        )
        for r in result.narrowing_options
    ] if request.include_scope_options else []

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


@app.get("/v1/supercategories", response_model=list[SuperCategoryResponse])
async def list_supercategories():
    engine = _get_engine()
    settings = _get_settings()

    results, _ = engine.client.scroll(
        collection_name=settings.collection_name("supercategories"),
        limit=100,
    )
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


@app.get("/v1/supercategories/{super_id}/subcategories", response_model=list[SubCategoryResponse])
async def list_subcategories(super_id: int):
    engine = _get_engine()
    settings = _get_settings()
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    results, _ = engine.client.scroll(
        collection_name=settings.collection_name("subcategories"),
        scroll_filter=Filter(must=[
            FieldCondition(key="parent_super_id", match=MatchValue(value=super_id)),
        ]),
        limit=100,
    )
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


@app.get("/v1/segments/{segment_id}", response_model=SegmentDetailResponse)
async def get_segment(segment_id: str):
    engine = _get_engine()
    settings = _get_settings()
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    results, _ = engine.client.scroll(
        collection_name=settings.collection_name("segments"),
        scroll_filter=Filter(must=[
            FieldCondition(key="segment_id", match=MatchValue(value=segment_id)),
        ]),
        limit=1,
    )
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


@app.get("/v1/segments/{segment_id}/equivalents", response_model=list[SegmentResponse])
async def get_equivalents(segment_id: str):
    engine = _get_engine()
    settings = _get_settings()
    from audience_targeting import qdrant_store
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    # Find the segment first
    results, _ = engine.client.scroll(
        collection_name=settings.collection_name("segments"),
        scroll_filter=Filter(must=[
            FieldCondition(key="segment_id", match=MatchValue(value=segment_id)),
        ]),
        limit=1,
        with_vectors=True,
    )
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


@app.get("/v1/platforms", response_model=list[str])
async def list_platforms():
    return _get_settings().platforms


@app.get("/v1/stats", response_model=SystemStatsResponse)
async def system_stats():
    engine = _get_engine()
    settings = _get_settings()

    collections = engine.client.get_collections().collections
    counts = {}
    for c in collections:
        info = engine.client.get_collection(c.name)
        counts[c.name] = info.points_count

    return SystemStatsResponse(
        supercategories=counts.get(settings.collection_name("supercategories"), 0),
        subcategories=counts.get(settings.collection_name("subcategories"), 0),
        segments=counts.get(settings.collection_name("segments"), 0),
        platforms=settings.platforms,
    )
