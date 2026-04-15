"""Pydantic request/response models for the FastAPI endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Requests ─────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    """Request body for the /v1/search endpoint.

    The caller sends a free-text client brief (or keyword query) and receives
    matched audience segments ranked by relevance across platforms.

    Threshold overrides allow the calling service to apply account-level
    match quality settings. When omitted, the server-wide defaults (or
    per-platform overrides from AT_PLATFORM_MATCH_THRESHOLDS) are used.
    """

    query: str = Field(..., min_length=1, max_length=5000, description="Client brief or keyword query")
    platforms: list[str] | None = Field(None, description="Filter to specific platforms")
    top_k: int = Field(10, ge=1, le=50, description="Max results per platform")
    include_recommendations: bool = Field(True, description="Include related suggestions")
    include_scope_options: bool = Field(True, description="Include broaden/narrow options")
    match_threshold: float | None = Field(None, ge=0.0, le=1.0, description="Override match threshold (account-level). Falls back to server default.")
    partial_match_threshold: float | None = Field(None, ge=0.0, le=1.0, description="Override partial match threshold (account-level). Falls back to server default.")


# ── Responses ────────────────────────────────────────────────────────────


class SegmentResponse(BaseModel):
    """A single audience segment returned from search or equivalents lookup.

    The score is the final composite similarity (0-1) after text + optional
    Node2Vec re-ranking and cohesion boost. match_label is derived from
    account-level or server-default thresholds: "match" for strong relevance,
    "partial_match" for weaker but still useful relevance. Segments scoring
    below the partial threshold are never returned.
    """

    segment_id: str
    name: str
    platform: str
    score: float
    match_label: str = Field(description="'match' (>= match threshold) or 'partial_match' (>= partial threshold)")
    hierarchy: list[str]
    segment_type: str
    audience_size: int | None
    description: str | None


class MatchedSubCategoryResponse(BaseModel):
    """A sub-category (Layer 1 cluster) that matched the query.

    Sub-categories group related segments across platforms. The score is the
    cosine similarity between the query embedding and the sub-category centroid.
    source_sentence indicates which sentence from the brief drove this match.
    """

    sub_id: int
    name: str
    score: float
    platforms: list[str]
    member_count: int
    source_sentence: str


class RecommendationResponse(BaseModel):
    """A recommended sub-category derived from pre-computed taxonomy relationships.

    relation is one of: "related" (similar topic, different parent),
    "broader" (parent-level generalization), or "narrower" (more specific
    child). These are computed at build time using cosine similarity between
    sub-category centroids (related: 0.65-0.85 range).
    """

    sub_id: int
    name: str
    relation: str
    score: float
    member_count: int
    platforms: list[str]


class SearchMetadata(BaseModel):
    """Timing and count metadata for a search response.

    total_segments is the count after threshold filtering (not the raw
    Qdrant result count). search_time_ms covers the full server-side
    pipeline: embedding, 3-layer search, re-ranking, and filtering.
    """

    total_segments: int
    platforms_matched: int
    sentences_processed: int
    search_time_ms: float


class SearchResponse(BaseModel):
    """Top-level response for POST /v1/search.

    sentence_topics maps each sentence extracted from the query to the
    super-category names it matched (for UI display / explainability).
    segments_by_platform is keyed by platform name; each value is a list
    of segments sorted by descending score, already filtered by match
    thresholds. Platforms with zero qualifying segments are omitted.
    """

    query: str
    sentence_topics: dict[str, list[str]]
    matched_subcategories: list[MatchedSubCategoryResponse]
    segments_by_platform: dict[str, list[SegmentResponse]]
    recommendations: list[RecommendationResponse]
    broadening_options: list[RecommendationResponse]
    narrowing_options: list[RecommendationResponse]
    metadata: SearchMetadata


class SegmentDetailResponse(BaseModel):
    """Full detail for a single segment, returned by GET /v1/segments/{id}.

    Includes the segment's position in the taxonomy hierarchy:
    super_category_id (Layer 0) -> subcategory_id (Layer 1) -> this segment.
    parent_segment_id links to the DSP's own hierarchy if one exists.
    """

    segment_id: str
    name: str
    platform: str
    hierarchy: list[str]
    segment_type: str
    audience_size: int | None
    description: str | None
    subcategory_id: int
    super_category_id: int
    parent_segment_id: str | None


class SuperCategoryResponse(BaseModel):
    """A Layer 0 super-category — the broadest grouping in the taxonomy.

    Created by HDBSCAN clustering over all segment embeddings. Each
    super-category contains one or more sub-categories (Layer 1).
    """

    super_id: int
    name: str
    subcategory_count: int
    platforms: list[str]
    member_count: int


class SubCategoryResponse(BaseModel):
    """A Layer 1 sub-category nested under a super-category.

    Groups semantically similar segments across platforms. parent_super_id
    links to the containing SuperCategory.
    """

    sub_id: int
    name: str
    parent_super_id: int
    platforms: list[str]
    member_count: int


class PlatformResponse(BaseModel):
    """Platform summary with segment count (unused — reserved for future use)."""

    name: str
    segment_count: int


class SystemStatsResponse(BaseModel):
    """Point counts across all three Qdrant collections and the configured platforms."""

    supercategories: int
    subcategories: int
    segments: int
    platforms: list[str]


class HealthResponse(BaseModel):
    """Health check response. status is 'ok' when both Qdrant and the
    embedding model are operational, 'degraded' otherwise.
    """

    status: str
    qdrant_connected: bool
    model_loaded: bool
