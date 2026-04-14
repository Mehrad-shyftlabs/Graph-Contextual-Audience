"""Pydantic request/response models for the FastAPI endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Requests ─────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="Client brief or keyword query")
    platforms: list[str] | None = Field(None, description="Filter to specific platforms")
    top_k: int = Field(10, ge=1, le=50, description="Max results per platform")
    include_recommendations: bool = Field(True, description="Include related suggestions")
    include_scope_options: bool = Field(True, description="Include broaden/narrow options")


# ── Responses ────────────────────────────────────────────────────────────


class SegmentResponse(BaseModel):
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
    sub_id: int
    name: str
    score: float
    platforms: list[str]
    member_count: int
    source_sentence: str


class RecommendationResponse(BaseModel):
    sub_id: int
    name: str
    relation: str
    score: float
    member_count: int
    platforms: list[str]


class SearchMetadata(BaseModel):
    total_segments: int
    platforms_matched: int
    sentences_processed: int
    search_time_ms: float


class SearchResponse(BaseModel):
    query: str
    sentence_topics: dict[str, list[str]]
    matched_subcategories: list[MatchedSubCategoryResponse]
    segments_by_platform: dict[str, list[SegmentResponse]]
    recommendations: list[RecommendationResponse]
    broadening_options: list[RecommendationResponse]
    narrowing_options: list[RecommendationResponse]
    metadata: SearchMetadata


class SegmentDetailResponse(BaseModel):
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
    super_id: int
    name: str
    subcategory_count: int
    platforms: list[str]
    member_count: int


class SubCategoryResponse(BaseModel):
    sub_id: int
    name: str
    parent_super_id: int
    platforms: list[str]
    member_count: int


class PlatformResponse(BaseModel):
    name: str
    segment_count: int


class SystemStatsResponse(BaseModel):
    supercategories: int
    subcategories: int
    segments: int
    platforms: list[str]


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    model_loaded: bool
