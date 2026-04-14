"""Shared data models used across the build pipeline, search engine, and API."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ── Core data model (used by data loader, embedder, clustering) ──────────


@dataclass
class Segment:
    """Unified representation of an audience segment across all platforms."""

    id: str
    name: str
    platform: str  # meta, tiktok, snapchat, yahoo_dsp, ttd, dv360
    source_file: str
    hierarchy: list[str] = field(default_factory=list)
    segment_type: str = ""  # interest, behavior, iab_content, lifestyle, app
    audience_size: int | None = None
    metadata: dict = field(default_factory=dict)
    description: str | None = None  # filled by enrichment
    embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def embed_text(self) -> str:
        """Text used for embedding: name + hierarchy context + description."""
        if self.description:
            return f"{self.name} | {' > '.join(self.hierarchy)} | {self.description}"
        if self.hierarchy:
            return f"{self.name} | {' > '.join(self.hierarchy)}"
        return self.name


# ── Clustering output models ─────────────────────────────────────────────


@dataclass
class SuperCategory:
    """Layer 0: Broad audience super-category (e.g., 'Automotive', 'Sports')."""

    id: int
    name: str
    subcategory_ids: list[int] = field(default_factory=list)
    centroid: np.ndarray | None = field(default=None, repr=False)
    platforms: set[str] = field(default_factory=set)
    member_count: int = 0


@dataclass
class SubCategory:
    """Layer 1: Specific audience sub-category (e.g., 'Luxury Vehicles')."""

    id: int
    name: str
    parent_id: int = -1
    segment_ids: list[str] = field(default_factory=list)
    centroid: np.ndarray | None = field(default=None, repr=False)
    platforms: set[str] = field(default_factory=set)
    member_count: int = 0


# ── Search result models ─────────────────────────────────────────────────


@dataclass
class MatchedSubCategory:
    """A matched sub-category with similarity score."""

    sub_category: SubCategory
    super_category: SuperCategory | None
    score: float
    source_sentence: str = ""


@dataclass
class Recommendation:
    """A recommended related sub-category."""

    sub_id: int
    name: str
    relation: str  # "related", "broader", "narrower"
    score: float
    member_count: int = 0
    platforms: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Full search result with matches, recommendations, and scope options."""

    query: str
    matched_subcategories: list[MatchedSubCategory] = field(default_factory=list)
    segments_by_platform: dict[str, list[tuple[Segment, float]]] = field(default_factory=dict)
    recommendations: list[Recommendation] = field(default_factory=list)
    broadening_options: list[Recommendation] = field(default_factory=list)
    narrowing_options: list[Recommendation] = field(default_factory=list)
    sentence_topics: dict[str, list[str]] = field(default_factory=dict)
