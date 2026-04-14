"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from audience_targeting.models import Segment, SubCategory, SuperCategory
from audience_targeting.settings import Settings


@pytest.fixture
def settings():
    """Test settings with defaults."""
    return Settings(qdrant_url="http://localhost:6333")


@pytest.fixture
def sample_segments() -> list[Segment]:
    """A small set of test segments across platforms."""
    return [
        Segment(id="meta_1", name="Luxury Cars", platform="meta", source_file="test.csv",
                hierarchy=["Automotive", "Luxury"], segment_type="interest", audience_size=1000000,
                description="People interested in high-end luxury vehicles"),
        Segment(id="tiktok_1", name="Vehicle Enthusiasts", platform="tiktok", source_file="test.csv",
                hierarchy=["Vehicles & Transportation"], segment_type="interest",
                description="Users who engage with automotive content"),
        Segment(id="snap_1", name="Auto Enthusiasts", platform="snapchat", source_file="test.csv",
                hierarchy=["Automotive"], segment_type="interest",
                description="Snapchat users interested in cars and vehicles"),
        Segment(id="iab_yahoo_IAB2_display", name="Automotive", platform="yahoo_dsp", source_file="test.csv",
                hierarchy=["Automotive"], segment_type="iab_content",
                metadata={"category_id": "IAB2"}),
        Segment(id="iab_yahoo_IAB2-10_display", name="Electric Vehicle", platform="yahoo_dsp", source_file="test.csv",
                hierarchy=["Automotive", "Electric Vehicle"], segment_type="iab_content",
                metadata={"category_id": "IAB2-10"}),
        Segment(id="meta_2", name="Pet Owners", platform="meta", source_file="test.csv",
                hierarchy=["Pets", "Pet Owners"], segment_type="interest", audience_size=5000000,
                description="People who own pets"),
        Segment(id="tiktok_2", name="Pet Lovers", platform="tiktok", source_file="test.csv",
                hierarchy=["Pets & Animals"], segment_type="interest",
                description="Users who love pets and animals"),
    ]


@pytest.fixture
def sample_embeddings(sample_segments) -> np.ndarray:
    """Random embeddings for sample segments (L2-normalized)."""
    rng = np.random.RandomState(42)
    embs = rng.randn(len(sample_segments), 384).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms


@pytest.fixture
def sample_super_categories() -> list[SuperCategory]:
    return [
        SuperCategory(id=0, name="Automotive", subcategory_ids=[0, 1],
                      centroid=np.zeros(384, dtype=np.float32), platforms={"meta", "tiktok", "snapchat", "yahoo_dsp"},
                      member_count=5),
        SuperCategory(id=1, name="Pets", subcategory_ids=[2],
                      centroid=np.zeros(384, dtype=np.float32), platforms={"meta", "tiktok"},
                      member_count=2),
    ]


@pytest.fixture
def sample_sub_categories() -> list[SubCategory]:
    return [
        SubCategory(id=0, name="Luxury Vehicles", parent_id=0,
                    segment_ids=["meta_1", "snap_1"],
                    centroid=np.zeros(384, dtype=np.float32),
                    platforms={"meta", "snapchat"}, member_count=2),
        SubCategory(id=1, name="Electric Vehicles", parent_id=0,
                    segment_ids=["tiktok_1", "iab_yahoo_IAB2_display", "iab_yahoo_IAB2-10_display"],
                    centroid=np.zeros(384, dtype=np.float32),
                    platforms={"tiktok", "yahoo_dsp"}, member_count=3),
        SubCategory(id=2, name="Pet Owners", parent_id=1,
                    segment_ids=["meta_2", "tiktok_2"],
                    centroid=np.zeros(384, dtype=np.float32),
                    platforms={"meta", "tiktok"}, member_count=2),
    ]
