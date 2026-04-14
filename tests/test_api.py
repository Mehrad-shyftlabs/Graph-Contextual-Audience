"""Tests for the FastAPI API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from audience_targeting.api_models import SearchRequest
from audience_targeting.models import (
    MatchedSubCategory,
    Recommendation,
    SearchResult,
    Segment,
    SubCategory,
)
from audience_targeting.settings import Settings


@pytest.fixture
def mock_engine():
    """Create a mock AudienceSearchEngine."""
    engine = MagicMock()
    engine.client = MagicMock()
    engine.client.get_collections.return_value = MagicMock(collections=[])
    return engine


@pytest.fixture
def client(mock_engine):
    """Create a TestClient with mocked engine."""
    from audience_targeting.api import _get_engine, _get_settings, app

    test_settings = Settings(qdrant_url="http://localhost:6333")

    app.dependency_overrides = {}
    # Patch globals directly for testing
    import audience_targeting.api as api_module

    original_engine = api_module._engine
    original_settings = api_module._settings
    api_module._engine = mock_engine
    api_module._settings = test_settings

    yield TestClient(app)

    api_module._engine = original_engine
    api_module._settings = original_settings


@pytest.fixture
def sample_search_result():
    """A minimal SearchResult for testing the /v1/search endpoint."""
    seg = Segment(
        id="meta_1", name="Luxury Cars", platform="meta", source_file="",
        hierarchy=["Automotive", "Luxury"], segment_type="interest",
        audience_size=1000000, description="Luxury car lovers",
    )
    sub = SubCategory(
        id=0, name="Luxury Vehicles", parent_id=0,
        platforms={"meta", "tiktok"}, member_count=5,
    )
    return SearchResult(
        query="luxury SUV shoppers",
        matched_subcategories=[
            MatchedSubCategory(
                sub_category=sub, super_category=None,
                score=0.85, source_sentence="luxury SUV shoppers",
            ),
        ],
        segments_by_platform={
            "meta": [(seg, 0.92)],
        },
        recommendations=[
            Recommendation(
                sub_id=1, name="Premium Travel", relation="related",
                score=0.72, member_count=10, platforms=["meta", "tiktok"],
            ),
        ],
        broadening_options=[],
        narrowing_options=[],
        sentence_topics={"luxury SUV shoppers": ["Automotive"]},
    )


# ── Health endpoints ─────────────────────────────────────────────────────


def test_health_ok(client, mock_engine):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["qdrant_connected"] is True
    assert data["model_loaded"] is True


def test_ready_ok(client, mock_engine):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_ready_qdrant_down(client, mock_engine):
    mock_engine.client.get_collections.side_effect = ConnectionError("no connection")
    resp = client.get("/ready")
    assert resp.status_code == 503


# ── Search endpoint ──────────────────────────────────────────────────────


def test_search_success(client, mock_engine, sample_search_result):
    mock_engine.search.return_value = sample_search_result

    resp = client.post("/v1/search", json={"query": "luxury SUV shoppers"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "luxury SUV shoppers"
    assert "meta" in data["segments_by_platform"]
    assert len(data["segments_by_platform"]["meta"]) == 1
    assert data["segments_by_platform"]["meta"][0]["name"] == "Luxury Cars"
    assert data["segments_by_platform"]["meta"][0]["score"] == 0.92
    assert len(data["matched_subcategories"]) == 1
    assert data["matched_subcategories"][0]["name"] == "Luxury Vehicles"
    assert len(data["recommendations"]) == 1
    assert data["metadata"]["total_segments"] == 1
    assert data["metadata"]["platforms_matched"] == 1


def test_search_empty_query(client):
    resp = client.post("/v1/search", json={"query": ""})
    assert resp.status_code == 422  # Pydantic validation: min_length=1


def test_search_with_platform_filter(client, mock_engine, sample_search_result):
    mock_engine.search.return_value = sample_search_result
    resp = client.post("/v1/search", json={
        "query": "luxury SUV",
        "platforms": ["meta"],
        "top_k": 5,
    })
    assert resp.status_code == 200
    mock_engine.search.assert_called_once_with(
        query="luxury SUV", platforms=["meta"], top_k_segments=5,
    )


def test_search_no_recommendations(client, mock_engine, sample_search_result):
    mock_engine.search.return_value = sample_search_result
    resp = client.post("/v1/search", json={
        "query": "luxury SUV",
        "include_recommendations": False,
    })
    data = resp.json()
    assert data["recommendations"] == []


def test_search_no_scope_options(client, mock_engine, sample_search_result):
    mock_engine.search.return_value = sample_search_result
    resp = client.post("/v1/search", json={
        "query": "luxury SUV",
        "include_scope_options": False,
    })
    data = resp.json()
    assert data["broadening_options"] == []
    assert data["narrowing_options"] == []


# ── Platforms endpoint ───────────────────────────────────────────────────


def test_list_platforms(client):
    resp = client.get("/v1/platforms")
    assert resp.status_code == 200
    platforms = resp.json()
    assert isinstance(platforms, list)
    assert "meta" in platforms
    assert "tiktok" in platforms
