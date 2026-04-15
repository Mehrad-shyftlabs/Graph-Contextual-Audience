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
    assert data["segments_by_platform"]["meta"][0]["match_label"] == "match"
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


# ── Match threshold filtering ────────────────────────────────────────────


def test_search_filters_low_score_segments(client, mock_engine):
    """Segments scoring below partial_match threshold (0.5) are excluded."""
    low_seg = Segment(
        id="meta_low", name="Weak Match", platform="meta", source_file="",
        hierarchy=["Misc"], segment_type="interest",
    )
    mid_seg = Segment(
        id="meta_mid", name="Partial", platform="meta", source_file="",
        hierarchy=["Automotive"], segment_type="interest",
    )
    high_seg = Segment(
        id="meta_high", name="Strong", platform="meta", source_file="",
        hierarchy=["Automotive", "Luxury"], segment_type="interest",
    )
    result = SearchResult(
        query="test",
        matched_subcategories=[],
        segments_by_platform={
            "meta": [(high_seg, 0.85), (mid_seg, 0.60), (low_seg, 0.30)],
        },
        recommendations=[],
        broadening_options=[],
        narrowing_options=[],
        sentence_topics={},
    )
    mock_engine.search.return_value = result
    resp = client.post("/v1/search", json={"query": "test"})
    data = resp.json()
    meta_segs = data["segments_by_platform"]["meta"]
    assert len(meta_segs) == 2  # low_seg filtered out
    assert meta_segs[0]["match_label"] == "match"
    assert meta_segs[0]["name"] == "Strong"
    assert meta_segs[1]["match_label"] == "partial_match"
    assert meta_segs[1]["name"] == "Partial"


def test_search_filters_entire_platform_when_all_below(client, mock_engine):
    """If all segments for a platform are below threshold, that platform is omitted."""
    low_seg = Segment(
        id="snap_1", name="Weak", platform="snapchat", source_file="",
        hierarchy=["Misc"], segment_type="interest",
    )
    result = SearchResult(
        query="test",
        matched_subcategories=[],
        segments_by_platform={
            "snapchat": [(low_seg, 0.3)],
        },
        recommendations=[],
        broadening_options=[],
        narrowing_options=[],
        sentence_topics={},
    )
    mock_engine.search.return_value = result
    resp = client.post("/v1/search", json={"query": "test"})
    data = resp.json()
    assert "snapchat" not in data["segments_by_platform"]
    assert data["metadata"]["total_segments"] == 0


# ── Platforms endpoint ───────────────────────────────────────────────────


def test_list_platforms(client):
    resp = client.get("/v1/platforms")
    assert resp.status_code == 200
    platforms = resp.json()
    assert isinstance(platforms, list)
    assert "meta" in platforms
    assert "tiktok" in platforms


# ── Auth tests ──────────────────────────────────────────────────────────


def test_auth_disabled_by_default(client, mock_engine, sample_search_result):
    """When api_key is None (default), all requests pass without a key."""
    mock_engine.search.return_value = sample_search_result
    resp = client.post("/v1/search", json={"query": "test"})
    assert resp.status_code == 200


def test_auth_rejects_missing_key(client, mock_engine, sample_search_result):
    """When api_key is set, requests without a key get 401."""
    import audience_targeting.api as api_module
    original = api_module._settings.api_key
    api_module._settings.api_key = "secret"
    try:
        resp = client.post("/v1/search", json={"query": "test"})
        assert resp.status_code == 401
    finally:
        api_module._settings.api_key = original


def test_auth_rejects_wrong_key(client, mock_engine, sample_search_result):
    """When api_key is set, a wrong key gets 401."""
    import audience_targeting.api as api_module
    original = api_module._settings.api_key
    api_module._settings.api_key = "secret"
    try:
        resp = client.post(
            "/v1/search",
            json={"query": "test"},
            headers={"X-API-Key": "wrong"},
        )
        assert resp.status_code == 401
    finally:
        api_module._settings.api_key = original


def test_auth_accepts_correct_key(client, mock_engine, sample_search_result):
    """When api_key is set, the correct key passes through."""
    import audience_targeting.api as api_module
    original = api_module._settings.api_key
    api_module._settings.api_key = "secret"
    mock_engine.search.return_value = sample_search_result
    try:
        resp = client.post(
            "/v1/search",
            json={"query": "test"},
            headers={"X-API-Key": "secret"},
        )
        assert resp.status_code == 200
    finally:
        api_module._settings.api_key = original


def test_health_exempt_from_auth(client):
    """Health and ready endpoints never require an API key."""
    import audience_targeting.api as api_module
    original = api_module._settings.api_key
    api_module._settings.api_key = "secret"
    try:
        resp = client.get("/health")
        assert resp.status_code == 200
        resp = client.get("/ready")
        assert resp.status_code == 200
    finally:
        api_module._settings.api_key = original


# ── Request ID tests ────────────────────────────────────────────────────


def test_request_id_returned(client):
    """When X-Request-ID is sent, the same value is returned."""
    resp = client.get("/health", headers={"X-Request-ID": "test-123"})
    assert resp.headers.get("X-Request-ID") == "test-123"


def test_request_id_generated(client):
    """When no X-Request-ID is sent, one is generated and returned."""
    resp = client.get("/health")
    request_id = resp.headers.get("X-Request-ID")
    assert request_id is not None
    assert len(request_id) > 0


# ── Account-level threshold tests ───────────────────────────────────────


def test_search_with_custom_match_threshold(client, mock_engine):
    """Account-level match_threshold overrides server default."""
    seg = Segment(
        id="meta_1", name="Strong", platform="meta", source_file="",
        hierarchy=["Auto"], segment_type="interest",
    )
    result = SearchResult(
        query="test",
        matched_subcategories=[],
        segments_by_platform={"meta": [(seg, 0.85)]},
        recommendations=[],
        broadening_options=[],
        narrowing_options=[],
        sentence_topics={},
    )
    mock_engine.search.return_value = result

    # Default threshold (0.7): score 0.85 -> "match"
    resp = client.post("/v1/search", json={"query": "test"})
    data = resp.json()
    assert data["segments_by_platform"]["meta"][0]["match_label"] == "match"

    # Custom threshold (0.9): score 0.85 -> "partial_match"
    resp = client.post("/v1/search", json={
        "query": "test",
        "match_threshold": 0.9,
    })
    data = resp.json()
    assert data["segments_by_platform"]["meta"][0]["match_label"] == "partial_match"


def test_search_with_custom_partial_threshold(client, mock_engine):
    """Account-level partial_match_threshold filters out low-scoring segments."""
    seg = Segment(
        id="meta_1", name="Weak", platform="meta", source_file="",
        hierarchy=["Misc"], segment_type="interest",
    )
    result = SearchResult(
        query="test",
        matched_subcategories=[],
        segments_by_platform={"meta": [(seg, 0.55)]},
        recommendations=[],
        broadening_options=[],
        narrowing_options=[],
        sentence_topics={},
    )
    mock_engine.search.return_value = result

    # Default partial threshold (0.5): score 0.55 -> "partial_match"
    resp = client.post("/v1/search", json={"query": "test"})
    data = resp.json()
    assert "meta" in data["segments_by_platform"]

    # Custom partial threshold (0.6): score 0.55 -> filtered out
    resp = client.post("/v1/search", json={
        "query": "test",
        "partial_match_threshold": 0.6,
    })
    data = resp.json()
    assert "meta" not in data["segments_by_platform"]
