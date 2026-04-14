"""Tests for the search engine module."""

from __future__ import annotations

import numpy as np
import pytest

from audience_targeting.models import Segment
from audience_targeting.search_engine import _cosine_sim, _to_segment, chunk_brief


# ── chunk_brief ──────────────────────────────────────────────────────────


def test_chunk_brief_short_query():
    """Short keyword query returns as a single chunk."""
    assert chunk_brief("luxury SUV") == ["luxury SUV"]


def test_chunk_brief_multi_sentence():
    """Multi-sentence brief splits correctly."""
    text = (
        "We target luxury SUV shoppers. They also enjoy golf. "
        "Budget is flexible for premium placements."
    )
    chunks = chunk_brief(text)
    assert len(chunks) == 3
    assert "luxury SUV" in chunks[0]
    assert "golf" in chunks[1]


def test_chunk_brief_filters_short_fragments():
    """Fragments <= 10 chars are filtered out."""
    text = "Buy now. We target affluent families who travel frequently."
    chunks = chunk_brief(text)
    # "Buy now." is only 8 chars — should be dropped
    assert all(len(c) > 10 for c in chunks)


def test_chunk_brief_empty_string():
    """Empty string returns original text as single chunk."""
    assert chunk_brief("") == [""]


def test_chunk_brief_no_sentence_end():
    """Text without sentence-ending punctuation stays as one chunk."""
    text = "luxury SUV shoppers who are affluent"
    assert chunk_brief(text) == [text]


# ── _cosine_sim ──────────────────────────────────────────────────────────


def test_cosine_sim_identical():
    a = [1.0, 0.0, 0.0]
    assert abs(_cosine_sim(a, a) - 1.0) < 1e-7


def test_cosine_sim_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine_sim(a, b)) < 1e-7


def test_cosine_sim_opposite():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert abs(_cosine_sim(a, b) + 1.0) < 1e-7


def test_cosine_sim_zero_vector():
    a = [0.0, 0.0]
    b = [1.0, 0.0]
    assert _cosine_sim(a, b) == 0.0


def test_cosine_sim_numpy_input():
    a = np.array([3.0, 4.0])
    b = np.array([4.0, 3.0])
    expected = (3 * 4 + 4 * 3) / (5.0 * 5.0)
    assert abs(_cosine_sim(a, b) - expected) < 1e-7


# ── _to_segment ──────────────────────────────────────────────────────────


def test_to_segment_full():
    result = {
        "segment_id": "meta_123",
        "name": "Luxury Cars",
        "platform": "meta",
        "hierarchy": ["Automotive", "Luxury"],
        "segment_type": "interest",
        "audience_size": 1000000,
        "description": "People interested in luxury cars",
    }
    seg = _to_segment(result)
    assert isinstance(seg, Segment)
    assert seg.id == "meta_123"
    assert seg.name == "Luxury Cars"
    assert seg.platform == "meta"
    assert seg.hierarchy == ["Automotive", "Luxury"]
    assert seg.audience_size == 1000000


def test_to_segment_missing_fields():
    result = {"segment_id": "x", "name": "Test", "platform": "tiktok"}
    seg = _to_segment(result)
    assert seg.id == "x"
    assert seg.hierarchy == []
    assert seg.audience_size is None
    assert seg.description is None


def test_to_segment_empty_dict():
    seg = _to_segment({})
    assert seg.id == ""
    assert seg.name == ""
    assert seg.platform == ""
