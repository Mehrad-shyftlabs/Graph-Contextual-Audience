"""Tests for the relationships module."""

from __future__ import annotations

import numpy as np

from audience_targeting.models import Segment, SubCategory
from audience_targeting.relationships import (
    compute_parent_segment_ids,
    compute_subcategory_map,
    compute_super_category_map,
)


# ── compute_parent_segment_ids ───────────────────────────────────────────


def test_iab_hierarchy():
    """IAB Tier 2 segments link to Tier 1 via category_id prefix."""
    segments = [
        Segment(
            id="iab_yahoo_IAB2_display", name="Automotive",
            platform="yahoo_dsp", source_file="test.csv",
            hierarchy=["Automotive"],
            metadata={"category_id": "IAB2"},
        ),
        Segment(
            id="iab_yahoo_IAB2-10_display", name="Electric Vehicle",
            platform="yahoo_dsp", source_file="test.csv",
            hierarchy=["Automotive", "Electric Vehicle"],
            metadata={"category_id": "IAB2-10"},
        ),
        Segment(
            id="iab_yahoo_IAB2-5_display", name="Luxury Cars",
            platform="yahoo_dsp", source_file="test.csv",
            hierarchy=["Automotive", "Luxury Cars"],
            metadata={"category_id": "IAB2-5"},
        ),
    ]
    parent_map = compute_parent_segment_ids(segments)
    assert parent_map["iab_yahoo_IAB2-10_display"] == "iab_yahoo_IAB2_display"
    assert parent_map["iab_yahoo_IAB2-5_display"] == "iab_yahoo_IAB2_display"
    assert "iab_yahoo_IAB2_display" not in parent_map  # Tier 1 has no parent


def test_iab_hierarchy_multi_platform():
    """IAB hierarchy works independently per platform."""
    segments = [
        Segment(id="yahoo_IAB2", name="Auto", platform="yahoo_dsp",
                source_file="t.csv", hierarchy=["Auto"], metadata={"category_id": "IAB2"}),
        Segment(id="yahoo_IAB2-10", name="EV", platform="yahoo_dsp",
                source_file="t.csv", hierarchy=["Auto", "EV"], metadata={"category_id": "IAB2-10"}),
        Segment(id="ttd_IAB2", name="Auto", platform="ttd",
                source_file="t.csv", hierarchy=["Auto"], metadata={"category_id": "IAB2"}),
        Segment(id="ttd_IAB2-10", name="EV", platform="ttd",
                source_file="t.csv", hierarchy=["Auto", "EV"], metadata={"category_id": "IAB2-10"}),
    ]
    parent_map = compute_parent_segment_ids(segments)
    assert parent_map["yahoo_IAB2-10"] == "yahoo_IAB2"
    assert parent_map["ttd_IAB2-10"] == "ttd_IAB2"


def test_social_hierarchy():
    """Social platform segments link child to parent category."""
    segments = [
        Segment(id="meta_cat_auto", name="Automotive", platform="meta",
                source_file="t.csv", hierarchy=["Automotive"]),
        Segment(id="meta_luxury", name="Luxury Cars", platform="meta",
                source_file="t.csv", hierarchy=["Automotive", "Luxury"]),
        Segment(id="meta_ev", name="Electric Vehicles", platform="meta",
                source_file="t.csv", hierarchy=["Automotive", "Electric"]),
    ]
    parent_map = compute_parent_segment_ids(segments)
    assert parent_map["meta_luxury"] == "meta_cat_auto"
    assert parent_map["meta_ev"] == "meta_cat_auto"
    assert "meta_cat_auto" not in parent_map


def test_social_hierarchy_inferred_parent():
    """When no explicit category segment exists, first child with that hierarchy[0] becomes parent."""
    segments = [
        Segment(id="tiktok_a", name="Fitness Yoga", platform="tiktok",
                source_file="t.csv", hierarchy=["Fitness", "Yoga"]),
        Segment(id="tiktok_b", name="Fitness Running", platform="tiktok",
                source_file="t.csv", hierarchy=["Fitness", "Running"]),
    ]
    parent_map = compute_parent_segment_ids(segments)
    # tiktok_a is registered as category for "Fitness" since it's first
    # tiktok_b should link to tiktok_a
    assert parent_map["tiktok_b"] == "tiktok_a"
    # tiktok_a should not be its own parent
    assert "tiktok_a" not in parent_map


def test_no_parents_for_flat_segments():
    """Segments with single-level hierarchy have no parents."""
    segments = [
        Segment(id="snap_1", name="Sports", platform="snapchat",
                source_file="t.csv", hierarchy=["Sports"]),
        Segment(id="snap_2", name="Music", platform="snapchat",
                source_file="t.csv", hierarchy=["Music"]),
    ]
    parent_map = compute_parent_segment_ids(segments)
    assert parent_map == {}


def test_empty_segments():
    assert compute_parent_segment_ids([]) == {}


def test_iab_no_parent_for_top_level():
    """IAB category_id without hyphen has no parent."""
    segments = [
        Segment(id="dv360_1", name="Tech", platform="dv360",
                source_file="t.csv", hierarchy=["Tech"], metadata={"category_id": "IAB19"}),
    ]
    parent_map = compute_parent_segment_ids(segments)
    assert parent_map == {}


# ── compute_subcategory_map ──────────────────────────────────────────────


def test_subcategory_map_basic():
    segments = [
        Segment(id="a", name="A", platform="meta", source_file="t.csv"),
        Segment(id="b", name="B", platform="meta", source_file="t.csv"),
        Segment(id="c", name="C", platform="meta", source_file="t.csv"),
    ]
    labels = np.array([0, 1, -1])  # -1 is noise
    sub_map = compute_subcategory_map(segments, labels, [])
    assert sub_map == {"a": 0, "b": 1}
    assert "c" not in sub_map  # noise excluded


def test_subcategory_map_all_noise():
    segments = [
        Segment(id="a", name="A", platform="meta", source_file="t.csv"),
    ]
    labels = np.array([-1])
    assert compute_subcategory_map(segments, labels, []) == {}


# ── compute_super_category_map ───────────────────────────────────────────


def test_super_category_map():
    subs = [
        SubCategory(id=0, name="Luxury", parent_id=10, platforms=set(), member_count=5),
        SubCategory(id=1, name="Economy", parent_id=10, platforms=set(), member_count=3),
        SubCategory(id=2, name="Pets", parent_id=20, platforms=set(), member_count=8),
    ]
    result = compute_super_category_map(subs)
    assert result == {0: 10, 1: 10, 2: 20}


def test_super_category_map_empty():
    assert compute_super_category_map([]) == {}
