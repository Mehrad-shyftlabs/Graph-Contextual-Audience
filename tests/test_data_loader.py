"""Tests for data loading."""

from audience_targeting.models import Segment


def test_segment_embed_text_with_description():
    seg = Segment(
        id="test_1", name="Luxury Cars", platform="meta", source_file="test.csv",
        hierarchy=["Automotive", "Luxury"],
        description="People interested in high-end luxury vehicles",
    )
    text = seg.embed_text
    assert "Luxury Cars" in text
    assert "Automotive > Luxury" in text
    assert "high-end luxury vehicles" in text


def test_segment_embed_text_without_description():
    seg = Segment(
        id="test_1", name="Luxury Cars", platform="meta", source_file="test.csv",
        hierarchy=["Automotive", "Luxury"],
    )
    text = seg.embed_text
    assert "Luxury Cars" in text
    assert "Automotive > Luxury" in text


def test_segment_embed_text_name_only():
    seg = Segment(
        id="test_1", name="Luxury Cars", platform="meta", source_file="test.csv",
    )
    assert seg.embed_text == "Luxury Cars"
