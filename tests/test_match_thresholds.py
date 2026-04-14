"""Tests for match quality thresholds and classification."""

from audience_targeting.settings import Settings


def test_classify_match_above_match_threshold():
    s = Settings()
    assert s.classify_match(0.85, "meta") == "match"
    assert s.classify_match(0.70, "meta") == "match"


def test_classify_match_partial():
    s = Settings()
    assert s.classify_match(0.65, "meta") == "partial_match"
    assert s.classify_match(0.50, "meta") == "partial_match"


def test_classify_match_below_threshold():
    s = Settings()
    assert s.classify_match(0.49, "meta") is None
    assert s.classify_match(0.0, "meta") is None


def test_default_thresholds_same_across_platforms():
    s = Settings()
    for platform in s.platforms:
        match_thr, partial_thr = s.get_match_thresholds(platform)
        assert match_thr == 0.7
        assert partial_thr == 0.5


def test_per_platform_override():
    s = Settings(platform_match_thresholds={
        "meta": {"match": 0.8, "partial_match": 0.6},
    })
    # Meta uses overrides
    match_thr, partial_thr = s.get_match_thresholds("meta")
    assert match_thr == 0.8
    assert partial_thr == 0.6

    # TikTok falls back to defaults
    match_thr, partial_thr = s.get_match_thresholds("tiktok")
    assert match_thr == 0.7
    assert partial_thr == 0.5


def test_per_platform_partial_override():
    """Override only one threshold; the other falls back to default."""
    s = Settings(platform_match_thresholds={
        "snapchat": {"match": 0.75},
    })
    match_thr, partial_thr = s.get_match_thresholds("snapchat")
    assert match_thr == 0.75
    assert partial_thr == 0.5  # default


def test_classify_with_platform_override():
    s = Settings(platform_match_thresholds={
        "tiktok": {"match": 0.8, "partial_match": 0.6},
    })
    # 0.65 is partial for default but below partial for tiktok
    assert s.classify_match(0.65, "meta") == "partial_match"
    assert s.classify_match(0.65, "tiktok") == "partial_match"
    assert s.classify_match(0.55, "tiktok") is None  # below tiktok's 0.6
    assert s.classify_match(0.55, "meta") == "partial_match"  # still above default 0.5
