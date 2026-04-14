"""Regression test suite — ports the 22 test cases from evaluate.py.

These tests require a running Qdrant instance with populated collections.
Mark as integration tests to skip in unit-test-only runs.

Usage:
    pytest tests/test_evaluation.py -m integration --timeout=120
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from audience_targeting.models import SearchResult


@dataclass
class EvalTestCase:
    """A test query with expected results."""
    query: str
    expected_keywords: list[str]
    expected_platforms: list[str]
    description: str = ""


# ── 15 Short keyword test cases ──────────────────────────────────────────

SHORT_TEST_CASES = [
    EvalTestCase(
        query="luxury SUV shoppers",
        expected_keywords=["luxury", "suv", "vehicle", "automotive", "car"],
        expected_platforms=["meta", "snapchat", "tiktok", "ttd", "yahoo_dsp"],
        description="Cross-platform automotive targeting",
    ),
    EvalTestCase(
        query="pet food buyers",
        expected_keywords=["pet", "food", "dog", "cat", "animal"],
        expected_platforms=["meta", "snapchat", "tiktok", "ttd", "yahoo_dsp"],
        description="Pet vertical with purchase intent",
    ),
    EvalTestCase(
        query="first-time homebuyers",
        expected_keywords=["home", "real estate", "house", "buyer", "property"],
        expected_platforms=["meta", "ttd", "yahoo_dsp"],
        description="Real estate targeting",
    ),
    EvalTestCase(
        query="fitness app users",
        expected_keywords=["fitness", "health", "gym", "exercise", "workout"],
        expected_platforms=["meta", "tiktok", "ttd", "yahoo_dsp"],
        description="Health & fitness",
    ),
    EvalTestCase(
        query="vegan food enthusiasts",
        expected_keywords=["vegan", "food", "plant", "organic", "health"],
        expected_platforms=["meta", "tiktok", "ttd"],
        description="Niche dietary targeting",
    ),
    EvalTestCase(
        query="basketball fans",
        expected_keywords=["basketball", "nba", "sports", "fan"],
        expected_platforms=["meta", "snapchat", "tiktok", "ttd", "yahoo_dsp"],
        description="Sports vertical",
    ),
    EvalTestCase(
        query="fashion and beauty enthusiasts",
        expected_keywords=["fashion", "beauty", "style", "clothing", "cosmetics"],
        expected_platforms=["meta", "snapchat", "tiktok", "ttd", "yahoo_dsp"],
        description="Style & fashion vertical",
    ),
    EvalTestCase(
        query="parents of young children",
        expected_keywords=["baby", "kids", "parent", "child", "family", "maternity"],
        expected_platforms=["meta", "tiktok", "snapchat"],
        description="Demographics / life stage",
    ),
    EvalTestCase(
        query="luxury travel",
        expected_keywords=["travel", "luxury", "hotel", "vacation", "resort"],
        expected_platforms=["meta", "tiktok", "ttd", "yahoo_dsp"],
        description="Travel vertical with luxury qualifier",
    ),
    EvalTestCase(
        query="gaming enthusiasts",
        expected_keywords=["game", "gaming", "gamer", "esport", "console", "mobile"],
        expected_platforms=["meta", "snapchat", "tiktok", "ttd", "dv360"],
        description="Gaming vertical",
    ),
    EvalTestCase(
        query="organic food shoppers",
        expected_keywords=["organic", "food", "health", "natural", "grocery"],
        expected_platforms=["meta", "tiktok", "ttd"],
        description="Food with organic qualifier",
    ),
    EvalTestCase(
        query="music festival goers",
        expected_keywords=["music", "concert", "festival", "live", "entertainment"],
        expected_platforms=["meta", "snapchat", "tiktok", "ttd", "yahoo_dsp"],
        description="Entertainment / events",
    ),
    EvalTestCase(
        query="electric vehicle buyers",
        expected_keywords=["electric", "ev", "vehicle", "car", "automotive", "hybrid"],
        expected_platforms=["meta", "tiktok", "ttd", "yahoo_dsp"],
        description="Auto sub-vertical",
    ),
    EvalTestCase(
        query="DIY home improvement",
        expected_keywords=["home", "diy", "improvement", "repair", "garden", "remodel"],
        expected_platforms=["meta", "snapchat", "tiktok", "ttd", "yahoo_dsp"],
        description="Home & garden vertical",
    ),
    EvalTestCase(
        query="college students",
        expected_keywords=["college", "student", "education", "university", "school"],
        expected_platforms=["meta", "ttd", "yahoo_dsp"],
        description="Education / demographics",
    ),
]

# ── 7 Long brief test cases ─────────────────────────────────────────────

LONG_BRIEF_TEST_CASES = [
    EvalTestCase(
        query=(
            "We're launching a premium SUV campaign targeting affluent families. "
            "The ideal audience owns luxury vehicles, has high household income, "
            "and shows interest in family activities and travel."
        ),
        expected_keywords=["luxury", "suv", "vehicle", "family", "travel", "income"],
        expected_platforms=["meta", "yahoo_dsp", "tiktok", "ttd"],
        description="Multi-topic luxury SUV brief",
    ),
    EvalTestCase(
        query=(
            "Our client sells organic pet food. We need to reach pet owners who "
            "care about nutrition and healthy eating. They're typically health-conscious "
            "millennials who also buy organic groceries for themselves."
        ),
        expected_keywords=["pet", "food", "organic", "health", "grocery"],
        expected_platforms=["meta", "tiktok", "ttd", "yahoo_dsp"],
        description="Multi-topic organic pet food brief",
    ),
    EvalTestCase(
        query=(
            "We're promoting a new fitness app for home workouts. Our target audience "
            "includes gym members, yoga practitioners, and people interested in nutrition. "
            "They tend to be tech-savvy and use health tracking devices."
        ),
        expected_keywords=["fitness", "workout", "gym", "yoga", "health", "app"],
        expected_platforms=["meta", "tiktok", "ttd", "yahoo_dsp"],
        description="Multi-topic fitness app brief",
    ),
    EvalTestCase(
        query=(
            "The campaign targets parents shopping for back-to-school supplies. "
            "We want to reach families with school-age children who are interested "
            "in education, children's clothing, and electronics for students."
        ),
        expected_keywords=["parent", "school", "education", "children", "family", "electronics"],
        expected_platforms=["meta", "tiktok", "snapchat", "ttd"],
        description="Multi-topic back-to-school brief",
    ),
    EvalTestCase(
        query=(
            "Our client is a real estate developer marketing luxury condos in Miami. "
            "The target audience includes high-net-worth individuals interested in "
            "investment properties, luxury lifestyle, and travel to Florida."
        ),
        expected_keywords=["real estate", "luxury", "property", "investment", "travel"],
        expected_platforms=["meta", "yahoo_dsp", "ttd"],
        description="Multi-topic luxury real estate brief",
    ),
    EvalTestCase(
        query=(
            "We need to reach sports fans who follow basketball and football. "
            "They should be interested in sports betting, fantasy sports, and "
            "streaming services for live games."
        ),
        expected_keywords=["sports", "basketball", "football", "betting", "streaming", "fantasy"],
        expected_platforms=["meta", "tiktok", "snapchat", "ttd", "yahoo_dsp"],
        description="Multi-topic sports brief",
    ),
    EvalTestCase(
        query=(
            "Targeting food enthusiasts who love cooking at home and dining out. "
            "They're interested in gourmet ingredients, kitchen appliances, "
            "restaurant reviews, and food delivery services."
        ),
        expected_keywords=["food", "cooking", "restaurant", "kitchen", "delivery", "gourmet"],
        expected_platforms=["meta", "tiktok", "snapchat", "ttd"],
        description="Multi-topic food enthusiast brief",
    ),
]

ALL_TEST_CASES = SHORT_TEST_CASES + LONG_BRIEF_TEST_CASES


# ── Evaluation helpers ───────────────────────────────────────────────────


def evaluate_result(result: SearchResult, test_case: EvalTestCase) -> dict:
    """Evaluate a single search result against expected outcomes."""
    all_names = set()
    for ms in result.matched_subcategories:
        all_names.add(ms.sub_category.name.lower())
    for platform, segs in result.segments_by_platform.items():
        for seg, score in segs:
            all_names.add(seg.name.lower())
            for h in seg.hierarchy:
                all_names.add(h.lower())

    all_text = " ".join(all_names)

    keyword_hits = sum(1 for kw in test_case.expected_keywords if kw.lower() in all_text)
    keyword_recall = keyword_hits / len(test_case.expected_keywords) if test_case.expected_keywords else 0

    matched_platforms = set(result.segments_by_platform.keys())
    platform_hits = len(matched_platforms & set(test_case.expected_platforms))
    platform_coverage = (
        platform_hits / len(test_case.expected_platforms)
        if test_case.expected_platforms
        else 0
    )

    return {
        "keyword_recall": keyword_recall,
        "keyword_hits": keyword_hits,
        "keyword_total": len(test_case.expected_keywords),
        "platform_coverage": platform_coverage,
        "platform_hits": platform_hits,
        "platform_total": len(test_case.expected_platforms),
    }


# ── Integration test (requires running Qdrant + populated data) ──────────


@pytest.mark.integration
class TestEvaluationRegression:
    """Regression tests that validate search quality against the 22 evaluation cases.

    Requires a running Qdrant instance with populated collections.
    Run with: pytest tests/test_evaluation.py -m integration
    """

    @pytest.fixture(autouse=True)
    def setup_engine(self):
        from audience_targeting.search_engine import create_engine
        self.engine = create_engine()

    @pytest.mark.parametrize(
        "test_case",
        SHORT_TEST_CASES,
        ids=[tc.description for tc in SHORT_TEST_CASES],
    )
    def test_short_query(self, test_case: EvalTestCase):
        result = self.engine.search(test_case.query)
        metrics = evaluate_result(result, test_case)
        assert metrics["keyword_recall"] >= 0.6, (
            f"Keyword recall {metrics['keyword_recall']:.0%} < 60% for '{test_case.query}' "
            f"({metrics['keyword_hits']}/{metrics['keyword_total']})"
        )
        assert metrics["platform_coverage"] >= 0.6, (
            f"Platform coverage {metrics['platform_coverage']:.0%} < 60% for '{test_case.query}' "
            f"({metrics['platform_hits']}/{metrics['platform_total']})"
        )

    @pytest.mark.parametrize(
        "test_case",
        LONG_BRIEF_TEST_CASES,
        ids=[tc.description for tc in LONG_BRIEF_TEST_CASES],
    )
    def test_long_brief(self, test_case: EvalTestCase):
        result = self.engine.search(test_case.query)
        metrics = evaluate_result(result, test_case)
        assert metrics["keyword_recall"] >= 0.5, (
            f"Keyword recall {metrics['keyword_recall']:.0%} < 50% for brief "
            f"({metrics['keyword_hits']}/{metrics['keyword_total']})"
        )
        assert metrics["platform_coverage"] >= 0.5, (
            f"Platform coverage {metrics['platform_coverage']:.0%} < 50% for brief "
            f"({metrics['platform_hits']}/{metrics['platform_total']})"
        )

    def test_aggregate_keyword_recall(self):
        """Average keyword recall across all 22 cases should be >= 80%."""
        recalls = []
        for tc in ALL_TEST_CASES:
            result = self.engine.search(tc.query)
            metrics = evaluate_result(result, tc)
            recalls.append(metrics["keyword_recall"])
        avg = sum(recalls) / len(recalls)
        assert avg >= 0.8, f"Average keyword recall {avg:.1%} < 80%"

    def test_aggregate_platform_coverage(self):
        """Average platform coverage across all 22 cases should be 100%."""
        coverages = []
        for tc in ALL_TEST_CASES:
            result = self.engine.search(tc.query)
            metrics = evaluate_result(result, tc)
            coverages.append(metrics["platform_coverage"])
        avg = sum(coverages) / len(coverages)
        assert avg >= 1.0, f"Average platform coverage {avg:.1%} < 100%"
