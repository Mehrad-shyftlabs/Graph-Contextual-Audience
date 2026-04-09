"""Evaluation harness for audience matching quality.

Supports both v1 (short keyword) and v2 (long brief) test cases,
plus v1 vs v2 comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

from query import AudienceSearchEngine, AudienceSearchEngineV2, SearchResult


@dataclass
class TestCase:
    """A test query with expected results."""
    query: str
    expected_keywords: list[str]
    expected_platforms: list[str]
    description: str = ""


# ── Short keyword test cases (v1 parity) ──────────────────────────────────

TEST_CASES = [
    TestCase(
        query="luxury SUV shoppers",
        expected_keywords=["luxury", "suv", "vehicle", "automotive", "car"],
        expected_platforms=["meta", "snapchat", "tiktok", "ttd", "yahoo_dsp"],
        description="Cross-platform automotive targeting",
    ),
    TestCase(
        query="pet food buyers",
        expected_keywords=["pet", "food", "dog", "cat", "animal"],
        expected_platforms=["meta", "snapchat", "tiktok", "ttd", "yahoo_dsp"],
        description="Pet vertical with purchase intent",
    ),
    TestCase(
        query="first-time homebuyers",
        expected_keywords=["home", "real estate", "house", "buyer", "property"],
        expected_platforms=["meta", "ttd", "yahoo_dsp"],
        description="Real estate targeting",
    ),
    TestCase(
        query="fitness app users",
        expected_keywords=["fitness", "health", "gym", "exercise", "workout"],
        expected_platforms=["meta", "tiktok", "ttd", "yahoo_dsp"],
        description="Health & fitness",
    ),
    TestCase(
        query="vegan food enthusiasts",
        expected_keywords=["vegan", "food", "plant", "organic", "health"],
        expected_platforms=["meta", "tiktok", "ttd"],
        description="Niche dietary targeting",
    ),
    TestCase(
        query="basketball fans",
        expected_keywords=["basketball", "nba", "sports", "fan"],
        expected_platforms=["meta", "snapchat", "tiktok", "ttd", "yahoo_dsp"],
        description="Sports vertical",
    ),
    TestCase(
        query="fashion and beauty enthusiasts",
        expected_keywords=["fashion", "beauty", "style", "clothing", "cosmetics"],
        expected_platforms=["meta", "snapchat", "tiktok", "ttd", "yahoo_dsp"],
        description="Style & fashion vertical",
    ),
    TestCase(
        query="parents of young children",
        expected_keywords=["baby", "kids", "parent", "child", "family", "maternity"],
        expected_platforms=["meta", "tiktok", "snapchat"],
        description="Demographics / life stage",
    ),
    TestCase(
        query="luxury travel",
        expected_keywords=["travel", "luxury", "hotel", "vacation", "resort"],
        expected_platforms=["meta", "tiktok", "ttd", "yahoo_dsp"],
        description="Travel vertical with luxury qualifier",
    ),
    TestCase(
        query="gaming enthusiasts",
        expected_keywords=["game", "gaming", "gamer", "esport", "console", "mobile"],
        expected_platforms=["meta", "snapchat", "tiktok", "ttd", "dv360"],
        description="Gaming vertical",
    ),
    TestCase(
        query="organic food shoppers",
        expected_keywords=["organic", "food", "health", "natural", "grocery"],
        expected_platforms=["meta", "tiktok", "ttd"],
        description="Food with organic qualifier",
    ),
    TestCase(
        query="music festival goers",
        expected_keywords=["music", "concert", "festival", "live", "entertainment"],
        expected_platforms=["meta", "snapchat", "tiktok", "ttd", "yahoo_dsp"],
        description="Entertainment / events",
    ),
    TestCase(
        query="electric vehicle buyers",
        expected_keywords=["electric", "ev", "vehicle", "car", "automotive", "hybrid"],
        expected_platforms=["meta", "tiktok", "ttd", "yahoo_dsp"],
        description="Auto sub-vertical",
    ),
    TestCase(
        query="DIY home improvement",
        expected_keywords=["home", "diy", "improvement", "repair", "garden", "remodel"],
        expected_platforms=["meta", "snapchat", "tiktok", "ttd", "yahoo_dsp"],
        description="Home & garden vertical",
    ),
    TestCase(
        query="college students",
        expected_keywords=["college", "student", "education", "university", "school"],
        expected_platforms=["meta", "ttd", "yahoo_dsp"],
        description="Education / demographics",
    ),
]

# ── Long brief test cases (v2 specific) ───────────────────────────────────

LONG_BRIEF_TEST_CASES = [
    TestCase(
        query=(
            "We're launching a premium SUV campaign targeting affluent families. "
            "The ideal audience owns luxury vehicles, has high household income, "
            "and shows interest in family activities and travel."
        ),
        expected_keywords=["luxury", "suv", "vehicle", "family", "travel", "income"],
        expected_platforms=["meta", "yahoo_dsp", "tiktok", "ttd"],
        description="Multi-topic luxury SUV brief",
    ),
    TestCase(
        query=(
            "Our client sells organic pet food. We need to reach pet owners who "
            "care about nutrition and healthy eating. They're typically health-conscious "
            "millennials who also buy organic groceries for themselves."
        ),
        expected_keywords=["pet", "food", "organic", "health", "grocery"],
        expected_platforms=["meta", "tiktok", "ttd", "yahoo_dsp"],
        description="Multi-topic organic pet food brief",
    ),
    TestCase(
        query=(
            "We're promoting a new fitness app for home workouts. Our target audience "
            "includes gym members, yoga practitioners, and people interested in nutrition. "
            "They tend to be tech-savvy and use health tracking devices."
        ),
        expected_keywords=["fitness", "workout", "gym", "yoga", "health", "app"],
        expected_platforms=["meta", "tiktok", "ttd", "yahoo_dsp"],
        description="Multi-topic fitness app brief",
    ),
    TestCase(
        query=(
            "The campaign targets parents shopping for back-to-school supplies. "
            "We want to reach families with school-age children who are interested "
            "in education, children's clothing, and electronics for students."
        ),
        expected_keywords=["parent", "school", "education", "children", "family", "electronics"],
        expected_platforms=["meta", "tiktok", "snapchat", "ttd"],
        description="Multi-topic back-to-school brief",
    ),
    TestCase(
        query=(
            "Our client is a real estate developer marketing luxury condos in Miami. "
            "The target audience includes high-net-worth individuals interested in "
            "investment properties, luxury lifestyle, and travel to Florida."
        ),
        expected_keywords=["real estate", "luxury", "property", "investment", "travel"],
        expected_platforms=["meta", "yahoo_dsp", "ttd"],
        description="Multi-topic luxury real estate brief",
    ),
    TestCase(
        query=(
            "We need to reach sports fans who follow basketball and football. "
            "They should be interested in sports betting, fantasy sports, and "
            "streaming services for live games."
        ),
        expected_keywords=["sports", "basketball", "football", "betting", "streaming", "fantasy"],
        expected_platforms=["meta", "tiktok", "snapchat", "ttd", "yahoo_dsp"],
        description="Multi-topic sports brief",
    ),
    TestCase(
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


# ── Evaluation logic ──────────────────────────────────────────────────────


def evaluate_result(result: SearchResult, test_case: TestCase) -> dict:
    """Evaluate a single search result against expected outcomes."""
    all_names = set()
    for mg in result.matched_groups:
        all_names.add(mg.group.name.lower())
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

    top_score = result.matched_groups[0].score if result.matched_groups else 0
    total_platforms = len(matched_platforms)

    return {
        "query": test_case.query[:50] + ("..." if len(test_case.query) > 50 else ""),
        "full_query": test_case.query,
        "description": test_case.description,
        "keyword_recall": keyword_recall,
        "keyword_hits": f"{keyword_hits}/{len(test_case.expected_keywords)}",
        "platform_coverage": platform_coverage,
        "platform_hits": f"{platform_hits}/{len(test_case.expected_platforms)}",
        "total_platforms": total_platforms,
        "top_group": result.matched_groups[0].group.name if result.matched_groups else "N/A",
        "top_score": top_score,
        "num_groups": len(result.matched_groups),
        "has_recommendations": len(result.recommendations) > 0,
    }


def run_evaluation(
    engine: AudienceSearchEngine | AudienceSearchEngineV2,
    test_cases: list[TestCase] | None = None,
    label: str = "",
) -> list[dict]:
    """Run test cases and return evaluation metrics."""
    if test_cases is None:
        test_cases = TEST_CASES

    tag = f" [{label}]" if label else ""
    results = []

    print(f"\n{'='*80}")
    print(f"EVALUATION{tag}: Running {len(test_cases)} test cases")
    print(f"{'='*80}")

    for tc in test_cases:
        result = engine.search(tc.query)
        metrics = evaluate_result(result, tc)
        results.append(metrics)

    # Summary
    avg_keyword_recall = sum(r["keyword_recall"] for r in results) / len(results)
    avg_platform_coverage = sum(r["platform_coverage"] for r in results) / len(results)
    avg_top_score = sum(r["top_score"] for r in results) / len(results)

    print(f"\n{'='*80}")
    print(f"RESULTS{tag}")
    print(f"{'='*80}")
    print(f"{'Query':<52} {'Keywords':>10} {'Platforms':>10} {'Top Score':>10}")
    print("-" * 82)

    for r in results:
        print(
            f"{r['query']:<52} {r['keyword_hits']:>10} "
            f"{r['platform_hits']:>10} {r['top_score']:>10.3f}"
        )

    print("-" * 82)
    print(f"\n--- Aggregate Metrics{tag} ---")
    print(f"Average keyword recall:    {avg_keyword_recall:.1%}")
    print(f"Average platform coverage: {avg_platform_coverage:.1%}")
    print(f"Average top-1 score:       {avg_top_score:.3f}")
    print(f"Queries with recommendations: {sum(1 for r in results if r['has_recommendations'])}/{len(results)}")

    return results


def compare_v1_v2(
    engine_v1: AudienceSearchEngine,
    engine_v2: AudienceSearchEngineV2,
    test_cases: list[TestCase] | None = None,
) -> dict:
    """Run same queries on both engines and compare metrics."""
    if test_cases is None:
        test_cases = TEST_CASES + LONG_BRIEF_TEST_CASES

    print(f"\n{'='*80}")
    print("V1 vs V2 COMPARISON")
    print(f"{'='*80}")

    v1_results = []
    v2_results = []

    for tc in test_cases:
        r1 = engine_v1.search(tc.query)
        r2 = engine_v2.search(tc.query)
        v1_results.append(evaluate_result(r1, tc))
        v2_results.append(evaluate_result(r2, tc))

    # Print comparison
    print(f"\n{'Query':<52} {'v1 KW':>8} {'v2 KW':>8} {'v1 Plat':>8} {'v2 Plat':>8}")
    print("-" * 84)

    for r1, r2 in zip(v1_results, v2_results):
        print(
            f"{r1['query']:<52} "
            f"{r1['keyword_hits']:>8} {r2['keyword_hits']:>8} "
            f"{r1['platform_hits']:>8} {r2['platform_hits']:>8}"
        )

    # Aggregate comparison
    v1_avg_kw = sum(r["keyword_recall"] for r in v1_results) / len(v1_results)
    v2_avg_kw = sum(r["keyword_recall"] for r in v2_results) / len(v2_results)
    v1_avg_plat = sum(r["platform_coverage"] for r in v1_results) / len(v1_results)
    v2_avg_plat = sum(r["platform_coverage"] for r in v2_results) / len(v2_results)

    print(f"\n--- Aggregate ---")
    print(f"Keyword recall:    v1={v1_avg_kw:.1%}  v2={v2_avg_kw:.1%}  delta={v2_avg_kw-v1_avg_kw:+.1%}")
    print(f"Platform coverage: v1={v1_avg_plat:.1%}  v2={v2_avg_plat:.1%}  delta={v2_avg_plat-v1_avg_plat:+.1%}")

    # Separate short vs long brief results
    n_short = len(TEST_CASES)
    short_v1 = v1_results[:n_short]
    short_v2 = v2_results[:n_short]
    long_v1 = v1_results[n_short:]
    long_v2 = v2_results[n_short:]

    if short_v1:
        s_v1 = sum(r["keyword_recall"] for r in short_v1) / len(short_v1)
        s_v2 = sum(r["keyword_recall"] for r in short_v2) / len(short_v2)
        print(f"\nShort queries:     v1={s_v1:.1%}  v2={s_v2:.1%}  delta={s_v2-s_v1:+.1%}")

    if long_v1:
        l_v1 = sum(r["keyword_recall"] for r in long_v1) / len(long_v1)
        l_v2 = sum(r["keyword_recall"] for r in long_v2) / len(long_v2)
        print(f"Long briefs:       v1={l_v1:.1%}  v2={l_v2:.1%}  delta={l_v2-l_v1:+.1%}")

    return {
        "v1_results": v1_results,
        "v2_results": v2_results,
        "v1_avg_keyword_recall": v1_avg_kw,
        "v2_avg_keyword_recall": v2_avg_kw,
        "v1_avg_platform_coverage": v1_avg_plat,
        "v2_avg_platform_coverage": v2_avg_plat,
    }


if __name__ == "__main__":
    from query import create_engine

    engine = create_engine("v2")

    # Short keyword tests
    print("\n" + "="*80)
    print("SHORT KEYWORD TESTS")
    print("="*80)
    run_evaluation(engine, TEST_CASES, label="v2 short")

    # Long brief tests
    print("\n" + "="*80)
    print("LONG BRIEF TESTS")
    print("="*80)
    run_evaluation(engine, LONG_BRIEF_TEST_CASES, label="v2 long")
