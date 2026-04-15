"""Streamlit app for the Audience Targeting PoC (v2)."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import config
from query import AudienceSearchEngineV2, SearchResult, create_engine, format_result
from visualize import PLATFORM_COLORS


# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Audience Targeting v2",
    page_icon="🎯",
    layout="wide",
)


# ── Cache the engine (loads once) ─────────────────────────────────────────


@st.cache_resource
def get_engine() -> AudienceSearchEngineV2:
    return create_engine("v2")


# ── Sidebar ───────────────────────────────────────────────────────────────

st.sidebar.title("AI Audience Targeting v2")
st.sidebar.markdown("Cross-platform audience matching using **hierarchical graph + BGE encoder**")

st.sidebar.markdown("---")
st.sidebar.markdown("### Platform Filters")
selected_platforms = []
for p in config.PLATFORMS:
    color = PLATFORM_COLORS.get(p, "#888")
    if st.sidebar.checkbox(f"{p.upper()}", value=True, key=f"plat_{p}"):
        selected_platforms.append(p)

st.sidebar.markdown("---")
top_k = st.sidebar.slider("Max results per platform", 3, 20, 10)

# ── Main ──────────────────────────────────────────────────────────────────

st.title("AI-Powered Audience Targeting v2")
st.markdown(
    "Enter a **client brief** or audience description. "
    "Long paragraphs are automatically split into sentences for multi-topic search."
)

# Query input — text area for long briefs
query = st.text_area(
    "Client Brief / Audience Description",
    placeholder=(
        "e.g., We're launching a premium SUV campaign targeting affluent families. "
        "The ideal audience owns luxury vehicles and shows interest in travel."
    ),
    height=100,
    key="query_input",
)

# Example queries
st.markdown("**Try these examples:**")
examples_short = [
    "luxury SUV shoppers",
    "pet food buyers",
    "fitness app users",
    "basketball fans",
]
examples_long = [
    ("Premium SUV campaign targeting affluent families who own luxury vehicles, "
     "have high household income, and show interest in family activities and travel."),
    ("Organic pet food brand reaching health-conscious pet owners who buy organic "
     "groceries and care about nutrition."),
]

col_short, col_long = st.columns(2)
with col_short:
    st.markdown("**Short queries:**")
    for i, example in enumerate(examples_short):
        if st.button(example, key=f"ex_short_{i}"):
            query = example
with col_long:
    st.markdown("**Long briefs:**")
    for i, example in enumerate(examples_long):
        if st.button(example[:50] + "...", key=f"ex_long_{i}"):
            query = example

if query:
    engine = get_engine()
    platforms = selected_platforms if selected_platforms else None
    result = engine.search(query, platforms=platforms, top_k_segments=top_k)

    # ── Summary metrics ───────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matched Groups", len(result.matched_groups))
    total_segs = sum(len(s) for s in result.segments_by_platform.values())
    col2.metric("Total Segments", total_segs)
    col3.metric("Platforms", len(result.segments_by_platform))
    top_score = result.matched_groups[0].score if result.matched_groups else 0
    col4.metric("Top Match Score", f"{top_score:.3f}")

    # ── Tabs ──────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Platform Results",
        "Matched Groups",
        "Detected Topics",
        "Recommendations",
        "Raw Output",
    ])

    # Tab 1: Platform Results
    with tab1:
        for platform in sorted(result.segments_by_platform):
            segs = result.segments_by_platform[platform]
            color = PLATFORM_COLORS.get(platform, "#888")

            with st.expander(f"{platform.upper()} -- {len(segs)} segments", expanded=True):
                rows = []
                for seg, score in segs:
                    size_str = ""
                    if seg.audience_size:
                        if seg.audience_size >= 1_000_000:
                            size_str = f"{seg.audience_size/1_000_000:.1f}M"
                        else:
                            size_str = f"{seg.audience_size:,}"

                    rows.append({
                        "Name": seg.name,
                        "Score": f"{score:.3f}",
                        "Type": seg.segment_type,
                        "Hierarchy": " > ".join(seg.hierarchy[:3]),
                        "Audience Size": size_str,
                    })

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

    # Tab 2: Matched Groups (sub-categories)
    with tab2:
        for i, mg in enumerate(result.matched_groups[:15], 1):
            platforms_str = ", ".join(sorted(mg.group.platforms))
            st.markdown(
                f"**{i}. {mg.group.name}** -- "
                f"Score: `{mg.score:.3f}` | "
                f"{mg.group.member_count} members | "
                f"Platforms: {platforms_str}"
            )

        if result.matched_groups:
            platform_counts = {}
            for mg in result.matched_groups:
                for p in mg.group.platforms:
                    platform_counts[p] = platform_counts.get(p, 0) + 1

            fig = go.Figure(data=[go.Bar(
                x=list(platform_counts.keys()),
                y=list(platform_counts.values()),
                marker_color=[PLATFORM_COLORS.get(p, "#888") for p in platform_counts],
            )])
            fig.update_layout(
                title="Groups per Platform",
                template="plotly_dark",
                height=300,
                xaxis_title="Platform",
                yaxis_title="Number of Matched Groups",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Detected Topics (sentence chunking breakdown)
    with tab3:
        if result.sentence_topics:
            st.markdown("### Sentence-Level Topic Detection")
            st.markdown(
                "Each sentence in your brief is searched independently. "
                "This shows which super-categories matched each sentence."
            )
            for sentence, topics in result.sentence_topics.items():
                with st.expander(f"\"{sentence[:80]}...\"" if len(sentence) > 80 else f"\"{sentence}\""):
                    if topics:
                        for topic in topics:
                            st.markdown(f"- **{topic}**")
                    else:
                        st.info("No strong topic match for this sentence.")
        else:
            st.info("Topic detection available for multi-sentence briefs.")

    # Tab 4: Recommendations
    with tab4:
        if result.recommendations:
            st.markdown("### Also Consider")
            for rec in result.recommendations:
                st.markdown(
                    f"- **{rec.name}** ({rec.relation}, similarity: `{rec.score:.2f}`) "
                    f"-- {rec.group.member_count} segments"
                )
        else:
            st.info("No additional recommendations for this query.")

        if result.broadening_options:
            st.markdown("### Broaden Reach")
            for opt in result.broadening_options:
                st.markdown(
                    f"- **{opt.name}** (similarity: `{opt.score:.2f}`) "
                    f"-- {opt.group.member_count} segments"
                )

        if result.narrowing_options:
            st.markdown("### Narrow / More Specific")
            for opt in result.narrowing_options:
                st.markdown(
                    f"- **{opt.name}** (similarity: `{opt.score:.2f}`) "
                    f"-- {opt.group.member_count} segments"
                )

    # Tab 5: Raw output
    with tab5:
        st.code(format_result(result), language="text")

else:
    # Show stats when no query
    st.markdown("---")
    st.markdown("### System Stats")

    engine = get_engine()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Segments", f"{len(engine.segments):,}")
    col2.metric("Super-Categories", len(engine.super_categories))
    col3.metric("Sub-Categories", len(engine.sub_categories))
    col4.metric("Platforms", len(config.PLATFORMS))

    st.markdown("### How It Works (v2)")
    st.markdown("""
    1. **Chunk** -- Long briefs are split into individual sentences
    2. **Embed** -- Each sentence is encoded with BGE asymmetric encoder (query prefix)
    3. **Coarse Search** -- FAISS search over Layer 0 super-categories, then Layer 1 sub-categories
    4. **Expand** -- Layer 2 platform-specific segments retrieved via graph MEMBER_OF edges
    5. **Re-Rank** -- Combined score: text similarity (70%) + Node2Vec graph similarity (30%) + neighbor boost
    6. **Aggregate** -- Results from all sentences merged, deduplicated, grouped by platform
    """)
