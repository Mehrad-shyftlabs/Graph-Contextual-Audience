"""Streamlit demo for the Audience Targeting Engine (v2 — Qdrant-backed)."""

from __future__ import annotations

import html as html_lib
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from audience_targeting.search_engine import AudienceSearchEngine, create_engine
from audience_targeting.settings import Settings

# ── Constants ────────────────────────────────────────────────────────────

PLATFORM_COLORS = {
    "meta": "#1877F2",
    "tiktok": "#fe2c55",
    "snapchat": "#FFFC00",
    "yahoo_dsp": "#720E9E",
    "ttd": "#00C853",
    "dv360": "#4285F4",
}

PLATFORM_LABELS = {
    "meta": "Meta",
    "tiktok": "TikTok",
    "snapchat": "Snapchat",
    "yahoo_dsp": "Yahoo DSP",
    "ttd": "The Trade Desk",
    "dv360": "DV360",
}

PLATFORM_ICONS = {
    "meta": "M",
    "tiktok": "T",
    "snapchat": "S",
    "yahoo_dsp": "Y",
    "ttd": "TTD",
    "dv360": "DV",
}


# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(page_title="Audience Targeting Engine", page_icon="\U0001F3AF", layout="wide")


# ── Light theme CSS ──────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Score bar container */
  .score-row { display: flex; align-items: center; gap: 10px; padding: 10px 0; border-bottom: 1px solid #eee; }
  .score-row:last-child { border-bottom: none; }
  .score-rank { width: 28px; font-size: 12px; font-weight: 800; color: #999; text-align: center; }
  .score-info { flex: 1; min-width: 0; }
  .score-name { font-size: 13.5px; font-weight: 600; color: #1a1a2e; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .score-hier { font-size: 11px; color: #666; margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .score-desc { font-size: 11px; color: #999; margin-top: 3px; display: -webkit-box; -webkit-line-clamp: 1; -webkit-box-orient: vertical; overflow: hidden; }
  .score-bar-wrap { width: 140px; flex-shrink: 0; }
  .score-bar-track { height: 6px; background: #e8e8e8; border-radius: 100px; overflow: hidden; }
  .score-bar-fill { height: 100%; border-radius: 100px; }
  .score-val { width: 48px; text-align: right; font-size: 14px; font-weight: 800; flex-shrink: 0; }
  .score-badge { width: 60px; flex-shrink: 0; text-align: center; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 100px; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }
  .badge-match { background: rgba(22,163,74,0.1); color: #16a34a; border: 1px solid rgba(22,163,74,0.25); }
  .badge-partial { background: rgba(234,88,12,0.1); color: #ea580c; border: 1px solid rgba(234,88,12,0.25); }

  /* Platform card header */
  .plat-header {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 16px; border-radius: 10px 10px 0 0;
    margin-bottom: 0;
  }
  .plat-icon {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 800; color: #fff;
    box-shadow: 0 1px 4px rgba(0,0,0,0.15);
  }
  .plat-title { font-size: 15px; font-weight: 700; color: #1a1a2e; }
  .plat-count { font-size: 12px; color: #666; margin-left: auto; }

  /* Platform card body */
  .plat-body { background: #fff; border: 1px solid #e0e0e0; border-top: none; border-radius: 0 0 10px 10px; padding: 8px 16px; margin-bottom: 16px; }

  /* Topic card */
  .topic-card { background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 10px; padding: 14px 16px; margin-bottom: 8px; }
  .topic-sentence { font-style: italic; color: #666; font-size: 13px; margin-bottom: 8px; }
  .topic-tag { display: inline-block; background: rgba(99,102,241,0.1); color: #6366f1; border: 1px solid rgba(99,102,241,0.25); border-radius: 100px; padding: 3px 10px; font-size: 11px; font-weight: 600; margin: 2px 4px 2px 0; }

  /* Rec pill */
  .rec-pill { display: inline-flex; align-items: center; gap: 6px; background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 8px 14px; margin: 4px; font-size: 12.5px; color: #1a1a2e; }
  .rec-members { color: #999; font-size: 11px; }

  /* Formula bar */
  .formula-bar { background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 10px; padding: 12px 16px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 12px; margin-bottom: 16px; }
  .fml-label { color: #999; font-weight: 700; }
  .fml-part { display: inline-flex; align-items: center; gap: 5px; background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 4px 10px; font-weight: 600; }
  .fml-dot { width: 8px; height: 8px; border-radius: 50%; }
  .fml-op { color: #999; }

  /* Sub-cat list item */
  .subcat-item { background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
  .subcat-name { font-size: 14px; font-weight: 700; color: #1a1a2e; }
  .subcat-meta { font-size: 12px; color: #666; margin-top: 4px; }
  .subcat-sentence { font-size: 11px; color: #999; font-style: italic; margin-top: 4px; }
  .subcat-score { color: #6366f1; font-weight: 800; }

  /* Section headers */
  .section-head { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #999; margin: 16px 0 10px 0; }
</style>
""", unsafe_allow_html=True)


# ── Cache ────────────────────────────────────────────────────────────────

@st.cache_resource
def get_engine() -> AudienceSearchEngine:
    return create_engine(Settings())


@st.cache_resource
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


# ── Sidebar ──────────────────────────────────────────────────────────────

st.sidebar.markdown("## \U0001F3AF Audience Targeting")
st.sidebar.caption("Cross-platform segment matching via BGE embeddings + Qdrant")

st.sidebar.markdown("---")
st.sidebar.markdown("### Platforms")
selected_platforms = []
for p in settings.platforms:
    if st.sidebar.checkbox(PLATFORM_LABELS.get(p, p), value=True, key=f"plat_{p}"):
        selected_platforms.append(p)

st.sidebar.markdown("---")
top_k = st.sidebar.slider("Results per platform", 3, 20, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("### Match Quality")
st.sidebar.markdown(
    f'<div style="font-size:12px;color:#666;">'
    f'<span style="color:#16a34a;font-weight:700;">Match</span> &ge; {settings.default_match_threshold}<br/>'
    f'<span style="color:#ea580c;font-weight:700;">Partial</span> &ge; {settings.default_partial_match_threshold}<br/>'
    f'<span style="color:#999;">Below {settings.default_partial_match_threshold}</span> &rarr; filtered out'
    f'</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Scoring Formula")
st.sidebar.markdown(
    '<div style="font-size:11px;color:#666;line-height:1.7;">'
    '<b style="color:#1a1a2e;">score</b> = BGE cosine similarity<br/>'
    '+ cohesion boost (0.05 &times; cluster density)<br/>'
    '<i style="color:#999;">+ Node2Vec 30% (when enabled)</i>'
    '</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.caption("9,095 segments | 6 platforms | 411 sub-categories")


# ── Main ─────────────────────────────────────────────────────────────────

st.markdown('<h1 style="color:#1a1a2e;margin-bottom:4px;">AI-Powered Audience Targeting</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#666;font-size:14px;margin-bottom:20px;">'
    'Enter a client brief or keyword query. Long paragraphs are automatically split into sentences for multi-topic search.'
    '</p>',
    unsafe_allow_html=True,
)

query = st.text_area(
    "Client Brief / Audience Description",
    placeholder="e.g., We're launching a premium SUV campaign targeting affluent families...",
    height=90,
    key="query_input",
    label_visibility="collapsed",
)

# Examples
st.markdown('<div class="section-head">Quick examples</div>', unsafe_allow_html=True)

examples_short = [
    "luxury SUV shoppers",
    "pet food buyers",
    "fitness app users",
    "basketball fans",
    "gaming enthusiasts",
]
examples_long = [
    (
        "We're launching a premium SUV campaign targeting affluent families. "
        "The ideal audience owns luxury vehicles, has high household income, "
        "and shows interest in family activities and travel."
    ),
    (
        "Our client sells organic pet food. We need to reach pet owners who "
        "care about nutrition and healthy eating. They're typically health-conscious "
        "millennials who also buy organic groceries for themselves."
    ),
    (
        "We need to reach sports fans who follow basketball and football. "
        "They should be interested in sports betting, fantasy sports, and "
        "streaming services for live games."
    ),
]

col_short, col_long = st.columns(2)
with col_short:
    st.markdown('<span style="color:#666;font-size:12px;font-weight:600;">SHORT QUERIES</span>', unsafe_allow_html=True)
    for i, ex in enumerate(examples_short):
        if st.button(ex, key=f"ex_s_{i}", use_container_width=True):
            query = ex
with col_long:
    st.markdown('<span style="color:#666;font-size:12px;font-weight:600;">LONG BRIEFS</span>', unsafe_allow_html=True)
    for i, ex in enumerate(examples_long):
        if st.button(ex[:65] + "...", key=f"ex_l_{i}", use_container_width=True):
            query = ex


# ── Helpers ──────────────────────────────────────────────────────────────

def esc(s: str) -> str:
    return html_lib.escape(str(s))


def score_color(score: float) -> str:
    if score >= settings.default_match_threshold:
        return "#4ade80"
    if score >= settings.default_partial_match_threshold:
        return "#fb923c"
    return "#52616e"


def render_score_bar(score: float) -> str:
    pct = max(0, min(100, score * 100))
    color = score_color(score)
    return (
        f'<div class="score-bar-track">'
        f'<div class="score-bar-fill" style="width:{pct:.0f}%;background:{color};"></div>'
        f'</div>'
    )


def render_badge(label: str) -> str:
    cls = "badge-match" if label == "match" else "badge-partial"
    text = "Match" if label == "match" else "Partial"
    return f'<span class="badge {cls}">{text}</span>'


def format_audience(size: int | None) -> str:
    if not size:
        return ""
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.1f}B"
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f}M"
    if size >= 1_000:
        return f"{size / 1_000:.0f}K"
    return f"{size:,}"


def render_platform_card(platform: str, segments: list) -> str:
    color = PLATFORM_COLORS.get(platform, "#888")
    label = PLATFORM_LABELS.get(platform, platform)
    icon = PLATFORM_ICONS.get(platform, "?")

    rows_html = ""
    for rank, (seg, score, match_label) in enumerate(segments, 1):
        desc = esc((seg.description or "")[:100])
        hier = esc(" > ".join(seg.hierarchy[:3]))
        size = format_audience(seg.audience_size)
        size_html = f'<span style="color:#999;font-size:11px;margin-left:8px;">{size}</span>' if size else ""

        rows_html += f'''
        <div class="score-row">
          <div class="score-rank">{rank}</div>
          <div class="score-info">
            <div class="score-name">{esc(seg.name)}{size_html}</div>
            <div class="score-hier">{hier}</div>
            {"<div class='score-desc'>" + desc + "</div>" if desc else ""}
          </div>
          <div class="score-bar-wrap">{render_score_bar(score)}</div>
          <div class="score-val" style="color:{score_color(score)};">{score:.0%}</div>
          <div class="score-badge">{render_badge(match_label)}</div>
        </div>'''

    return f'''
    <div class="plat-header" style="background:{color}22;border:1px solid {color}44;border-bottom:none;border-radius:10px 10px 0 0;">
      <div class="plat-icon" style="background:{color};">{icon}</div>
      <div class="plat-title">{label}</div>
      <div class="plat-count">{len(segments)} segment{"s" if len(segments) != 1 else ""}</div>
    </div>
    <div class="plat-body">{rows_html}</div>
    '''


# ── Search ───────────────────────────────────────────────────────────────

if query:
    engine = get_engine()
    platforms_filter = selected_platforms if selected_platforms else None

    t0 = time.time()
    result = engine.search(query, platforms=platforms_filter, top_k_segments=top_k)
    elapsed_ms = (time.time() - t0) * 1000

    # Apply match threshold filtering
    filtered_by_platform: dict[str, list] = {}
    for platform, seg_scores in result.segments_by_platform.items():
        filtered = []
        for seg, score in seg_scores:
            label = settings.classify_match(score, platform)
            if label is not None:
                filtered.append((seg, score, label))
        if filtered:
            filtered_by_platform[platform] = filtered

    total_segs = sum(len(s) for s in filtered_by_platform.values())
    n_subs = len(result.matched_subcategories)
    top_score = result.matched_subcategories[0].score if result.matched_subcategories else 0

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Sub-Categories", n_subs)
    col2.metric("Segments", total_segs)
    col3.metric("Platforms", len(filtered_by_platform))
    col4.metric("Top Score", f"{top_score:.1%}")
    col5.metric("Latency", f"{elapsed_ms:.0f}ms")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Platform Results",
        "Matched Sub-Categories",
        "Detected Topics",
        "Recommendations",
    ])

    # ── Tab 1: Platform Results ──────────────────────────────────
    with tab1:
        if not filtered_by_platform:
            st.warning("No segments matched above the quality threshold. Try a broader query.")
        else:
            platform_order = ["meta", "tiktok", "snapchat", "yahoo_dsp", "ttd", "dv360"]
            for platform in platform_order:
                if platform in filtered_by_platform:
                    st.markdown(
                        render_platform_card(platform, filtered_by_platform[platform]),
                        unsafe_allow_html=True,
                    )

    # ── Tab 2: Matched Sub-Categories ────────────────────────────
    with tab2:
        for i, ms in enumerate(result.matched_subcategories[:20], 1):
            plats = " ".join(
                f'<span style="display:inline-block;padding:1px 6px;border-radius:4px;'
                f'background:{PLATFORM_COLORS.get(p, "#888")}22;color:{PLATFORM_COLORS.get(p, "#888")};'
                f'font-size:10px;font-weight:700;margin-right:3px;">{PLATFORM_LABELS.get(p, p)}</span>'
                for p in sorted(ms.sub_category.platforms)
            )
            st.markdown(
                f'<div class="subcat-item">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div class="subcat-name">{i}. {esc(ms.sub_category.name)}</div>'
                f'<div class="subcat-score">{ms.score:.1%}</div>'
                f'</div>'
                f'<div class="subcat-meta">{ms.sub_category.member_count} members &middot; {plats}</div>'
                f'{"<div class=subcat-sentence>Matched: &ldquo;" + esc(ms.source_sentence[:90]) + "&rdquo;</div>" if ms.source_sentence else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Bar chart
        if result.matched_subcategories:
            platform_counts: dict[str, int] = {}
            for ms in result.matched_subcategories:
                for p in ms.sub_category.platforms:
                    platform_counts[p] = platform_counts.get(p, 0) + 1
            sorted_plats = sorted(platform_counts.keys())
            fig = go.Figure(data=[go.Bar(
                x=[PLATFORM_LABELS.get(p, p) for p in sorted_plats],
                y=[platform_counts[p] for p in sorted_plats],
                marker_color=[PLATFORM_COLORS.get(p, "#888") for p in sorted_plats],
            )])
            fig.update_layout(
                title=dict(text="Sub-Categories per Platform", font=dict(size=14)),
                template="plotly_white",
                height=280,
                xaxis=dict(title=""),
                yaxis=dict(title=""),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Detected Topics ───────────────────────────────────
    with tab3:
        if result.sentence_topics:
            st.markdown(
                '<p style="color:#666;font-size:13px;margin-bottom:12px;">'
                'Each sentence is searched independently. Shows which super-categories matched.</p>',
                unsafe_allow_html=True,
            )
            for sentence, topics in result.sentence_topics.items():
                tags_html = " ".join(f'<span class="topic-tag">{esc(t)}</span>' for t in topics) if topics else '<span style="color:#999;font-size:12px;">No strong topic match</span>'
                st.markdown(
                    f'<div class="topic-card">'
                    f'<div class="topic-sentence">&ldquo;{esc(sentence)}&rdquo;</div>'
                    f'{tags_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Topic detection is available for multi-sentence briefs.")

    # ── Tab 4: Recommendations ───────────────────────────────────
    with tab4:
        if result.recommendations:
            st.markdown('<div class="section-head">Also consider targeting</div>', unsafe_allow_html=True)
            pills = ""
            for rec in result.recommendations:
                pills += (
                    f'<div class="rec-pill">'
                    f'<span style="font-weight:600;">{esc(rec.name)}</span>'
                    f'<span class="rec-members">{rec.member_count} segs</span>'
                    f'</div>'
                )
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#999;">No additional recommendations for this query.</p>', unsafe_allow_html=True)

        if result.broadening_options:
            st.markdown('<div class="section-head">Broaden reach</div>', unsafe_allow_html=True)
            pills = ""
            for opt in result.broadening_options:
                pills += (
                    f'<div class="rec-pill">'
                    f'<span style="font-weight:600;">{esc(opt.name)}</span>'
                    f'<span class="rec-members">{opt.member_count} segs</span>'
                    f'</div>'
                )
            st.markdown(pills, unsafe_allow_html=True)

        if result.narrowing_options:
            st.markdown('<div class="section-head">Narrow / more specific</div>', unsafe_allow_html=True)
            pills = ""
            for opt in result.narrowing_options:
                pills += (
                    f'<div class="rec-pill">'
                    f'<span style="font-weight:600;">{esc(opt.name)}</span>'
                    f'<span class="rec-members">{opt.member_count} segs</span>'
                    f'</div>'
                )
            st.markdown(pills, unsafe_allow_html=True)

else:
    # ── Empty state — system stats ───────────────────────────────
    st.markdown("---")
    engine = get_engine()
    try:
        collections = engine.client.get_collections().collections
        counts = {}
        for c in collections:
            info = engine.client.get_collection(c.name)
            counts[c.name] = info.points_count

        seg_count = counts.get(settings.collection_name("segments"), 0)
        sub_count = counts.get(settings.collection_name("subcategories"), 0)
        sup_count = counts.get(settings.collection_name("supercategories"), 0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Segments", f"{seg_count:,}")
        col2.metric("Super-Categories", sup_count)
        col3.metric("Sub-Categories", sub_count)
        col4.metric("Platforms", len(settings.platforms))
    except Exception:
        pass

    st.markdown(
        '<div style="text-align:center;padding:40px 0;">'
        '<div style="font-size:48px;opacity:0.15;margin-bottom:8px;">\U0001F50D</div>'
        '<h3 style="color:#666;font-size:16px;">Enter a query above to find audience segments</h3>'
        '<p style="color:#999;font-size:13px;max-width:500px;margin:8px auto 0;">'
        'Type a keyword or paste a full client brief. The engine will match segments across all 6 ad platforms '
        'using semantic similarity.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="formula-bar" style="margin-top:24px;">'
        '<span class="fml-label">HOW IT WORKS</span>'
        '<div class="fml-part"><span style="color:#6366f1;">1. Chunk</span></div>'
        '<span class="fml-op">&rarr;</span>'
        '<div class="fml-part"><span style="color:#2dd4bf;">2. Embed (BGE)</span></div>'
        '<span class="fml-op">&rarr;</span>'
        '<div class="fml-part"><span style="color:#ea580c;">3. Coarse Search</span></div>'
        '<span class="fml-op">&rarr;</span>'
        '<div class="fml-part"><span style="color:#16a34a;">4. Expand &amp; Re-rank</span></div>'
        '<span class="fml-op">&rarr;</span>'
        '<div class="fml-part"><span style="color:#1a1a2e;">5. Filter &amp; Label</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )
