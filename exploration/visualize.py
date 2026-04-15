"""Visualization tools: interactive graph, UMAP scatter, comparison tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network

import config
from clustering import AudienceGroup
from data_loader import Segment
from query import SearchResult


# ── Platform colors ────────────────────────────────────────────────────────

PLATFORM_COLORS = {
    "meta": "#1877F2",
    "tiktok": "#000000",
    "snapchat": "#FFFC00",
    "yahoo_dsp": "#720E9E",
    "ttd": "#00C853",
    "dv360": "#4285F4",
    "group": "#FF6B6B",
}

EDGE_COLORS = {
    "MEMBER_OF": "#CCCCCC",
    "IS_CHILD_OF": "#888888",
    "EQUIVALENT_TO": "#FF6B6B",
    "RELATED_TO": "#4ECDC4",
    "BROADER_THAN": "#45B7D1",
    "NARROWER_THAN": "#96CEB4",
}


# ── Interactive Graph (pyvis) ──────────────────────────────────────────────


def create_graph_viz(
    search_result: SearchResult,
    segments: list[Segment],
    groups: list[AudienceGroup],
    graph,
    output_path: Path | str = "graph_viz.html",
    max_nodes: int = 150,
) -> str:
    """Create an interactive graph visualization for a search result (v1 compat)."""
    return create_graph_viz_v2(
        search_result=search_result,
        segments=segments,
        graph=graph,
        output_path=output_path,
    )


def create_graph_viz_v2(
    search_result: SearchResult,
    segments: list[Segment],
    graph,
    super_categories=None,
    sub_categories=None,
    output_path: Path | str = "graph_viz.html",
    max_segs_per_platform: int = 3,
) -> str:
    """Create an interactive 3-layer graph visualization for a v2 search result.

    Shows: Query -> Super-Categories -> Sub-Categories -> Platform Segments
    with cross-platform EQUIVALENT_TO edges and RELATED_TO recommendation edges.
    """
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#0f0f23",
        font_color="white",
        directed=True,
    )
    net.barnes_hut(gravity=-4000, central_gravity=0.3, spring_length=200)

    id_to_seg = {s.id: s for s in segments}
    added_nodes = set()
    segment_node_ids = set()  # track segment nodes for cross-platform edges

    # ── Query node (center) ───────────────────────────────────────────
    query_label = search_result.query[:50] + ("..." if len(search_result.query) > 50 else "")
    net.add_node(
        "QUERY",
        label=f"Query: {query_label}",
        color="#FFD700",
        size=45,
        shape="star",
        title=f"Query: {search_result.query}",
        font={"size": 16, "color": "#FFD700", "bold": True},
        level=0,
    )
    added_nodes.add("QUERY")

    # ── Layer 0: Super-categories (from sentence topics) ──────────────
    super_cat_names = set()
    if search_result.sentence_topics:
        for sentence, topics in search_result.sentence_topics.items():
            for t in topics:
                super_cat_names.add(t)

    for sc_name in super_cat_names:
        sc_node = f"L0_{sc_name}"
        if sc_node in added_nodes:
            continue
        net.add_node(
            sc_node,
            label=sc_name,
            color="#FF6B6B",
            size=35,
            shape="diamond",
            title=f"Super-Category (Layer 0)\n{sc_name}",
            font={"size": 14, "color": "white", "bold": True},
            level=1,
        )
        added_nodes.add(sc_node)
        # Edge from query to super-category
        net.add_edge("QUERY", sc_node, color="#FFD700", width=2, title="matched topic")

    # ── Layer 1: Sub-categories (matched groups) ──────────────────────
    matched_sub_nodes = []
    for i, mg in enumerate(search_result.matched_groups[:12]):
        sub_node = f"L1_{mg.group.id}"
        if sub_node in added_nodes:
            continue

        platforms_str = ", ".join(sorted(mg.group.platforms))
        net.add_node(
            sub_node,
            label=mg.group.name[:35],
            color="#FF8C42",
            size=25 + min(mg.group.member_count // 5, 15),
            shape="diamond",
            title=(
                f"Sub-Category (Layer 1)\n"
                f"{mg.group.name}\n"
                f"Score: {mg.score:.3f}\n"
                f"Members: {mg.group.member_count}\n"
                f"Platforms: {platforms_str}"
            ),
            font={"size": 11, "color": "white"},
            level=2,
        )
        added_nodes.add(sub_node)
        matched_sub_nodes.append((sub_node, mg))

        # Connect sub-category to its super-category if we can match it
        connected_to_l0 = False
        for sc_name in super_cat_names:
            # Check if sub-category name starts with or relates to super-category
            if (sc_name.lower() in mg.group.name.lower()
                    or mg.group.name.lower().startswith(sc_name.split(" > ")[0].lower())):
                net.add_edge(
                    f"L0_{sc_name}", sub_node,
                    color="#FF6B6B", width=2, title="CONTAINS",
                )
                connected_to_l0 = True
                break

        # Fallback: connect to query if no L0 match
        if not connected_to_l0:
            net.add_edge("QUERY", sub_node, color="#555555", width=1, dashes=True)

    # ── Layer 2: Platform segments ────────────────────────────────────
    # Build a map: segment_id -> which sub-categories it belongs to
    seg_to_subs = {}
    for sub_node, mg in matched_sub_nodes:
        for seg_id in mg.group.segment_ids:
            if seg_id not in seg_to_subs:
                seg_to_subs[seg_id] = []
            seg_to_subs[seg_id].append(sub_node)

    for platform in sorted(search_result.segments_by_platform):
        segs = search_result.segments_by_platform[platform]
        for seg, score in segs[:max_segs_per_platform]:
            seg_node = seg.id
            if seg_node in added_nodes:
                continue

            size_label = ""
            if seg.audience_size and seg.audience_size >= 1_000_000:
                size_label = f"\nReach: {seg.audience_size/1_000_000:.1f}M"

            color = PLATFORM_COLORS.get(seg.platform, "#888888")
            # Make snapchat visible on dark background
            if seg.platform == "snapchat":
                color = "#E8E800"

            net.add_node(
                seg_node,
                label=f"[{seg.platform.upper()[:4]}] {seg.name[:25]}",
                color=color,
                size=14,
                shape="dot",
                title=(
                    f"{seg.name}\n"
                    f"Platform: {seg.platform}\n"
                    f"Type: {seg.segment_type}\n"
                    f"Score: {score:.3f}{size_label}\n"
                    f"Hierarchy: {' > '.join(seg.hierarchy)}"
                ),
                font={"size": 9, "color": "#CCCCCC"},
                level=3,
            )
            added_nodes.add(seg_node)
            segment_node_ids.add(seg_node)

            # Connect segment to its sub-category
            parent_subs = seg_to_subs.get(seg.id, [])
            if parent_subs:
                net.add_edge(
                    parent_subs[0], seg_node,
                    color="#555555", width=1, title="MEMBER_OF",
                )
            else:
                # Fallback: connect to closest matched sub by score
                if matched_sub_nodes:
                    net.add_edge(
                        matched_sub_nodes[0][0], seg_node,
                        color="#333333", width=1, dashes=True,
                    )

    # ── Cross-platform EQUIVALENT_TO edges ────────────────────────────
    equiv_count = 0
    for u, v, data in graph.edges(data=True):
        if data.get("edge_type") != "EQUIVALENT_TO":
            continue
        if u in segment_node_ids and v in segment_node_ids:
            u_plat = id_to_seg.get(u)
            v_plat = id_to_seg.get(v)
            if u_plat and v_plat and u_plat.platform != v_plat.platform:
                net.add_edge(
                    u, v,
                    color="#FF6B6B",
                    width=2,
                    title=f"EQUIVALENT_TO (sim: {data.get('weight', 0):.2f})",
                    dashes=False,
                )
                equiv_count += 1
                if equiv_count > 30:
                    break

    # ── Recommendations ───────────────────────────────────────────────
    for rec in search_result.recommendations[:4]:
        rec_node = f"REC_{rec.name}"
        if rec_node in added_nodes:
            continue
        net.add_node(
            rec_node,
            label=f">> {rec.name[:30]}",
            color="#4ECDC4",
            size=20,
            shape="triangle",
            title=(
                f"Recommendation\n"
                f"{rec.name}\n"
                f"Relation: {rec.relation}\n"
                f"Similarity: {rec.score:.2f}\n"
                f"Members: {rec.group.member_count}"
            ),
            font={"size": 10, "color": "#4ECDC4"},
            level=2,
        )
        added_nodes.add(rec_node)

        # Connect to closest matched sub-category
        if matched_sub_nodes:
            net.add_edge(
                matched_sub_nodes[0][0], rec_node,
                color="#4ECDC4", width=1, dashes=True, title="RELATED_TO",
            )

    # ── SIBLING_OF edges between sub-categories ──────────────────────
    sub_node_ids = [n for n in added_nodes if n.startswith("L1_")]
    for u in sub_node_ids:
        for v in sub_node_ids:
            if u >= v:
                continue
            u_id = u.replace("L1_", "sub_")
            v_id = v.replace("L1_", "sub_")
            if graph.has_edge(u_id, v_id):
                edge_data = graph.edges[u_id, v_id]
                if edge_data.get("edge_type") == "SIBLING_OF":
                    net.add_edge(u, v, color="#96CEB4", width=1, dashes=True, title="SIBLING_OF")

    output_path = str(output_path)
    net.save_graph(output_path)
    print(f"Graph visualization saved to {output_path} ({len(added_nodes)} nodes)")
    return output_path


# ── UMAP Embedding Scatter (plotly) ────────────────────────────────────────


def create_umap_scatter(
    embeddings: np.ndarray,
    segments: list[Segment],
    labels: np.ndarray,
    groups: list[AudienceGroup],
    output_path: Path | str = "umap_scatter.html",
    sample_size: int = 3000,
) -> str:
    """Create a 2D UMAP scatter plot of embeddings, colored by cluster."""
    import umap

    # Sample for performance if needed
    n = len(embeddings)
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        emb_sample = embeddings[indices]
        segs_sample = [segments[i] for i in indices]
        labels_sample = labels[indices]
    else:
        emb_sample = embeddings
        segs_sample = segments
        labels_sample = labels

    print(f"Running UMAP on {len(emb_sample)} embeddings...")
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    coords = reducer.fit_transform(emb_sample.astype(np.float64))

    # Build hover text
    hover_texts = []
    platforms = []
    cluster_names = []
    group_map = {g.id: g.name for g in groups}

    for i, seg in enumerate(segs_sample):
        label = int(labels_sample[i])
        cluster_name = group_map.get(label, f"Cluster {label}")
        hover_texts.append(
            f"Name: {seg.name}<br>"
            f"Platform: {seg.platform}<br>"
            f"Type: {seg.segment_type}<br>"
            f"Cluster: {cluster_name}<br>"
            f"Hierarchy: {' > '.join(seg.hierarchy[:3])}"
        )
        platforms.append(seg.platform)
        cluster_names.append(cluster_name if label >= 0 else "Noise")

    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=platforms,
        color_discrete_map=PLATFORM_COLORS,
        hover_name=[s.name for s in segs_sample],
        custom_data=[hover_texts],
        title="Audience Segments in Embedding Space (UMAP 2D)",
        labels={"x": "UMAP-1", "y": "UMAP-2", "color": "Platform"},
    )

    fig.update_traces(
        marker=dict(size=5, opacity=0.7),
        hovertemplate="%{customdata[0]}<extra></extra>",
    )

    fig.update_layout(
        template="plotly_dark",
        width=1200,
        height=800,
        legend=dict(font=dict(size=14)),
    )

    output_path = str(output_path)
    fig.write_html(output_path)
    print(f"UMAP scatter saved to {output_path}")
    return output_path


# ── Cluster by platform heatmap ────────────────────────────────────────────


def create_cluster_platform_heatmap(
    groups: list[AudienceGroup],
    output_path: Path | str = "cluster_heatmap.html",
    top_n: int = 30,
) -> str:
    """Create a heatmap showing cluster x platform distribution."""
    # Sort groups by size
    sorted_groups = sorted(groups, key=lambda g: g.member_count, reverse=True)[:top_n]

    platforms = sorted(config.PLATFORMS)
    matrix = []
    group_names = []

    for g in sorted_groups:
        row = []
        for p in platforms:
            count = sum(1 for sid in g.segment_ids if sid.startswith(f"iab_{p}") or sid.startswith(f"social_{p}") or sid.startswith(f"{p}_") or sid.startswith(f"meta_") and p == "meta" or sid.startswith(f"yahoo_") and p == "yahoo_dsp" or sid.startswith(f"ttd_") and p == "ttd")
            row.append(count)
        matrix.append(row)
        group_names.append(f"{g.name} ({g.member_count})")

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[p.upper() for p in platforms],
        y=group_names,
        colorscale="YlOrRd",
        hoverongaps=False,
        hovertemplate="Group: %{y}<br>Platform: %{x}<br>Segments: %{z}<extra></extra>",
    ))

    fig.update_layout(
        title="Top Audience Groups: Segment Count by Platform",
        template="plotly_dark",
        width=1000,
        height=800,
        yaxis=dict(autorange="reversed"),
    )

    output_path = str(output_path)
    fig.write_html(output_path)
    print(f"Cluster heatmap saved to {output_path}")
    return output_path


# ── Search result visualization ────────────────────────────────────────────


def create_result_sunburst(
    result: SearchResult,
    output_path: Path | str = "result_sunburst.html",
) -> str:
    """Create a sunburst chart of search results: Query > Platform > Segments."""
    labels = [result.query]
    parents = [""]
    values = [0]
    colors = ["#333333"]

    for platform in sorted(result.segments_by_platform):
        segs = result.segments_by_platform[platform]
        platform_label = platform.upper()
        labels.append(platform_label)
        parents.append(result.query)
        values.append(len(segs))
        colors.append(PLATFORM_COLORS.get(platform, "#888888"))

        for seg, score in segs[:5]:
            seg_label = f"{seg.name} ({score:.2f})"
            labels.append(seg_label)
            parents.append(platform_label)
            values.append(1)
            colors.append(PLATFORM_COLORS.get(platform, "#888888"))

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colors=colors),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Search Results: \"{result.query}\"",
        template="plotly_dark",
        width=800,
        height=800,
    )

    output_path = str(output_path)
    fig.write_html(output_path)
    print(f"Result sunburst saved to {output_path}")
    return output_path


if __name__ == "__main__":
    from data_loader import load_all
    from embedder import load_artifacts, load_model
    from clustering import load_clusters
    from graph_builder import load_graph
    from query import create_engine

    # Load everything
    segments = load_all()
    embeddings, index, segment_ids = load_artifacts(prefix="raw")
    groups, labels, centroids = load_clusters(prefix="raw")
    graph = load_graph()

    # Create engine and run a query
    engine = create_engine()
    result = engine.search("luxury SUV shoppers")

    # Generate all visualizations
    create_graph_viz(result, segments, groups, graph, output_path="viz_graph.html")
    create_umap_scatter(embeddings, segments, labels, groups, output_path="viz_umap.html")
    create_cluster_platform_heatmap(groups, output_path="viz_heatmap.html")
    create_result_sunburst(result, output_path="viz_sunburst.html")
