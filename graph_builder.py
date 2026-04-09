"""Build a knowledge graph of audience segments and groups using NetworkX.

v1: Flat graph with group + segment nodes.
v2: 3-layer hierarchical graph (SuperCategory -> SubCategory -> Segment).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

import config
from clustering import AudienceGroup, SubCategory, SuperCategory
from data_loader import Segment


# ── v2: 3-Layer Graph ─────────────────────────────────────────────────────


def build_graph_v2(
    segments: list[Segment],
    super_categories: list[SuperCategory],
    sub_categories: list[SubCategory],
    embeddings: np.ndarray,
    l1_labels: np.ndarray,
) -> nx.DiGraph:
    """Build the 3-layer hierarchical audience knowledge graph.

    Node types:
    - 'super_category': Layer 0 — broad audience themes
    - 'sub_category':   Layer 1 — specific audience sub-categories
    - 'segment':        Layer 2 — platform-specific segments

    Edge types:
    - CONTAINS:       Layer 0 -> Layer 1
    - MEMBER_OF:      Layer 2 -> Layer 1
    - IS_CHILD_OF:    Layer 2 -> Layer 2 (within-platform hierarchy)
    - EQUIVALENT_TO:  Layer 2 <-> Layer 2 (cross-platform, same sub-cat, high sim)
    - RELATED_TO:     Layer 1 <-> Layer 1 (moderate centroid similarity)
    - BROADER_THAN:   Layer 1 -> Layer 1
    - NARROWER_THAN:  Layer 1 -> Layer 1
    - SIBLING_OF:     Layer 1 <-> Layer 1 (shared parent)
    """
    G = nx.DiGraph()

    id_to_seg = {s.id: s for s in segments}
    id_to_sub = {sub.id: sub for sub in sub_categories}

    # ── Layer 0: Super-category nodes ─────────────────────────────────
    for sc in super_categories:
        G.add_node(
            f"super_{sc.id}",
            node_type="super_category",
            layer=0,
            name=sc.name,
            member_count=sc.member_count,
            platforms=list(sc.platforms),
            n_subcategories=len(sc.subcategory_ids),
        )

    # ── Layer 1: Sub-category nodes ───────────────────────────────────
    for sub in sub_categories:
        G.add_node(
            f"sub_{sub.id}",
            node_type="sub_category",
            layer=1,
            name=sub.name,
            parent_id=sub.parent_id,
            member_count=sub.member_count,
            platforms=list(sub.platforms),
        )

    # ── Layer 2: Segment nodes ────────────────────────────────────────
    for seg in segments:
        G.add_node(
            seg.id,
            node_type="segment",
            layer=2,
            name=seg.name,
            platform=seg.platform,
            hierarchy=" > ".join(seg.hierarchy),
            segment_type=seg.segment_type,
            audience_size=seg.audience_size or 0,
        )

    # ── CONTAINS edges: Layer 0 -> Layer 1 ────────────────────────────
    print("Adding CONTAINS edges (Layer 0 -> Layer 1)...")
    contains_count = 0
    for sc in super_categories:
        for sub_id in sc.subcategory_ids:
            G.add_edge(
                f"super_{sc.id}",
                f"sub_{sub_id}",
                edge_type="CONTAINS",
                weight=1.0,
            )
            contains_count += 1
    print(f"  Added {contains_count} CONTAINS edges")

    # ── MEMBER_OF edges: Layer 2 -> Layer 1 ───────────────────────────
    print("Adding MEMBER_OF edges (Layer 2 -> Layer 1)...")
    member_count = 0
    for sub in sub_categories:
        for seg_id in sub.segment_ids:
            seg = id_to_seg.get(seg_id)
            if seg is None:
                continue
            # Weight = cosine similarity to sub-category centroid
            seg_idx = next((i for i, s in enumerate(segments) if s.id == seg_id), None)
            if seg_idx is not None and sub.centroid is not None:
                weight = float(embeddings[seg_idx].astype(np.float64) @ sub.centroid.astype(np.float64))
            else:
                weight = 0.5
            G.add_edge(
                seg_id,
                f"sub_{sub.id}",
                edge_type="MEMBER_OF",
                weight=weight,
            )
            member_count += 1
    print(f"  Added {member_count} MEMBER_OF edges")

    # ── IS_CHILD_OF edges: within-platform hierarchy ──────────────────
    print("Adding IS_CHILD_OF edges (within-platform hierarchy)...")
    child_count = _add_hierarchy_edges(G, segments)
    print(f"  Added {child_count} IS_CHILD_OF edges")

    # ── EQUIVALENT_TO edges: cross-platform, same sub-cat, high sim ───
    print("Adding EQUIVALENT_TO edges (cross-platform matches)...")
    equiv_count = _add_equivalent_edges_v2(G, segments, sub_categories, embeddings)
    print(f"  Added {equiv_count} EQUIVALENT_TO edge pairs")

    # ── RELATED_TO edges: between sub-categories ──────────────────────
    print("Adding RELATED_TO edges (between sub-categories)...")
    related_count = _add_related_edges_v2(G, sub_categories)
    print(f"  Added {related_count} RELATED_TO edge pairs")

    # ── BROADER_THAN / NARROWER_THAN edges ────────────────────────────
    print("Adding BROADER_THAN/NARROWER_THAN edges...")
    broader_count = _add_broader_edges_v2(G, sub_categories)
    print(f"  Added {broader_count} BROADER_THAN/NARROWER_THAN edge pairs")

    # ── SIBLING_OF edges: sub-categories sharing same parent ──────────
    print("Adding SIBLING_OF edges (shared parent)...")
    sibling_count = _add_sibling_edges(G, super_categories, sub_categories)
    print(f"  Added {sibling_count} SIBLING_OF edge pairs")

    _print_graph_stats(G)
    return G


# ── Edge builders ─────────────────────────────────────────────────────────


def _add_hierarchy_edges(G: nx.DiGraph, segments: list[Segment]) -> int:
    """Add IS_CHILD_OF edges based on within-platform hierarchy."""
    by_platform: dict[str, list[Segment]] = defaultdict(list)
    for s in segments:
        by_platform[s.platform].append(s)

    count = 0

    # IAB hierarchy: link Tier 2 -> Tier 1 using category_id prefix
    for platform in ["yahoo_dsp", "ttd", "dv360"]:
        platform_segs = by_platform.get(platform, [])
        cat_id_to_seg: dict[str, Segment] = {}
        for s in platform_segs:
            cat_id = s.metadata.get("category_id", "")
            if cat_id:
                cat_id_to_seg[cat_id] = s

        for cat_id, seg in cat_id_to_seg.items():
            if "-" in cat_id:
                parent_id = cat_id.rsplit("-", 1)[0]
                parent_seg = cat_id_to_seg.get(parent_id)
                if parent_seg:
                    G.add_edge(seg.id, parent_seg.id, edge_type="IS_CHILD_OF")
                    count += 1

    # Social hierarchy: Sub-Segment -> Category
    for platform in ["tiktok", "snapchat", "meta"]:
        platform_segs = by_platform.get(platform, [])
        category_segs: dict[str, Segment] = {}
        child_segs: list[Segment] = []
        for s in platform_segs:
            if len(s.hierarchy) == 1:
                category_segs[s.hierarchy[0]] = s
            elif len(s.hierarchy) >= 2:
                child_segs.append(s)
                if s.hierarchy[0] not in category_segs:
                    category_segs[s.hierarchy[0]] = s

        for s in child_segs:
            parent_cat = s.hierarchy[0]
            parent = category_segs.get(parent_cat)
            if parent and parent.id != s.id:
                G.add_edge(s.id, parent.id, edge_type="IS_CHILD_OF")
                count += 1

    return count


def _add_equivalent_edges_v2(
    G: nx.DiGraph,
    segments: list[Segment],
    sub_categories: list[SubCategory],
    embeddings: np.ndarray,
) -> int:
    """Add EQUIVALENT_TO edges between segments in the same sub-category
    from different platforms with cosine similarity > threshold.
    """
    threshold = config.SIMILARITY_THRESHOLD_EQUIVALENT
    count = 0
    id_to_idx = {s.id: i for i, s in enumerate(segments)}

    for sub in sub_categories:
        if len(sub.segment_ids) < 2:
            continue

        # Group by platform within sub-category
        by_platform: dict[str, list[str]] = defaultdict(list)
        for seg_id in sub.segment_ids:
            seg = next((s for s in segments if s.id == seg_id), None)
            if seg:
                by_platform[seg.platform].append(seg_id)

        # Only create cross-platform pairs
        platforms = list(by_platform.keys())
        for i in range(len(platforms)):
            for j in range(i + 1, len(platforms)):
                for sid_a in by_platform[platforms[i]]:
                    idx_a = id_to_idx.get(sid_a)
                    if idx_a is None:
                        continue
                    for sid_b in by_platform[platforms[j]]:
                        idx_b = id_to_idx.get(sid_b)
                        if idx_b is None:
                            continue
                        sim = float(
                            embeddings[idx_a].astype(np.float64)
                            @ embeddings[idx_b].astype(np.float64)
                        )
                        if sim >= threshold:
                            G.add_edge(sid_a, sid_b, edge_type="EQUIVALENT_TO", weight=sim)
                            G.add_edge(sid_b, sid_a, edge_type="EQUIVALENT_TO", weight=sim)
                            count += 1

    return count


def _add_related_edges_v2(
    G: nx.DiGraph,
    sub_categories: list[SubCategory],
) -> int:
    """Add RELATED_TO edges between sub-category nodes with moderate centroid similarity."""
    threshold_low = config.SIMILARITY_THRESHOLD_RELATED
    threshold_high = config.SIMILARITY_THRESHOLD_EQUIVALENT
    count = 0

    centroids = np.array(
        [sub.centroid.astype(np.float64) for sub in sub_categories],
        dtype=np.float64,
    )
    sims = centroids @ centroids.T

    for i in range(len(sub_categories)):
        for j in range(i + 1, len(sub_categories)):
            # Skip siblings (they're already connected via SIBLING_OF)
            if sub_categories[i].parent_id == sub_categories[j].parent_id:
                continue
            sim = float(sims[i, j])
            if threshold_low <= sim < threshold_high:
                G.add_edge(
                    f"sub_{sub_categories[i].id}",
                    f"sub_{sub_categories[j].id}",
                    edge_type="RELATED_TO",
                    weight=sim,
                )
                G.add_edge(
                    f"sub_{sub_categories[j].id}",
                    f"sub_{sub_categories[i].id}",
                    edge_type="RELATED_TO",
                    weight=sim,
                )
                count += 1

    return count


def _add_broader_edges_v2(
    G: nx.DiGraph,
    sub_categories: list[SubCategory],
) -> int:
    """Add BROADER_THAN / NARROWER_THAN edges between sub-categories."""
    count = 0
    centroids = {sub.id: sub.centroid.astype(np.float64) for sub in sub_categories}

    for i, sub_broad in enumerate(sub_categories):
        for j, sub_narrow in enumerate(sub_categories):
            if i == j:
                continue
            # Only if broad group is significantly larger
            if sub_broad.member_count < sub_narrow.member_count * 1.5:
                continue
            # Must share the same parent
            if sub_broad.parent_id != sub_narrow.parent_id:
                continue
            # Check centroid similarity
            if sub_broad.id in centroids and sub_narrow.id in centroids:
                sim = float(centroids[sub_broad.id] @ centroids[sub_narrow.id])
                if sim >= 0.6:
                    G.add_edge(
                        f"sub_{sub_broad.id}",
                        f"sub_{sub_narrow.id}",
                        edge_type="BROADER_THAN",
                        weight=sim,
                    )
                    G.add_edge(
                        f"sub_{sub_narrow.id}",
                        f"sub_{sub_broad.id}",
                        edge_type="NARROWER_THAN",
                        weight=sim,
                    )
                    count += 1

    return count


def _add_sibling_edges(
    G: nx.DiGraph,
    super_categories: list[SuperCategory],
    sub_categories: list[SubCategory],
) -> int:
    """Add SIBLING_OF edges between sub-categories sharing the same parent."""
    count = 0

    for sc in super_categories:
        sub_ids = sc.subcategory_ids
        for i in range(len(sub_ids)):
            for j in range(i + 1, len(sub_ids)):
                G.add_edge(
                    f"sub_{sub_ids[i]}",
                    f"sub_{sub_ids[j]}",
                    edge_type="SIBLING_OF",
                    weight=1.0,
                )
                G.add_edge(
                    f"sub_{sub_ids[j]}",
                    f"sub_{sub_ids[i]}",
                    edge_type="SIBLING_OF",
                    weight=1.0,
                )
                count += 1

    return count


# ── v1: Flat graph (kept for backward compatibility) ──────────────────────


def build_graph(
    segments: list[Segment],
    groups: list[AudienceGroup],
    labels: np.ndarray,
    embeddings: np.ndarray,
) -> nx.DiGraph:
    """Build the flat audience knowledge graph (v1)."""
    G = nx.DiGraph()

    id_to_seg = {s.id: s for s in segments}

    for g in groups:
        G.add_node(
            f"group_{g.id}",
            node_type="group",
            name=g.name,
            member_count=g.member_count,
            platforms=list(g.platforms),
            top_iab_category=g.top_iab_category,
        )

    for seg in segments:
        G.add_node(
            seg.id,
            node_type="segment",
            name=seg.name,
            platform=seg.platform,
            hierarchy=" > ".join(seg.hierarchy),
            segment_type=seg.segment_type,
            audience_size=seg.audience_size or 0,
        )

    print("Adding MEMBER_OF edges...")
    for seg_idx, label in enumerate(labels):
        if label < 0:
            continue
        seg = segments[seg_idx]
        group_node = f"group_{label}"
        if group_node in G:
            group = next((g for g in groups if g.id == label), None)
            if group is not None and group.centroid is not None:
                weight = float(embeddings[seg_idx].astype(np.float64) @ group.centroid.astype(np.float64))
            else:
                weight = 0.0
            G.add_edge(seg.id, group_node, edge_type="MEMBER_OF", weight=weight)

    print("Adding IS_CHILD_OF edges...")
    _add_hierarchy_edges(G, segments)

    print("Adding EQUIVALENT_TO edges...")
    _add_equivalent_edges_v1(G, segments, groups, labels, embeddings)

    print("Adding RELATED_TO edges...")
    _add_related_edges_v1(G, groups)

    print("Adding BROADER_THAN edges...")
    _add_broader_edges_v1(G, groups)

    _print_graph_stats(G)
    return G


def _add_equivalent_edges_v1(G, segments, groups, labels, embeddings):
    threshold = config.SIMILARITY_THRESHOLD_EQUIVALENT
    cluster_segments: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label >= 0:
            cluster_segments[label].append(idx)

    for cluster_id, member_indices in cluster_segments.items():
        if len(member_indices) < 2:
            continue
        by_platform: dict[str, list[int]] = defaultdict(list)
        for idx in member_indices:
            by_platform[segments[idx].platform].append(idx)
        platforms = list(by_platform.keys())
        for i in range(len(platforms)):
            for j in range(i + 1, len(platforms)):
                for idx_a in by_platform[platforms[i]]:
                    for idx_b in by_platform[platforms[j]]:
                        sim = float(embeddings[idx_a].astype(np.float64) @ embeddings[idx_b].astype(np.float64))
                        if sim >= threshold:
                            G.add_edge(segments[idx_a].id, segments[idx_b].id, edge_type="EQUIVALENT_TO", weight=sim)
                            G.add_edge(segments[idx_b].id, segments[idx_a].id, edge_type="EQUIVALENT_TO", weight=sim)


def _add_related_edges_v1(G, groups):
    threshold_low = config.SIMILARITY_THRESHOLD_RELATED
    threshold_high = config.SIMILARITY_THRESHOLD_EQUIVALENT
    centroids = np.array([g.centroid.astype(np.float64) for g in groups], dtype=np.float64)
    sims = centroids @ centroids.T
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            sim = float(sims[i, j])
            if threshold_low <= sim < threshold_high:
                G.add_edge(f"group_{groups[i].id}", f"group_{groups[j].id}", edge_type="RELATED_TO", weight=sim)
                G.add_edge(f"group_{groups[j].id}", f"group_{groups[i].id}", edge_type="RELATED_TO", weight=sim)


def _add_broader_edges_v1(G, groups):
    iab_groups = [g for g in groups if g.top_iab_category]
    centroids = {g.id: g.centroid.astype(np.float64) for g in groups if g.centroid is not None}
    for i, g_broad in enumerate(iab_groups):
        for j, g_narrow in enumerate(iab_groups):
            if i == j:
                continue
            if g_broad.member_count < g_narrow.member_count * 1.5:
                continue
            if g_broad.top_iab_category != g_narrow.top_iab_category:
                continue
            if g_broad.id in centroids and g_narrow.id in centroids:
                sim = float(centroids[g_broad.id] @ centroids[g_narrow.id])
                if sim >= 0.6:
                    G.add_edge(f"group_{g_broad.id}", f"group_{g_narrow.id}", edge_type="BROADER_THAN", weight=sim)
                    G.add_edge(f"group_{g_narrow.id}", f"group_{g_broad.id}", edge_type="NARROWER_THAN", weight=sim)


# ── Stats ─────────────────────────────────────────────────────────────────


def _print_graph_stats(G: nx.DiGraph) -> None:
    """Print graph statistics."""
    print(f"\n{'='*60}")
    print("GRAPH STATISTICS")
    print(f"{'='*60}")
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")

    node_types = defaultdict(int)
    for _, data in G.nodes(data=True):
        node_types[data.get("node_type", "unknown")] += 1
    print("\nNodes by type:")
    for nt, count in sorted(node_types.items()):
        print(f"  {nt}: {count}")

    edge_types = defaultdict(int)
    for _, _, data in G.edges(data=True):
        edge_types[data.get("edge_type", "unknown")] += 1
    print("\nEdges by type:")
    for et, count in sorted(edge_types.items()):
        print(f"  {et}: {count}")

    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))
    print(f"\nConnected components: {len(components)}")
    if components:
        sizes = sorted([len(c) for c in components], reverse=True)
        print(f"  Largest: {sizes[0]} nodes")
        if len(sizes) > 1:
            print(f"  2nd largest: {sizes[1]} nodes")
        print(f"  Smallest: {sizes[-1]} nodes")


# ── Persistence ───────────────────────────────────────────────────────────


def save_graph(
    G: nx.DiGraph,
    output_dir: Path = config.GRAPHS_DIR,
    filename: str = "audience_graph.graphml",
) -> None:
    """Save graph to GraphML format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    for node, data in G.nodes(data=True):
        for key, val in list(data.items()):
            if isinstance(val, (list, set)):
                data[key] = ", ".join(str(v) for v in val)

    nx.write_graphml(G, path)
    print(f"Graph saved to {path}")


def load_graph(
    output_dir: Path = config.GRAPHS_DIR,
    filename: str = "audience_graph.graphml",
) -> nx.DiGraph:
    """Load graph from GraphML format."""
    return nx.read_graphml(output_dir / filename)


if __name__ == "__main__":
    from data_loader import load_all
    from embedder import load_artifacts
    from clustering import load_clusters_v2

    segments = load_all()
    embeddings, index, segment_ids = load_artifacts(prefix="v2")
    super_cats, sub_cats, l0_labels, l1_labels = load_clusters_v2()

    G = build_graph_v2(segments, super_cats, sub_cats, embeddings, l1_labels)
    save_graph(G, filename="audience_graph_v2.graphml")
