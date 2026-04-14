"""Compute structural relationships between segments (extracted from graph_builder.py).

These relationships are stored as metadata in Qdrant rather than as graph edges.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from audience_targeting.models import Segment, SubCategory, SuperCategory


def compute_parent_segment_ids(segments: list[Segment]) -> dict[str, str]:
    """Compute IS_CHILD_OF relationships: {child_segment_id: parent_segment_id}.

    - IAB hierarchy: Tier 2 -> Tier 1 using category_id prefix (e.g., IAB2-10 -> IAB2)
    - Social hierarchy: Sub-Segment -> Category
    """
    by_platform: dict[str, list[Segment]] = defaultdict(list)
    for s in segments:
        by_platform[s.platform].append(s)

    parent_map: dict[str, str] = {}

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
                pid = cat_id.rsplit("-", 1)[0]
                parent_seg = cat_id_to_seg.get(pid)
                if parent_seg:
                    parent_map[seg.id] = parent_seg.id

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
                parent_map[s.id] = parent.id

    return parent_map


def compute_subcategory_map(
    segments: list[Segment],
    l1_labels: np.ndarray,
    sub_categories: list[SubCategory],
) -> dict[str, int]:
    """Map each segment to its sub-category ID: {segment_id: subcategory_id}."""
    seg_to_sub: dict[str, int] = {}
    for i, seg in enumerate(segments):
        sub_id = int(l1_labels[i])
        if sub_id >= 0:
            seg_to_sub[seg.id] = sub_id
    return seg_to_sub


def compute_super_category_map(
    sub_categories: list[SubCategory],
) -> dict[int, int]:
    """Map each sub-category to its super-category: {sub_id: super_id}."""
    return {sub.id: sub.parent_id for sub in sub_categories}


def compute_subcategory_relationships(
    sub_categories: list[SubCategory],
    similarity_threshold_related: float = 0.65,
    similarity_threshold_equivalent: float = 0.85,
) -> dict[int, dict[str, list[int]]]:
    """Pre-compute related/broader/narrower relationships between sub-categories.

    Returns {sub_id: {"related": [ids], "broader": [ids], "narrower": [ids]}}.
    - related: different parent, centroid similarity in [related_threshold, equivalent_threshold)
    - broader: same parent, >=1.5x members, similarity >= 0.6
    - narrower: same parent, <=0.67x members, similarity >= 0.6
    """
    # Build centroid matrix
    valid = [sub for sub in sub_categories if sub.centroid is not None]
    if not valid:
        return {}

    centroids = np.stack([sub.centroid.astype(np.float64) for sub in valid])
    # L2-normalize for cosine via dot product
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centroids = centroids / norms

    sim_matrix = centroids @ centroids.T

    id_to_idx = {sub.id: i for i, sub in enumerate(valid)}
    relationships: dict[int, dict[str, list[int]]] = {}

    for sub in valid:
        idx = id_to_idx[sub.id]
        related: list[int] = []
        broader: list[int] = []
        narrower: list[int] = []

        for other in valid:
            if other.id == sub.id:
                continue
            other_idx = id_to_idx[other.id]
            sim = float(sim_matrix[idx, other_idx])

            # Related: different parent, sim in [related, equivalent)
            if other.parent_id != sub.parent_id:
                if similarity_threshold_related <= sim < similarity_threshold_equivalent:
                    related.append(other.id)
            else:
                # Same parent — check broader/narrower
                if sim < 0.6:
                    continue
                sub_count = max(sub.member_count, 1)
                other_count = max(other.member_count, 1)
                ratio = other_count / sub_count
                if ratio >= 1.5:
                    broader.append(other.id)
                elif ratio <= 0.67:
                    narrower.append(other.id)

        relationships[sub.id] = {
            "related": related,
            "broader": broader,
            "narrower": narrower,
        }

    return relationships
