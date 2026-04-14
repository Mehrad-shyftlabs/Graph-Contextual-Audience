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
