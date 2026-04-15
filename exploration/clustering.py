"""Cluster audience segments using HDBSCAN to find cross-platform groups.

v1: Single-pass HDBSCAN -> flat AudienceGroup list.
v2: Two-pass hierarchical HDBSCAN -> SuperCategory (Layer 0) + SubCategory (Layer 1).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import hdbscan
import numpy as np

import config
from data_loader import Segment


# ── v1 dataclass (kept for backward compatibility) ────────────────────────


@dataclass
class AudienceGroup:
    """A cluster of semantically related segments across platforms (v1)."""

    id: int
    name: str
    segment_ids: list[str] = field(default_factory=list)
    centroid: np.ndarray | None = field(default=None, repr=False)
    platforms: set[str] = field(default_factory=set)
    member_count: int = 0
    top_iab_category: str = ""


# ── v2 dataclasses ────────────────────────────────────────────────────────


@dataclass
class SuperCategory:
    """Layer 0: Broad audience super-category (e.g., 'Automotive', 'Sports')."""

    id: int
    name: str
    subcategory_ids: list[int] = field(default_factory=list)
    centroid: np.ndarray | None = field(default=None, repr=False)
    platforms: set[str] = field(default_factory=set)
    member_count: int = 0


@dataclass
class SubCategory:
    """Layer 1: Specific audience sub-category (e.g., 'Luxury Vehicles', 'Electric Vehicles')."""

    id: int
    name: str
    parent_id: int = -1
    segment_ids: list[str] = field(default_factory=list)
    centroid: np.ndarray | None = field(default=None, repr=False)
    platforms: set[str] = field(default_factory=set)
    member_count: int = 0


# ── v1: Single-pass clustering ────────────────────────────────────────────


def cluster_segments(
    embeddings: np.ndarray,
    segments: list[Segment],
    min_cluster_size: int = config.HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = config.HDBSCAN_MIN_SAMPLES,
) -> tuple[np.ndarray, hdbscan.HDBSCAN]:
    """Run HDBSCAN clustering on the embedding matrix (v1)."""
    print(f"Running HDBSCAN on {len(embeddings)} embeddings...")
    print(f"  min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=config.HDBSCAN_METRIC,
        cluster_selection_method=config.HDBSCAN_CLUSTER_SELECTION,
    )
    labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Found {n_clusters} clusters, {n_noise} noise points ({n_noise/len(labels)*100:.1f}%)")

    return labels, clusterer


def build_audience_groups(
    labels: np.ndarray,
    embeddings: np.ndarray,
    segments: list[Segment],
) -> list[AudienceGroup]:
    """Build AudienceGroup objects from cluster labels (v1)."""
    cluster_to_segments: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label >= 0:
            cluster_to_segments[label].append(idx)

    groups: list[AudienceGroup] = []

    for cluster_id, member_indices in sorted(cluster_to_segments.items()):
        member_segs = [segments[i] for i in member_indices]
        member_embs = embeddings[member_indices]

        centroid = member_embs.astype(np.float64).mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroid = centroid.astype(np.float32)

        name = _name_cluster(member_segs, member_embs, centroid)
        platforms = set(s.platform for s in member_segs)

        iab_categories = []
        for s in member_segs:
            if s.segment_type == "iab_content" and s.hierarchy:
                iab_categories.append(s.hierarchy[0])
        top_iab = Counter(iab_categories).most_common(1)
        top_iab_name = top_iab[0][0] if top_iab else ""

        group = AudienceGroup(
            id=cluster_id,
            name=name,
            segment_ids=[s.id for s in member_segs],
            centroid=centroid,
            platforms=platforms,
            member_count=len(member_segs),
            top_iab_category=top_iab_name,
        )
        groups.append(group)

    return groups


# ── v2: Two-pass hierarchical clustering ──────────────────────────────────


def cluster_two_level(
    embeddings: np.ndarray,
    segments: list[Segment],
) -> tuple[list[SuperCategory], list[SubCategory], np.ndarray, np.ndarray]:
    """Two-pass HDBSCAN: Layer 0 (super-categories) then Layer 1 (sub-categories).

    Returns (super_categories, sub_categories, l0_labels, l1_labels).
    l0_labels[i] = super-category ID for segment i
    l1_labels[i] = sub-category ID for segment i
    """
    n = len(segments)

    # ── Pass 1: Layer 0 (broad super-categories) ─────────────────────
    print(f"\n{'='*60}")
    print("PASS 1: Layer 0 super-categories")
    print(f"{'='*60}")

    l0_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.LAYER0_CLUSTER_SIZE,
        min_samples=config.LAYER0_MIN_SAMPLES,
        metric=config.HDBSCAN_METRIC,
        cluster_selection_method=config.HDBSCAN_CLUSTER_SELECTION,
    )
    l0_labels_raw = l0_clusterer.fit_predict(embeddings)

    n_l0_clusters = len(set(l0_labels_raw)) - (1 if -1 in l0_labels_raw else 0)
    n_l0_noise = (l0_labels_raw == -1).sum()
    print(f"  Found {n_l0_clusters} super-categories, {n_l0_noise} noise points ({n_l0_noise/n*100:.1f}%)")

    # Compute Layer 0 centroids and assign noise
    l0_centroids = _compute_centroids(l0_labels_raw, embeddings)
    l0_labels = _assign_noise(l0_labels_raw, embeddings, l0_centroids)
    l0_centroids = _compute_centroids(l0_labels, embeddings)  # recompute after noise assignment

    # Build SuperCategory objects
    super_categories: list[SuperCategory] = []
    l0_cluster_ids = sorted(set(l0_labels))

    for l0_id in l0_cluster_ids:
        mask = l0_labels == l0_id
        member_segs = [segments[i] for i in range(n) if mask[i]]
        member_embs = embeddings[mask]

        name = _name_from_members(member_segs, member_embs, l0_centroids[l0_id])
        platforms = set(s.platform for s in member_segs)

        super_categories.append(SuperCategory(
            id=l0_id,
            name=name,
            centroid=l0_centroids[l0_id],
            platforms=platforms,
            member_count=int(mask.sum()),
        ))

    print(f"  Built {len(super_categories)} super-categories")

    # ── Pass 2: Layer 1 (sub-categories within each super-category) ──
    print(f"\n{'='*60}")
    print("PASS 2: Layer 1 sub-categories")
    print(f"{'='*60}")

    sub_categories: list[SubCategory] = []
    l1_labels = np.full(n, -1, dtype=np.int64)
    next_sub_id = 0

    for sc in super_categories:
        l0_mask = l0_labels == sc.id
        l0_indices = np.where(l0_mask)[0]
        l0_embs = embeddings[l0_indices]

        if len(l0_indices) < config.LAYER1_CLUSTER_SIZE:
            # Too small to sub-cluster — treat entire super-category as one sub-category
            sub_id = next_sub_id
            next_sub_id += 1

            member_segs = [segments[i] for i in l0_indices]
            centroid = l0_embs.astype(np.float64).mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroid = centroid.astype(np.float32)

            name = _name_from_members(member_segs, l0_embs, centroid)
            platforms = set(s.platform for s in member_segs)

            sub_categories.append(SubCategory(
                id=sub_id,
                name=name,
                parent_id=sc.id,
                segment_ids=[s.id for s in member_segs],
                centroid=centroid,
                platforms=platforms,
                member_count=len(member_segs),
            ))
            sc.subcategory_ids.append(sub_id)

            for idx in l0_indices:
                l1_labels[idx] = sub_id

            continue

        # Run HDBSCAN within this super-category
        l1_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config.LAYER1_CLUSTER_SIZE,
            min_samples=config.LAYER1_MIN_SAMPLES,
            metric=config.HDBSCAN_METRIC,
            cluster_selection_method=config.HDBSCAN_CLUSTER_SELECTION,
        )
        local_labels_raw = l1_clusterer.fit_predict(l0_embs)

        # If HDBSCAN found 0 clusters (all noise), treat as single sub-category
        n_found = len(set(local_labels_raw)) - (1 if -1 in local_labels_raw else 0)
        if n_found == 0:
            sub_id = next_sub_id
            next_sub_id += 1
            member_segs = [segments[i] for i in l0_indices]
            centroid = l0_embs.astype(np.float64).mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroid = centroid.astype(np.float32)
            name = _name_from_members(member_segs, l0_embs, centroid)
            platforms = set(s.platform for s in member_segs)
            sub_categories.append(SubCategory(
                id=sub_id, name=name, parent_id=sc.id,
                segment_ids=[s.id for s in member_segs],
                centroid=centroid, platforms=platforms,
                member_count=len(member_segs),
            ))
            sc.subcategory_ids.append(sub_id)
            for idx in l0_indices:
                l1_labels[idx] = sub_id
            print(f"  [{sc.name}] {len(l0_indices)} segments -> 1 sub-category (no sub-clusters found)")
            continue

        # Compute local centroids and assign noise within this super-category
        local_centroids = _compute_centroids(local_labels_raw, l0_embs)
        local_labels = _assign_noise(local_labels_raw, l0_embs, local_centroids)
        local_centroids = _compute_centroids(local_labels, l0_embs)

        n_local = len(set(local_labels))
        print(f"  [{sc.name}] {len(l0_indices)} segments -> {n_local} sub-categories")

        for local_id in sorted(set(local_labels)):
            sub_id = next_sub_id
            next_sub_id += 1

            local_mask = local_labels == local_id
            local_member_indices = l0_indices[local_mask]
            member_segs = [segments[i] for i in local_member_indices]
            member_embs = embeddings[local_member_indices]

            centroid = local_centroids[local_id]
            name = _name_from_members(member_segs, member_embs, centroid)
            platforms = set(s.platform for s in member_segs)

            sub_categories.append(SubCategory(
                id=sub_id,
                name=name,
                parent_id=sc.id,
                segment_ids=[s.id for s in member_segs],
                centroid=centroid,
                platforms=platforms,
                member_count=len(member_segs),
            ))
            sc.subcategory_ids.append(sub_id)

            for idx in local_member_indices:
                l1_labels[idx] = sub_id

    print(f"\n  Total: {len(sub_categories)} sub-categories across {len(super_categories)} super-categories")

    return super_categories, sub_categories, l0_labels, l1_labels


# ── Helpers ───────────────────────────────────────────────────────────────


def _compute_centroids(labels: np.ndarray, embeddings: np.ndarray) -> dict[int, np.ndarray]:
    """Compute L2-normalized centroid for each cluster."""
    centroids: dict[int, np.ndarray] = {}
    for cid in set(labels):
        if cid < 0:
            continue
        mask = labels == cid
        centroid = embeddings[mask].astype(np.float64).mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[cid] = centroid.astype(np.float32)
    return centroids


def _assign_noise(
    labels: np.ndarray,
    embeddings: np.ndarray,
    centroids: dict[int, np.ndarray],
) -> np.ndarray:
    """Assign noise points (-1) to nearest cluster centroid."""
    noise_mask = labels == -1
    if not noise_mask.any() or not centroids:
        return labels.copy()

    centroid_ids = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[cid] for cid in centroid_ids], dtype=np.float32)

    noise_embs = embeddings[noise_mask].astype(np.float64)
    sims = noise_embs @ centroid_matrix.astype(np.float64).T
    nearest = sims.argmax(axis=1)

    updated = labels.copy()
    noise_indices = np.where(noise_mask)[0]
    for i, noise_idx in enumerate(noise_indices):
        updated[noise_idx] = centroid_ids[nearest[i]]

    print(f"  Assigned {noise_mask.sum()} noise points to nearest clusters")
    return updated


def _name_cluster(
    members: list[Segment],
    member_embs: np.ndarray,
    centroid: np.ndarray,
) -> str:
    """Name a cluster using the member closest to the centroid (v1)."""
    sims = member_embs.astype(np.float64) @ centroid.astype(np.float64)
    closest_idx = sims.argmax()
    closest = members[closest_idx]

    if len(closest.hierarchy) > 1:
        return " > ".join(closest.hierarchy[:2])
    return closest.name


def _name_from_members(
    members: list[Segment],
    member_embs: np.ndarray,
    centroid: np.ndarray,
) -> str:
    """Name a category using the member closest to the centroid.

    Prefers IAB Tier 1 names for Layer 0, specific names for Layer 1.
    """
    if len(members) == 0:
        return "Unknown"

    sims = member_embs.astype(np.float64) @ centroid.astype(np.float64)
    closest_idx = sims.argmax()
    closest = members[closest_idx]

    # Use hierarchy for context
    if len(closest.hierarchy) > 1:
        return " > ".join(closest.hierarchy[:2])
    return closest.name


def assign_noise_to_nearest(
    labels: np.ndarray,
    embeddings: np.ndarray,
    groups: list[AudienceGroup],
) -> np.ndarray:
    """Assign noise points to nearest cluster centroid (v1 compat)."""
    centroids = {g.id: g.centroid for g in groups if g.centroid is not None}
    return _assign_noise(labels, embeddings, centroids)


# ── Persistence ───────────────────────────────────────────────────────────


def save_clusters(
    groups: list[AudienceGroup],
    labels: np.ndarray,
    output_dir: Path = config.EMBEDDINGS_DIR,
    prefix: str = "raw",
) -> None:
    """Save v1 cluster data to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{prefix}_labels.npy", labels)

    groups_data = []
    for g in groups:
        groups_data.append({
            "id": int(g.id),
            "name": g.name,
            "segment_ids": g.segment_ids,
            "platforms": list(g.platforms),
            "member_count": int(g.member_count),
            "top_iab_category": g.top_iab_category,
        })

    with open(output_dir / f"{prefix}_groups.json", "w") as f:
        json.dump(groups_data, f, indent=2)

    centroids = np.array([g.centroid for g in groups], dtype=np.float32)
    np.save(output_dir / f"{prefix}_centroids.npy", centroids)

    print(f"Saved {len(groups)} clusters to {output_dir}/ with prefix '{prefix}'")


def load_clusters(
    output_dir: Path = config.EMBEDDINGS_DIR,
    prefix: str = "raw",
) -> tuple[list[AudienceGroup], np.ndarray, np.ndarray]:
    """Load v1 saved clusters."""
    labels = np.load(output_dir / f"{prefix}_labels.npy")
    centroids = np.load(output_dir / f"{prefix}_centroids.npy")

    with open(output_dir / f"{prefix}_groups.json") as f:
        groups_data = json.load(f)

    groups = []
    for gd, centroid in zip(groups_data, centroids):
        groups.append(AudienceGroup(
            id=gd["id"],
            name=gd["name"],
            segment_ids=gd["segment_ids"],
            centroid=centroid,
            platforms=set(gd["platforms"]),
            member_count=gd["member_count"],
            top_iab_category=gd["top_iab_category"],
        ))

    return groups, labels, centroids


def save_clusters_v2(
    super_categories: list[SuperCategory],
    sub_categories: list[SubCategory],
    l0_labels: np.ndarray,
    l1_labels: np.ndarray,
    output_dir: Path = config.EMBEDDINGS_DIR,
    prefix: str = "v2",
) -> None:
    """Save v2 two-level cluster data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{prefix}_l0_labels.npy", l0_labels)
    np.save(output_dir / f"{prefix}_l1_labels.npy", l1_labels)

    # Super-categories
    sc_data = []
    sc_centroids = []
    for sc in super_categories:
        sc_data.append({
            "id": int(sc.id),
            "name": sc.name,
            "subcategory_ids": [int(x) for x in sc.subcategory_ids],
            "platforms": list(sc.platforms),
            "member_count": int(sc.member_count),
        })
        sc_centroids.append(sc.centroid)

    with open(output_dir / f"{prefix}_super_categories.json", "w") as f:
        json.dump(sc_data, f, indent=2)
    np.save(output_dir / f"{prefix}_l0_centroids.npy",
            np.array(sc_centroids, dtype=np.float32))

    # Sub-categories
    sub_data = []
    sub_centroids = []
    for sub in sub_categories:
        sub_data.append({
            "id": int(sub.id),
            "name": sub.name,
            "parent_id": int(sub.parent_id),
            "segment_ids": sub.segment_ids,
            "platforms": list(sub.platforms),
            "member_count": int(sub.member_count),
        })
        sub_centroids.append(sub.centroid)

    with open(output_dir / f"{prefix}_sub_categories.json", "w") as f:
        json.dump(sub_data, f, indent=2)
    np.save(output_dir / f"{prefix}_l1_centroids.npy",
            np.array(sub_centroids, dtype=np.float32))

    print(f"Saved v2 clusters: {len(super_categories)} super-cats, {len(sub_categories)} sub-cats")


def load_clusters_v2(
    output_dir: Path = config.EMBEDDINGS_DIR,
    prefix: str = "v2",
) -> tuple[list[SuperCategory], list[SubCategory], np.ndarray, np.ndarray]:
    """Load v2 two-level cluster data."""
    l0_labels = np.load(output_dir / f"{prefix}_l0_labels.npy")
    l1_labels = np.load(output_dir / f"{prefix}_l1_labels.npy")

    l0_centroids = np.load(output_dir / f"{prefix}_l0_centroids.npy")
    l1_centroids = np.load(output_dir / f"{prefix}_l1_centroids.npy")

    with open(output_dir / f"{prefix}_super_categories.json") as f:
        sc_data = json.load(f)

    super_categories = []
    for sd, centroid in zip(sc_data, l0_centroids):
        super_categories.append(SuperCategory(
            id=sd["id"],
            name=sd["name"],
            subcategory_ids=sd["subcategory_ids"],
            centroid=centroid,
            platforms=set(sd["platforms"]),
            member_count=sd["member_count"],
        ))

    with open(output_dir / f"{prefix}_sub_categories.json") as f:
        sub_data = json.load(f)

    sub_categories = []
    for sd, centroid in zip(sub_data, l1_centroids):
        sub_categories.append(SubCategory(
            id=sd["id"],
            name=sd["name"],
            parent_id=sd["parent_id"],
            segment_ids=sd["segment_ids"],
            centroid=centroid,
            platforms=set(sd["platforms"]),
            member_count=sd["member_count"],
        ))

    return super_categories, sub_categories, l0_labels, l1_labels


# ── Validation ────────────────────────────────────────────────────────────


def print_cluster_summary(groups: list[AudienceGroup], segments: list[Segment]) -> None:
    """Print summary of v1 clustering results."""
    id_to_seg = {s.id: s for s in segments}

    print(f"\n{'='*70}")
    print(f"CLUSTERING SUMMARY: {len(groups)} audience groups")
    print(f"{'='*70}")

    multi_platform = [g for g in groups if len(g.platforms) > 1]
    print(f"\nGroups with 2+ platforms: {len(multi_platform)} ({len(multi_platform)/len(groups)*100:.1f}%)")

    platform_dist = Counter()
    for g in groups:
        platform_dist[len(g.platforms)] += 1
    print("Platform count distribution:")
    for n_plat, count in sorted(platform_dist.items()):
        print(f"  {n_plat} platforms: {count} groups")

    sizes = [g.member_count for g in groups]
    print(f"\nGroup size: min={min(sizes)}, max={max(sizes)}, median={sorted(sizes)[len(sizes)//2]}, avg={sum(sizes)/len(sizes):.1f}")

    sorted_groups = sorted(groups, key=lambda g: g.member_count, reverse=True)
    print(f"\n{'='*70}")
    print("TOP 20 LARGEST GROUPS (cross-platform analysis)")
    print(f"{'='*70}")

    for g in sorted_groups[:20]:
        platforms_str = ", ".join(sorted(g.platforms))
        print(f"\n[Group {g.id}] \"{g.name}\" -- {g.member_count} members, {len(g.platforms)} platforms")
        print(f"  Platforms: {platforms_str}")
        if g.top_iab_category:
            print(f"  Top IAB: {g.top_iab_category}")

        by_platform = defaultdict(list)
        for sid in g.segment_ids:
            seg = id_to_seg.get(sid)
            if seg:
                by_platform[seg.platform].append(seg)

        for plat in sorted(by_platform):
            members = by_platform[plat][:3]
            names = [m.name for m in members]
            extra = f" (+{len(by_platform[plat])-3} more)" if len(by_platform[plat]) > 3 else ""
            print(f"    {plat:12s}: {', '.join(names)}{extra}")


def print_v2_summary(
    super_categories: list[SuperCategory],
    sub_categories: list[SubCategory],
    segments: list[Segment],
) -> None:
    """Print summary of v2 two-level clustering."""
    id_to_seg = {s.id: s for s in segments}

    print(f"\n{'='*70}")
    print(f"V2 CLUSTERING SUMMARY")
    print(f"{'='*70}")
    print(f"Layer 0 (super-categories): {len(super_categories)}")
    print(f"Layer 1 (sub-categories):   {len(sub_categories)}")
    print(f"Layer 2 (segments):         {len(segments)}")

    # Super-category breakdown
    print(f"\n{'='*70}")
    print("SUPER-CATEGORIES (Layer 0)")
    print(f"{'='*70}")

    for sc in sorted(super_categories, key=lambda x: x.member_count, reverse=True):
        platforms_str = ", ".join(sorted(sc.platforms))
        n_subs = len(sc.subcategory_ids)
        print(f"\n[{sc.id}] \"{sc.name}\" -- {sc.member_count} segments, {n_subs} sub-cats, {len(sc.platforms)} platforms")
        print(f"  Platforms: {platforms_str}")

        # Show sub-categories
        subs = [sub for sub in sub_categories if sub.parent_id == sc.id]
        for sub in sorted(subs, key=lambda x: x.member_count, reverse=True)[:5]:
            sub_plats = ", ".join(sorted(sub.platforms))
            print(f"    -> [{sub.id}] \"{sub.name}\" ({sub.member_count} segs, {sub_plats})")

        remaining = len(subs) - 5
        if remaining > 0:
            print(f"    ... +{remaining} more sub-categories")


# ── CLI ────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    from data_loader import load_all
    from embedder import load_artifacts

    segments = load_all()

    # Load v2 embeddings (or fall back to enriched/raw)
    try:
        embeddings, index, segment_ids = load_artifacts(prefix="v2")
    except FileNotFoundError:
        print("v2 embeddings not found, using enriched...")
        embeddings, index, segment_ids = load_artifacts(prefix="enriched")

    # v2 two-level clustering
    super_cats, sub_cats, l0_labels, l1_labels = cluster_two_level(embeddings, segments)

    # Save
    save_clusters_v2(super_cats, sub_cats, l0_labels, l1_labels)

    # Print summary
    print_v2_summary(super_cats, sub_cats, segments)
