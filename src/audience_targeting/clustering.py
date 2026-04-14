"""Two-pass hierarchical HDBSCAN clustering.

Produces SuperCategory (Layer 0) + SubCategory (Layer 1) hierarchy from embeddings.
"""

from __future__ import annotations

from collections import defaultdict

import hdbscan
import numpy as np

from audience_targeting.models import Segment, SubCategory, SuperCategory
from audience_targeting.settings import Settings


def cluster_two_level(
    embeddings: np.ndarray,
    segments: list[Segment],
    settings: Settings | None = None,
) -> tuple[list[SuperCategory], list[SubCategory], np.ndarray, np.ndarray]:
    """Two-pass HDBSCAN: Layer 0 (super-categories) then Layer 1 (sub-categories).

    Returns (super_categories, sub_categories, l0_labels, l1_labels).
    """
    if settings is None:
        settings = Settings()

    n = len(segments)

    # ── Pass 1: Layer 0 (broad super-categories) ─────────────────────
    print(f"\n{'='*60}")
    print("PASS 1: Layer 0 super-categories")
    print(f"{'='*60}")

    l0_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=settings.layer0_cluster_size,
        min_samples=settings.layer0_min_samples,
        metric=settings.hdbscan_metric,
        cluster_selection_method=settings.hdbscan_cluster_selection,
    )
    l0_labels_raw = l0_clusterer.fit_predict(embeddings)

    n_l0 = len(set(l0_labels_raw)) - (1 if -1 in l0_labels_raw else 0)
    n_noise = (l0_labels_raw == -1).sum()
    print(f"  Found {n_l0} super-categories, {n_noise} noise points ({n_noise/n*100:.1f}%)")

    l0_centroids = _compute_centroids(l0_labels_raw, embeddings)
    l0_labels = _assign_noise(l0_labels_raw, embeddings, l0_centroids)
    l0_centroids = _compute_centroids(l0_labels, embeddings)

    super_categories: list[SuperCategory] = []
    for l0_id in sorted(set(l0_labels)):
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

        if len(l0_indices) < settings.layer1_cluster_size:
            sub_id = next_sub_id
            next_sub_id += 1

            member_segs = [segments[i] for i in l0_indices]
            centroid = _single_centroid(l0_embs)
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
            continue

        l1_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=settings.layer1_cluster_size,
            min_samples=settings.layer1_min_samples,
            metric=settings.hdbscan_metric,
            cluster_selection_method=settings.hdbscan_cluster_selection,
        )
        local_labels_raw = l1_clusterer.fit_predict(l0_embs)

        n_found = len(set(local_labels_raw)) - (1 if -1 in local_labels_raw else 0)
        if n_found == 0:
            sub_id = next_sub_id
            next_sub_id += 1
            member_segs = [segments[i] for i in l0_indices]
            centroid = _single_centroid(l0_embs)
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
                id=sub_id, name=name, parent_id=sc.id,
                segment_ids=[s.id for s in member_segs],
                centroid=centroid, platforms=platforms,
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


def _single_centroid(embs: np.ndarray) -> np.ndarray:
    """Compute a single L2-normalized centroid from a set of embeddings."""
    centroid = embs.astype(np.float64).mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid.astype(np.float32)


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


def _name_from_members(
    members: list[Segment],
    member_embs: np.ndarray,
    centroid: np.ndarray,
) -> str:
    """Name a category using the member closest to the centroid."""
    if len(members) == 0:
        return "Unknown"

    sims = member_embs.astype(np.float64) @ centroid.astype(np.float64)
    closest_idx = sims.argmax()
    closest = members[closest_idx]

    if len(closest.hierarchy) > 1:
        return " > ".join(closest.hierarchy[:2])
    return closest.name
