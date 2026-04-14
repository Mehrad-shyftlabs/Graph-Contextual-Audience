"""Qdrant vector database operations — collection creation, ingestion, and query helpers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    NamedVector,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from audience_targeting.models import Segment, SubCategory, SuperCategory
from audience_targeting.settings import Settings

logger = logging.getLogger(__name__)


# ── Collection creation ──────────────────────────────────────────────────


def create_collections(client: QdrantClient, settings: Settings) -> None:
    """Create (or recreate) the 3 Qdrant collections with payload indexes."""

    # supercategories: single BGE vector
    client.recreate_collection(
        collection_name=settings.collection_name("supercategories"),
        vectors_config={
            "bge": VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        },
    )

    # subcategories: BGE + optional Node2Vec
    vectors = {"bge": VectorParams(size=settings.embedding_dim, distance=Distance.COSINE)}
    if settings.use_node2vec:
        vectors["node2vec"] = VectorParams(size=settings.node2vec_dim, distance=Distance.COSINE)
    client.recreate_collection(
        collection_name=settings.collection_name("subcategories"),
        vectors_config=vectors,
    )

    # segments: BGE + optional Node2Vec
    vectors = {"bge": VectorParams(size=settings.embedding_dim, distance=Distance.COSINE)}
    if settings.use_node2vec:
        vectors["node2vec"] = VectorParams(size=settings.node2vec_dim, distance=Distance.COSINE)
    client.recreate_collection(
        collection_name=settings.collection_name("segments"),
        vectors_config=vectors,
    )

    # Payload indexes for filtered search
    _create_payload_indexes(client, settings)
    logger.info("Created 3 Qdrant collections with payload indexes")


def _create_payload_indexes(client: QdrantClient, settings: Settings) -> None:
    """Create payload indexes for efficient filtered search."""
    client.create_payload_index(
        settings.collection_name("supercategories"), "super_id", PayloadSchemaType.INTEGER
    )
    for field in ["parent_super_id", "sub_id"]:
        client.create_payload_index(
            settings.collection_name("subcategories"), field, PayloadSchemaType.INTEGER
        )
    for field in ["subcategory_id", "super_category_id", "platform", "segment_id"]:
        schema = PayloadSchemaType.KEYWORD if field in ("platform", "segment_id") else PayloadSchemaType.INTEGER
        client.create_payload_index(
            settings.collection_name("segments"), field, schema
        )


# ── Ingestion ────────────────────────────────────────────────────────────


def _py(val):
    """Convert numpy scalars to native Python types for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def ingest_supercategories(
    client: QdrantClient,
    super_categories: list[SuperCategory],
    settings: Settings,
) -> None:
    """Upsert super-category points into Qdrant."""
    points = []
    for i, sc in enumerate(super_categories):
        points.append(PointStruct(
            id=i,
            vector={"bge": sc.centroid.tolist()},
            payload={
                "super_id": int(sc.id),
                "name": sc.name,
                "subcategory_ids": [int(x) for x in sc.subcategory_ids],
                "platforms": list(sc.platforms),
                "member_count": int(sc.member_count),
            },
        ))

    client.upsert(
        collection_name=settings.collection_name("supercategories"),
        points=points,
    )
    logger.info(f"Ingested {len(points)} super-categories")


def ingest_subcategories(
    client: QdrantClient,
    sub_categories: list[SubCategory],
    node2vec_embeddings: dict[str, np.ndarray] | None,
    settings: Settings,
) -> None:
    """Upsert sub-category points into Qdrant."""
    points = []
    for i, sub in enumerate(sub_categories):
        vectors: dict[str, list[float]] = {"bge": sub.centroid.tolist()}

        if settings.use_node2vec and node2vec_embeddings:
            n2v_key = f"sub_{sub.id}"
            if n2v_key in node2vec_embeddings:
                vectors["node2vec"] = node2vec_embeddings[n2v_key].tolist()

        points.append(PointStruct(
            id=i,
            vector=vectors,
            payload={
                "sub_id": int(sub.id),
                "name": sub.name,
                "parent_super_id": int(sub.parent_id),
                "segment_ids": [str(s) for s in sub.segment_ids],
                "platforms": list(sub.platforms),
                "member_count": int(sub.member_count),
            },
        ))

    client.upsert(
        collection_name=settings.collection_name("subcategories"),
        points=points,
    )
    logger.info(f"Ingested {len(points)} sub-categories")


def ingest_segments(
    client: QdrantClient,
    segments: list[Segment],
    embeddings: np.ndarray,
    subcategory_map: dict[str, int],
    super_category_map: dict[int, int],
    parent_segment_map: dict[str, str],
    node2vec_embeddings: dict[str, np.ndarray] | None,
    settings: Settings,
    batch_size: int = 100,
) -> None:
    """Upsert segment points into Qdrant in batches."""
    total = 0
    batch: list[PointStruct] = []

    for i, seg in enumerate(segments):
        sub_id = subcategory_map.get(seg.id, -1)
        super_id = super_category_map.get(sub_id, -1)

        vectors: dict[str, list[float]] = {"bge": embeddings[i].tolist()}

        if settings.use_node2vec and node2vec_embeddings:
            if seg.id in node2vec_embeddings:
                vectors["node2vec"] = node2vec_embeddings[seg.id].tolist()

        batch.append(PointStruct(
            id=i,
            vector=vectors,
            payload={
                "segment_id": str(seg.id),
                "name": str(seg.name),
                "platform": str(seg.platform),
                "subcategory_id": int(sub_id),
                "super_category_id": int(super_id),
                "hierarchy": [str(h) for h in seg.hierarchy],
                "hierarchy_text": " > ".join(seg.hierarchy),
                "segment_type": str(seg.segment_type or ""),
                "audience_size": int(seg.audience_size) if seg.audience_size else 0,
                "description": str(seg.description or ""),
                "parent_segment_id": parent_segment_map.get(seg.id),
            },
        ))

        if len(batch) >= batch_size:
            client.upsert(
                collection_name=settings.collection_name("segments"),
                points=batch,
            )
            total += len(batch)
            batch = []

    if batch:
        client.upsert(
            collection_name=settings.collection_name("segments"),
            points=batch,
        )
        total += len(batch)

    logger.info(f"Ingested {total} segments")


# ── Search helpers ───────────────────────────────────────────────────────


def search_supercategories(
    client: QdrantClient,
    query_vector: list[float],
    settings: Settings,
    score_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """Search super-categories by BGE vector similarity."""
    results = client.search(
        collection_name=settings.collection_name("supercategories"),
        query_vector=NamedVector(name="bge", vector=query_vector),
        limit=settings.layer0_top_k,
        score_threshold=score_threshold,
    )
    return [{"score": r.score, **r.payload} for r in results]


def search_subcategories(
    client: QdrantClient,
    query_vector: list[float],
    parent_super_ids: list[int],
    settings: Settings,
    with_vectors: bool = False,
) -> list[dict[str, Any]]:
    """Search sub-categories filtered by parent super-category IDs."""
    results = client.search(
        collection_name=settings.collection_name("subcategories"),
        query_vector=NamedVector(name="bge", vector=query_vector),
        query_filter=Filter(must=[
            FieldCondition(key="parent_super_id", match=MatchAny(any=parent_super_ids)),
        ]),
        limit=settings.layer1_top_k,
        with_vectors=with_vectors,
    )
    out = []
    for r in results:
        entry = {"score": r.score, "point_id": r.id, **r.payload}
        if with_vectors and r.vector:
            entry["vectors"] = r.vector
        out.append(entry)
    return out


def search_segments(
    client: QdrantClient,
    query_vector: list[float],
    subcategory_ids: list[int],
    settings: Settings,
    platforms: list[str] | None = None,
    top_k: int | None = None,
    with_vectors: bool = False,
) -> list[dict[str, Any]]:
    """Search segments filtered by sub-category IDs and optionally by platform."""
    must_conditions = [
        FieldCondition(key="subcategory_id", match=MatchAny(any=subcategory_ids)),
    ]
    if platforms:
        must_conditions.append(
            FieldCondition(key="platform", match=MatchAny(any=platforms)),
        )

    limit = top_k or (settings.layer2_top_k * 6)
    results = client.search(
        collection_name=settings.collection_name("segments"),
        query_vector=NamedVector(name="bge", vector=query_vector),
        query_filter=Filter(must=must_conditions),
        limit=limit,
        with_vectors=with_vectors,
    )
    out = []
    for r in results:
        entry = {"score": r.score, "point_id": r.id, **r.payload}
        if with_vectors and r.vector:
            entry["vectors"] = r.vector
        out.append(entry)
    return out


def get_related_subcategories(
    client: QdrantClient,
    centroid_vector: list[float],
    exclude_sub_ids: list[int],
    exclude_parent_id: int | None,
    settings: Settings,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Find sub-categories with centroid similarity in [0.65, 0.85) — the RELATED_TO replacement."""
    must_not = [
        FieldCondition(key="sub_id", match=MatchAny(any=exclude_sub_ids)),
    ]
    if exclude_parent_id is not None:
        must_not.append(
            FieldCondition(key="parent_super_id", match=MatchValue(value=exclude_parent_id)),
        )

    results = client.search(
        collection_name=settings.collection_name("subcategories"),
        query_vector=NamedVector(name="bge", vector=centroid_vector),
        query_filter=Filter(must_not=must_not),
        limit=max_results * 3,  # over-fetch to filter by score range
        score_threshold=settings.similarity_threshold_related,
    )

    related = []
    for r in results:
        if r.score < settings.similarity_threshold_equivalent:
            related.append({"score": r.score, **r.payload})
        if len(related) >= max_results:
            break

    return related


def get_siblings(
    client: QdrantClient,
    parent_super_id: int,
    exclude_sub_ids: list[int],
    settings: Settings,
    with_vectors: bool = True,
) -> list[dict[str, Any]]:
    """Get sibling sub-categories (same parent), used for broaden/narrow."""
    results, _ = client.scroll(
        collection_name=settings.collection_name("subcategories"),
        scroll_filter=Filter(
            must=[
                FieldCondition(key="parent_super_id", match=MatchValue(value=parent_super_id)),
            ],
            must_not=[
                FieldCondition(key="sub_id", match=MatchAny(any=exclude_sub_ids)),
            ],
        ),
        limit=50,
        with_vectors=with_vectors,
    )
    out = []
    for r in results:
        entry = {"point_id": r.id, **r.payload}
        if with_vectors and r.vector:
            entry["vectors"] = r.vector
        out.append(entry)
    return out


def get_segment_equivalents(
    client: QdrantClient,
    segment_vector: list[float],
    subcategory_id: int,
    exclude_platform: str,
    settings: Settings,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Find cross-platform equivalent segments — the EQUIVALENT_TO replacement."""
    results = client.search(
        collection_name=settings.collection_name("segments"),
        query_vector=NamedVector(name="bge", vector=segment_vector),
        query_filter=Filter(
            must=[
                FieldCondition(key="subcategory_id", match=MatchValue(value=subcategory_id)),
            ],
            must_not=[
                FieldCondition(key="platform", match=MatchValue(value=exclude_platform)),
            ],
        ),
        limit=max_results,
        score_threshold=settings.similarity_threshold_equivalent,
    )
    return [{"score": r.score, **r.payload} for r in results]
