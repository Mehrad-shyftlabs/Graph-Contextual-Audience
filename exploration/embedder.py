"""Embed audience segments and build FAISS index for similarity search.

v1: Symmetric all-MiniLM-L6-v2 encoding.
v2: Asymmetric BGE encoding with query/document prefixes + per-layer indices.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from data_loader import Segment


# ── Model loading ─────────────────────────────────────────────────────────


def load_model(version: str = "v2") -> SentenceTransformer:
    """Load the sentence-transformers model."""
    if version == "v2":
        return SentenceTransformer(config.EMBEDDING_MODEL_V2)
    return SentenceTransformer(config.EMBEDDING_MODEL)


# ── v1: Symmetric embedding ──────────────────────────────────────────────


def embed_segments(
    segments: list[Segment],
    model: SentenceTransformer | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Embed all segments and return L2-normalized embedding matrix (v1 symmetric)."""
    if model is None:
        model = load_model("v1")

    texts = [s.embed_text for s in segments]
    print(f"Embedding {len(texts)} segments with {config.EMBEDDING_MODEL}...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    for seg, emb in zip(segments, embeddings):
        seg.embedding = emb

    return np.array(embeddings, dtype=np.float32)


def embed_texts(
    texts: list[str],
    model: SentenceTransformer | None = None,
) -> np.ndarray:
    """Embed arbitrary text strings (v1 symmetric). Returns L2-normalized vectors."""
    if model is None:
        model = load_model("v1")

    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


# ── v2: Asymmetric BGE embedding ─────────────────────────────────────────


def embed_documents(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 256,
) -> np.ndarray:
    """Embed document texts with BGE document prefix (v2 asymmetric)."""
    prefixed = [config.BGE_DOC_PREFIX + t for t in texts]
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def embed_query(
    text: str,
    model: SentenceTransformer,
) -> np.ndarray:
    """Embed a single query with BGE query prefix (v2 asymmetric)."""
    prefixed = config.BGE_QUERY_PREFIX + text
    embedding = model.encode([prefixed], normalize_embeddings=True)
    return np.array(embedding, dtype=np.float32)


def embed_segments_v2(
    segments: list[Segment],
    model: SentenceTransformer,
    batch_size: int = 256,
) -> np.ndarray:
    """Embed all segments with BGE document prefix (v2). Assigns embeddings to segments."""
    texts = [s.embed_text for s in segments]
    print(f"Embedding {len(texts)} segments with {config.EMBEDDING_MODEL_V2} (document prefix)...")

    embeddings = embed_documents(texts, model, batch_size)

    for seg, emb in zip(segments, embeddings):
        seg.embedding = emb

    return embeddings


# ── FAISS index ───────────────────────────────────────────────────────────


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index (cosine similarity on normalized vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, {dim} dimensions")
    return index


def search_index(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    top_k: int = config.FAISS_TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Search the FAISS index. Returns (scores, indices)."""
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    scores, indices = index.search(query_embedding, top_k)
    return scores[0], indices[0]


# ── Persistence ───────────────────────────────────────────────────────────


def save_artifacts(
    embeddings: np.ndarray,
    index: faiss.IndexFlatIP,
    segment_ids: list[str],
    output_dir: Path = config.EMBEDDINGS_DIR,
    prefix: str = "raw",
) -> None:
    """Save embeddings, FAISS index, and ID mapping to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{prefix}_embeddings.npy", embeddings)
    faiss.write_index(index, str(output_dir / f"{prefix}_index.faiss"))

    with open(output_dir / f"{prefix}_id_mapping.json", "w") as f:
        json.dump(segment_ids, f)

    print(f"Saved artifacts to {output_dir}/ with prefix '{prefix}'")


def load_artifacts(
    output_dir: Path = config.EMBEDDINGS_DIR,
    prefix: str = "raw",
) -> tuple[np.ndarray, faiss.IndexFlatIP, list[str]]:
    """Load saved embeddings, FAISS index, and ID mapping."""
    embeddings = np.load(output_dir / f"{prefix}_embeddings.npy")
    index = faiss.read_index(str(output_dir / f"{prefix}_index.faiss"))

    with open(output_dir / f"{prefix}_id_mapping.json") as f:
        segment_ids = json.load(f)

    return embeddings, index, segment_ids


def save_layer_indices(
    layer0_centroids: np.ndarray,
    layer1_centroids: np.ndarray,
    segment_embeddings: np.ndarray,
    output_dir: Path = config.EMBEDDINGS_DIR,
    prefix: str = "v2",
) -> None:
    """Save per-layer FAISS indices and embeddings for v2."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Layer 0
    l0_index = build_faiss_index(layer0_centroids)
    faiss.write_index(l0_index, str(output_dir / f"{prefix}_layer0_index.faiss"))
    np.save(output_dir / f"{prefix}_layer0_centroids.npy", layer0_centroids)

    # Layer 1
    l1_index = build_faiss_index(layer1_centroids)
    faiss.write_index(l1_index, str(output_dir / f"{prefix}_layer1_index.faiss"))
    np.save(output_dir / f"{prefix}_layer1_centroids.npy", layer1_centroids)

    # Layer 2 (segment-level)
    l2_index = build_faiss_index(segment_embeddings)
    faiss.write_index(l2_index, str(output_dir / f"{prefix}_layer2_index.faiss"))
    np.save(output_dir / f"{prefix}_layer2_embeddings.npy", segment_embeddings)

    print(f"Saved v2 layer indices: L0={l0_index.ntotal}, L1={l1_index.ntotal}, L2={l2_index.ntotal}")


def load_layer_indices(
    output_dir: Path = config.EMBEDDINGS_DIR,
    prefix: str = "v2",
) -> tuple[faiss.IndexFlatIP, faiss.IndexFlatIP, faiss.IndexFlatIP,
           np.ndarray, np.ndarray, np.ndarray]:
    """Load per-layer FAISS indices and embeddings for v2.

    Returns (l0_index, l1_index, l2_index, l0_centroids, l1_centroids, l2_embeddings).
    """
    l0_index = faiss.read_index(str(output_dir / f"{prefix}_layer0_index.faiss"))
    l1_index = faiss.read_index(str(output_dir / f"{prefix}_layer1_index.faiss"))
    l2_index = faiss.read_index(str(output_dir / f"{prefix}_layer2_index.faiss"))

    l0_centroids = np.load(output_dir / f"{prefix}_layer0_centroids.npy")
    l1_centroids = np.load(output_dir / f"{prefix}_layer1_centroids.npy")
    l2_embeddings = np.load(output_dir / f"{prefix}_layer2_embeddings.npy")

    return l0_index, l1_index, l2_index, l0_centroids, l1_centroids, l2_embeddings


# ── Node2Vec ──────────────────────────────────────────────────────────────


def train_node2vec(graph) -> dict[str, np.ndarray]:
    """Train Node2Vec on the audience graph. Returns {node_id: embedding}."""
    from node2vec import Node2Vec

    print(f"Training Node2Vec: dim={config.NODE2VEC_DIM}, walks={config.NODE2VEC_NUM_WALKS}...")

    G_undirected = graph.to_undirected()

    node2vec = Node2Vec(
        G_undirected,
        dimensions=config.NODE2VEC_DIM,
        walk_length=config.NODE2VEC_WALK_LENGTH,
        num_walks=config.NODE2VEC_NUM_WALKS,
        p=config.NODE2VEC_P,
        q=config.NODE2VEC_Q,
        workers=config.NODE2VEC_WORKERS,
        quiet=True,
    )

    model = node2vec.fit(window=config.NODE2VEC_WINDOW, min_count=1)

    embeddings = {}
    for node in graph.nodes():
        if node in model.wv:
            embeddings[node] = model.wv[node].astype(np.float32)

    print(f"Node2Vec trained: {len(embeddings)} node embeddings")
    return embeddings


def save_node2vec(
    embeddings: dict[str, np.ndarray],
    output_dir: Path = config.EMBEDDINGS_DIR,
    prefix: str = "v2",
) -> None:
    """Save Node2Vec embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)

    node_ids = list(embeddings.keys())
    emb_matrix = np.array([embeddings[nid] for nid in node_ids], dtype=np.float32)

    np.save(output_dir / f"{prefix}_node2vec_embeddings.npy", emb_matrix)
    with open(output_dir / f"{prefix}_node2vec_id_mapping.json", "w") as f:
        json.dump(node_ids, f)

    print(f"Saved Node2Vec: {len(node_ids)} embeddings to {output_dir}/")


def load_node2vec(
    output_dir: Path = config.EMBEDDINGS_DIR,
    prefix: str = "v2",
) -> dict[str, np.ndarray]:
    """Load Node2Vec embeddings."""
    emb_matrix = np.load(output_dir / f"{prefix}_node2vec_embeddings.npy")
    with open(output_dir / f"{prefix}_node2vec_id_mapping.json") as f:
        node_ids = json.load(f)

    return {nid: emb_matrix[i] for i, nid in enumerate(node_ids)}


# ── Sanity check ──────────────────────────────────────────────────────────


def run_sanity_check(
    segments: list[Segment],
    index: faiss.IndexFlatIP,
    model: SentenceTransformer,
    version: str = "v2",
) -> None:
    """Run sanity-check queries against the index."""
    test_queries = [
        "luxury cars",
        "pet food",
        "fitness",
        "real estate",
        "sports fans",
        "sweet desserts",
    ]

    segment_ids = [s.id for s in segments]

    print(f"\n{'='*70}")
    print(f"SANITY CHECK ({version}): Top-5 results for test queries")
    print(f"{'='*70}")

    for query in test_queries:
        if version == "v2":
            query_emb = embed_query(query, model)
        else:
            query_emb = embed_texts([query], model)

        scores, indices = search_index(index, query_emb, top_k=5)

        print(f"\nQuery: \"{query}\"")
        for rank, (score, idx) in enumerate(zip(scores, indices), 1):
            if idx < 0:
                continue
            seg = segments[idx]
            print(
                f"  {rank}. [{seg.platform:10s}] {seg.name:40s} "
                f"(score: {score:.3f}, hierarchy: {' > '.join(seg.hierarchy[:3])})"
            )


# ── CLI ────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    from data_loader import load_all

    segments = load_all()
    segment_ids = [s.id for s in segments]

    # v2: BGE asymmetric embedding
    model = load_model("v2")
    embeddings = embed_segments_v2(segments, model)

    # Build index
    index = build_faiss_index(embeddings)

    # Save
    save_artifacts(embeddings, index, segment_ids, prefix="v2")

    # Sanity check
    run_sanity_check(segments, index, model, version="v2")
