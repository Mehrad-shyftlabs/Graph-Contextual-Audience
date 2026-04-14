"""Embed audience segments with BGE asymmetric dual encoder.

Provides:
- Document embedding with BGE doc prefix (for indexing)
- Query embedding with BGE query prefix (for search)
- Node2Vec training on a graph (build-only)
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from audience_targeting.models import Segment
from audience_targeting.settings import Settings


def load_model(settings: Settings | None = None) -> SentenceTransformer:
    """Load the BGE sentence-transformers model."""
    if settings is None:
        settings = Settings()
    return SentenceTransformer(settings.embedding_model)


def embed_documents(
    texts: list[str],
    model: SentenceTransformer,
    settings: Settings | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Embed document texts with BGE document prefix. Returns L2-normalized float32 vectors."""
    if settings is None:
        settings = Settings()
    prefixed = [settings.bge_doc_prefix + t for t in texts]
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
    settings: Settings | None = None,
) -> np.ndarray:
    """Embed a single query with BGE query prefix. Returns L2-normalized (1, 384) float32 array."""
    if settings is None:
        settings = Settings()
    prefixed = settings.bge_query_prefix + text
    embedding = model.encode([prefixed], normalize_embeddings=True)
    return np.array(embedding, dtype=np.float32)


def embed_segments(
    segments: list[Segment],
    model: SentenceTransformer,
    settings: Settings | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Embed all segments with BGE document prefix. Assigns embeddings to segment objects."""
    if settings is None:
        settings = Settings()
    texts = [s.embed_text for s in segments]
    print(f"Embedding {len(texts)} segments with {settings.embedding_model} (document prefix)...")

    embeddings = embed_documents(texts, model, settings, batch_size)

    for seg, emb in zip(segments, embeddings):
        seg.embedding = emb

    return embeddings


def train_node2vec(graph, settings: Settings | None = None) -> dict[str, np.ndarray]:
    """Train Node2Vec on the audience graph. Returns {node_id: embedding}.

    This is a build-only function — the graph is discarded after training.
    """
    from node2vec import Node2Vec

    if settings is None:
        settings = Settings()

    print(f"Training Node2Vec: dim={settings.node2vec_dim}, walks={settings.node2vec_num_walks}...")

    G_undirected = graph.to_undirected()

    node2vec = Node2Vec(
        G_undirected,
        dimensions=settings.node2vec_dim,
        walk_length=settings.node2vec_walk_length,
        num_walks=settings.node2vec_num_walks,
        p=settings.node2vec_p,
        q=settings.node2vec_q,
        workers=settings.node2vec_workers,
        quiet=True,
    )

    model = node2vec.fit(window=settings.node2vec_window, min_count=1)

    embeddings = {}
    for node in graph.nodes():
        if node in model.wv:
            embeddings[node] = model.wv[node].astype(np.float32)

    print(f"Node2Vec trained: {len(embeddings)} node embeddings")
    return embeddings
