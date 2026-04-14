"""Tests for the embedding module."""

import numpy as np
import pytest

from audience_targeting.settings import Settings


@pytest.fixture
def settings():
    return Settings()


def test_embed_query_shape(settings):
    """Verify embed_query produces correct shape and normalization."""
    from audience_targeting.embedder import embed_query, load_model

    model = load_model(settings)
    result = embed_query("luxury SUV shoppers", model, settings)

    assert result.shape == (1, 384)
    assert result.dtype == np.float32
    # Check L2 normalization
    norm = np.linalg.norm(result)
    assert abs(norm - 1.0) < 1e-5


def test_embed_documents_shape(settings):
    """Verify embed_documents produces correct shape."""
    from audience_targeting.embedder import embed_documents, load_model

    model = load_model(settings)
    texts = ["luxury cars", "pet food", "fitness apps"]
    result = embed_documents(texts, model, settings)

    assert result.shape == (3, 384)
    assert result.dtype == np.float32
    # All rows should be L2-normalized
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_asymmetric_encoding_differs(settings):
    """Query and document encodings of the same text should differ (asymmetric)."""
    from audience_targeting.embedder import embed_documents, embed_query, load_model

    model = load_model(settings)
    text = "luxury SUV shoppers"

    query_emb = embed_query(text, model, settings).flatten()
    doc_emb = embed_documents([text], model, settings).flatten()

    # They should NOT be identical (asymmetric prefixes)
    assert not np.allclose(query_emb, doc_emb, atol=1e-3)
    # But they should be somewhat similar (same underlying meaning)
    sim = float(query_emb @ doc_emb)
    assert sim > 0.5
