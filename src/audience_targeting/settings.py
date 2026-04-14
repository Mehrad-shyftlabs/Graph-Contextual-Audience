"""Configuration via Pydantic BaseSettings — all values overridable via AT_-prefixed env vars."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central settings for the Audience Targeting system."""

    # ── Qdrant ────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_prefix: str = ""

    # ── Embedding ─────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    bge_query_prefix: str = "Represent this sentence for searching relevant audience categories: "
    bge_doc_prefix: str = "Represent this sentence for retrieval: "

    # ── Search (coarse-to-fine) ───────────────────────────────────────────
    layer0_top_k: int = 10
    layer1_top_k: int = 30
    layer2_top_k: int = 10
    rerank_text_weight: float = 0.7
    rerank_graph_weight: float = 0.3
    use_node2vec: bool = True

    # ── Similarity thresholds ─────────────────────────────────────────────
    similarity_threshold_equivalent: float = 0.85
    similarity_threshold_related: float = 0.65

    # ── Clustering (build-only) ───────────────────────────────────────────
    hdbscan_metric: str = "euclidean"
    hdbscan_cluster_selection: str = "eom"
    layer0_cluster_size: int = 30
    layer0_min_samples: int = 10
    layer1_cluster_size: int = 8
    layer1_min_samples: int = 3

    # ── Node2Vec (build-only) ─────────────────────────────────────────────
    node2vec_dim: int = 64
    node2vec_walk_length: int = 30
    node2vec_num_walks: int = 200
    node2vec_p: float = 1.0
    node2vec_q: float = 0.5
    node2vec_workers: int = 4
    node2vec_window: int = 10

    # ── LLM enrichment (build-only) ──────────────────────────────────────
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    enrichment_batch_size: int = 25

    # ── Data paths (build-only) ──────────────────────────────────────────
    data_dir: Path = Path("data")
    enriched_dir: Path = Path("data/enriched")

    # ── API ───────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # ── Platforms ─────────────────────────────────────────────────────────
    platforms: list[str] = [
        "meta",
        "tiktok",
        "snapchat",
        "yahoo_dsp",
        "ttd",
        "dv360",
    ]

    model_config = {"env_prefix": "AT_", "env_file": ".env", "env_file_encoding": "utf-8"}

    # ── Derived paths ─────────────────────────────────────────────────────

    @property
    def iab_csv(self) -> Path:
        return self.data_dir / "IAB_Categories_All_DSPs_Complete.csv"

    @property
    def social_csv(self) -> Path:
        return self.data_dir / "TiktokSnapMeta.csv"

    @property
    def ttd_apps_csv(self) -> Path:
        return self.data_dir / "ttd_top_1000_apps.csv"

    @property
    def meta_json_files(self) -> list[Path]:
        return sorted(self.data_dir.glob("meta_*.json"))

    @property
    def yahoo_json_files(self) -> list[Path]:
        return sorted(self.data_dir.glob("yahoo_*.json"))

    def collection_name(self, base: str) -> str:
        """Return a collection name with optional prefix for multi-tenant setups."""
        if self.qdrant_collection_prefix:
            return f"{self.qdrant_collection_prefix}_{base}"
        return base


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
