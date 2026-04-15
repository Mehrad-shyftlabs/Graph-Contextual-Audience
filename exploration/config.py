"""Central configuration for the Audience Targeting PoC."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
ENRICHED_DIR = DATA_DIR / "enriched"
INDICES_DIR = DATA_DIR / "indices"
GRAPHS_DIR = DATA_DIR / "graphs"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# ── Data files ─────────────────────────────────────────────────────────────
IAB_CSV = DATA_DIR / "IAB_Categories_All_DSPs_Complete.csv"
SOCIAL_CSV = DATA_DIR / "TiktokSnapMeta.csv"
TTD_APPS_CSV = DATA_DIR / "ttd_top_1000_apps.csv"
META_JSON_FILES = sorted(DATA_DIR.glob("meta_*.json"))
YAHOO_JSON_FILES = sorted(DATA_DIR.glob("yahoo_*.json"))

# ── Embedding (v1) ────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ── Embedding (v2) ────────────────────────────────────────────────────────
EMBEDDING_MODEL_V2 = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM_V2 = 384
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant audience categories: "
BGE_DOC_PREFIX = "Represent this sentence for retrieval: "

# ── Clustering (v1) ───────────────────────────────────────────────────────
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 3
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION = "eom"

# ── Clustering (v2: two-level) ────────────────────────────────────────────
LAYER0_CLUSTER_SIZE = 30
LAYER0_MIN_SAMPLES = 10
LAYER1_CLUSTER_SIZE = 8
LAYER1_MIN_SAMPLES = 3

# ── Graph / similarity thresholds ──────────────────────────────────────────
SIMILARITY_THRESHOLD_EQUIVALENT = 0.85
SIMILARITY_THRESHOLD_RELATED = 0.65

# ── Search (v1) ───────────────────────────────────────────────────────────
FAISS_TOP_K = 20

# ── Search (v2: coarse-to-fine) ───────────────────────────────────────────
LAYER0_TOP_K = 10
LAYER1_TOP_K = 30
LAYER2_TOP_K = 10
RERANK_TEXT_WEIGHT = 0.7
RERANK_GRAPH_WEIGHT = 0.3

# ── Node2Vec (v2) ─────────────────────────────────────────────────────────
NODE2VEC_DIM = 64
NODE2VEC_WALK_LENGTH = 30
NODE2VEC_NUM_WALKS = 200
NODE2VEC_P = 1.0
NODE2VEC_Q = 0.5
NODE2VEC_WORKERS = 4
NODE2VEC_WINDOW = 10

# ── LLM enrichment ────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o-mini"
ENRICHMENT_BATCH_SIZE = 25

# ── Platform canonical names ───────────────────────────────────────────────
PLATFORMS = [
    "meta",
    "tiktok",
    "snapchat",
    "yahoo_dsp",
    "ttd",
    "dv360",
]
