# AI-Powered Cross-Platform Audience Targeting

Match natural language client briefs to audience segments across 6 ad platforms using embeddings, hierarchical clustering, and vector search.

## What It Does

You send a client brief like:

> "We're launching a premium SUV campaign targeting affluent families who own luxury vehicles and show interest in travel."

The API returns matched audience segments from **Meta, TikTok, Snapchat, Yahoo DSP, The Trade Desk, and DV360** — ranked by relevance, with cross-platform equivalents and recommendations.

## Architecture

```
Client Brief
    │
    ▼
Sentence Chunking → BGE Asymmetric Encoder (384-dim)
    │
    ▼
Layer 0: Qdrant search over ~2 super-categories
    │
    ▼
Layer 1: Filter ~411 sub-categories (children of matched Layer 0)
    │
    ▼
Layer 2: Expand to ~9,095 platform-specific segments
    │
    ▼
Re-Rank: 70% text similarity + 30% Node2Vec graph similarity + cohesion boost
    │
    ▼
Threshold Filter: match / partial_match / filtered (per-platform, account-level overrides)
    │
    ▼
Results grouped by platform + recommendations (related / broader / narrower)
```

No LLM calls at query time. All intelligence is baked into the embeddings and Qdrant at build time.

## Quick Start

### Prerequisites

- Python 3.10+
- Docker + Docker Compose (for production deployment)

### Local Development

```bash
# Install with serve dependencies
pip install -e ".[serve]"

# Copy environment config
cp .env.example .env

# Run the build pipeline (populates Qdrant collections)
python -m audience_targeting.build_pipeline

# Start the API server
uvicorn audience_targeting.api:app --host 0.0.0.0 --port 8000

# Or run the Streamlit demo
streamlit run streamlit_app.py
```

### Docker Deployment

```bash
# 1. Initialize Qdrant collections (one-time)
docker compose --profile build run --rm build-pipeline

# 2. Start the API
docker compose up -d

# 3. Verify
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## API Endpoints

| Method | Route | Auth | Description |
|--------|-------|------|-------------|
| GET | `/health` | No | Liveness probe — always 200, check `status` field |
| GET | `/ready` | No | Readiness probe — 200 if Qdrant reachable, 503 otherwise |
| POST | `/v1/search` | Yes | Core search — returns segments ranked by relevance |
| GET | `/v1/supercategories` | Yes | List all Layer 0 super-categories |
| GET | `/v1/supercategories/{id}/subcategories` | Yes | List sub-categories under a super-category |
| GET | `/v1/segments/{id}` | Yes | Get full detail for a single segment |
| GET | `/v1/segments/{id}/equivalents` | Yes | Find cross-platform equivalent segments |
| GET | `/v1/platforms` | Yes | List configured platform names |
| GET | `/v1/stats` | Yes | Collection point counts and platform list |

### Search Request Example

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key-here" \
  -d '{
    "query": "luxury SUV shoppers interested in premium automotive brands",
    "platforms": ["meta", "tiktok"],
    "top_k": 5,
    "match_threshold": 0.75,
    "partial_match_threshold": 0.5
  }'
```

`match_threshold` and `partial_match_threshold` are optional account-level overrides. When omitted, server defaults apply.

### Request & Response Payloads

#### `POST /v1/search` — Request

```json
{
  "query": "luxury SUV shoppers interested in premium automotive brands",
  "platforms": ["meta", "tiktok"],
  "top_k": 5,
  "include_recommendations": true,
  "include_scope_options": true,
  "match_threshold": 0.75,
  "partial_match_threshold": 0.5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | — | Client brief or keyword query (1–5000 chars) |
| `platforms` | string[] | No | all | Filter to specific platforms (e.g. `["meta", "tiktok"]`) |
| `top_k` | int | No | 10 | Max results per platform (1–50) |
| `include_recommendations` | bool | No | true | Include related sub-category suggestions |
| `include_scope_options` | bool | No | true | Include broadening/narrowing options |
| `match_threshold` | float | No | server default | Account-level override for match threshold (0.0–1.0) |
| `partial_match_threshold` | float | No | server default | Account-level override for partial match threshold (0.0–1.0) |

#### `POST /v1/search` — Response

```json
{
  "query": "luxury SUV shoppers interested in premium automotive brands",
  "sentence_topics": {
    "luxury SUV shoppers interested in premium automotive brands": [
      "Automotive & Vehicles"
    ]
  },
  "matched_subcategories": [
    {
      "sub_id": 42,
      "name": "Luxury Vehicles",
      "score": 0.87,
      "platforms": ["meta", "tiktok", "snapchat"],
      "member_count": 15,
      "source_sentence": "luxury SUV shoppers interested in premium automotive brands"
    }
  ],
  "segments_by_platform": {
    "meta": [
      {
        "segment_id": "meta_1234",
        "name": "Luxury Vehicle Shoppers",
        "platform": "meta",
        "score": 0.92,
        "match_label": "match",
        "hierarchy": ["Automotive", "Luxury Vehicles"],
        "segment_type": "interest",
        "audience_size": 1500000,
        "description": "People interested in luxury vehicle brands"
      }
    ],
    "tiktok": [
      {
        "segment_id": "tt_567",
        "name": "Auto Enthusiasts",
        "platform": "tiktok",
        "score": 0.78,
        "match_label": "match",
        "hierarchy": ["Automotive"],
        "segment_type": "interest",
        "audience_size": null,
        "description": null
      }
    ]
  },
  "recommendations": [
    {
      "sub_id": 55,
      "name": "Premium Travel & Lifestyle",
      "relation": "related",
      "score": 0.72,
      "member_count": 10,
      "platforms": ["meta", "tiktok", "dv360"]
    }
  ],
  "broadening_options": [
    {
      "sub_id": 10,
      "name": "Automotive & Vehicles",
      "relation": "broader",
      "score": 0.68,
      "member_count": 45,
      "platforms": ["meta", "tiktok", "snapchat", "yahoo_dsp", "ttd", "dv360"]
    }
  ],
  "narrowing_options": [
    {
      "sub_id": 48,
      "name": "Electric SUVs",
      "relation": "narrower",
      "score": 0.65,
      "member_count": 5,
      "platforms": ["meta", "ttd"]
    }
  ],
  "metadata": {
    "total_segments": 2,
    "platforms_matched": 2,
    "sentences_processed": 1,
    "search_time_ms": 187.43
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Echo of the original query |
| `sentence_topics` | dict | Maps each query sentence to matched super-category names (explainability) |
| `matched_subcategories` | array | Layer 1 sub-categories that matched, with scores and source sentence |
| `segments_by_platform` | dict | Platform-keyed map of segments, sorted by descending score, filtered by thresholds |
| `segments_by_platform[].score` | float | Composite similarity (0–1): 70% text + 30% graph + cohesion boost |
| `segments_by_platform[].match_label` | string | `"match"` (≥ match threshold) or `"partial_match"` (≥ partial threshold) |
| `recommendations` | array | Related sub-categories from taxonomy (omitted if `include_recommendations: false`) |
| `broadening_options` | array | Parent-level generalizations (omitted if `include_scope_options: false`) |
| `narrowing_options` | array | More specific children (omitted if `include_scope_options: false`) |
| `metadata` | object | Timing, segment count, and platform count after filtering |

#### `GET /v1/segments/{id}` — Response

```json
{
  "segment_id": "meta_1234",
  "name": "Luxury Vehicle Shoppers",
  "platform": "meta",
  "hierarchy": ["Automotive", "Luxury Vehicles"],
  "segment_type": "interest",
  "audience_size": 1500000,
  "description": "People interested in luxury vehicle brands",
  "subcategory_id": 42,
  "super_category_id": 3,
  "parent_segment_id": "meta_100"
}
```

#### `GET /v1/supercategories` — Response

```json
[
  {
    "super_id": 3,
    "name": "Automotive & Vehicles",
    "subcategory_count": 8,
    "platforms": ["meta", "tiktok", "snapchat", "yahoo_dsp", "ttd", "dv360"],
    "member_count": 120
  }
]
```

#### `GET /v1/supercategories/{id}/subcategories` — Response

```json
[
  {
    "sub_id": 42,
    "name": "Luxury Vehicles",
    "parent_super_id": 3,
    "platforms": ["meta", "tiktok", "snapchat"],
    "member_count": 15
  }
]
```

#### `GET /v1/segments/{id}/equivalents` — Response

```json
[
  {
    "segment_id": "tt_567",
    "name": "Auto Enthusiasts",
    "platform": "tiktok",
    "score": 0.88,
    "match_label": "match",
    "hierarchy": ["Automotive"],
    "segment_type": "interest",
    "audience_size": null,
    "description": null
  }
]
```

#### `GET /v1/stats` — Response

```json
{
  "supercategories": 24,
  "subcategories": 411,
  "segments": 9095,
  "platforms": ["meta", "tiktok", "snapchat", "yahoo_dsp", "ttd", "dv360"]
}
```

#### `GET /health` — Response

```json
{
  "status": "ok",
  "qdrant_connected": true,
  "model_loaded": true
}
```

## Environment Variables

All prefixed with `AT_`. See [.env.example](.env.example) for the full list.

### Required for Production

| Variable | Description |
|----------|-------------|
| `AT_QDRANT_URL` | Qdrant server URL (default: `http://localhost:6333`) |
| `AT_API_KEY` | API key for `/v1/*` endpoints. Leave unset to disable auth. |
| `AT_CORS_ORIGINS` | JSON list of allowed origins, e.g. `["https://app.example.com"]`. Default: `["*"]` |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `AT_LOG_LEVEL` | `INFO` | Logging level |
| `AT_RATE_LIMIT` | `60/minute` | Rate limit for `/v1/search` |
| `WEB_CONCURRENCY` | `2` | Gunicorn worker count |
| `AT_DEFAULT_MATCH_THRESHOLD` | `0.7` | Server-wide match threshold |
| `AT_DEFAULT_PARTIAL_MATCH_THRESHOLD` | `0.5` | Server-wide partial match threshold |
| `AT_PLATFORM_MATCH_THRESHOLDS` | `{}` | Per-platform overrides as JSON |

## Health Checks

- **Liveness**: `GET /health` — always returns 200; `status` is `"ok"` or `"degraded"`
- **Readiness**: `GET /ready` — returns 200 when Qdrant is reachable, 503 otherwise

The API fails fast on startup if any of the 3 required Qdrant collections (`supercategories`, `subcategories`, `segments`) are missing.

## Logging

JSON structured logs to stdout:

```json
{"timestamp": "2026-04-15T10:37:19", "level": "INFO", "logger": "audience_targeting.api", "message": "request completed", "method": "POST", "path": "/v1/search", "status": 200, "duration_ms": 231.63, "request_id": "96bd9d15-f4e8-4317-a39e-121021e8204e"}
```

The `X-Request-ID` header is propagated (or generated) for distributed tracing.

## Security

- All `/v1/*` endpoints require `X-API-Key` header when `AT_API_KEY` is set
- `/health` and `/ready` are exempt from auth (for orchestrator probes)
- Run behind a reverse proxy (nginx, ALB) with TLS — the API serves plain HTTP
- Use a secrets manager for `AT_API_KEY` and `AT_QDRANT_API_KEY` in production
- Restrict `AT_CORS_ORIGINS` to trusted domains — never use `["*"]` in production
- Container runs as non-root user (`appuser`, UID 1000)

## Project Structure

```
.
├── src/audience_targeting/     # Core service
│   ├── api.py                  # FastAPI application (9 endpoints)
│   ├── api_models.py           # Pydantic request/response models
│   ├── settings.py             # Configuration (Pydantic BaseSettings)
│   ├── search_engine.py        # 3-layer coarse-to-fine search
│   ├── qdrant_store.py         # Qdrant operations (ingest + search)
│   ├── build_pipeline.py       # Offline build: cluster, embed, ingest
│   ├── clustering.py           # Two-level HDBSCAN
│   ├── embedder.py             # BGE embedding (asymmetric)
│   ├── data_loader.py          # Load raw data from CSV/JSON
│   ├── relationships.py        # Pre-compute taxonomy relationships
│   ├── logging_config.py       # JSON structured logging
│   └── models.py               # Domain models (Segment, SubCategory, etc.)
├── tests/                      # 57 unit tests + 24 integration tests
├── data/                       # Raw segment data (gitignored)
├── Dockerfile                  # Multi-stage: base, build, serve
├── docker-compose.yml          # Qdrant + API + build-pipeline
├── pyproject.toml              # Dependencies and project config
├── .env.example                # Environment variable template
├── streamlit_app.py            # Interactive demo UI (dev only)
└── exploration/                # Legacy scripts, notebooks, proposals
```

## Platforms Covered

| Platform | Segments | Source |
|----------|----------|--------|
| Meta | Interests, Behaviours, Demographics | `TiktokSnapMeta.csv` + `meta_*.json` |
| TikTok | Interest & behaviour categories | `TiktokSnapMeta.csv` |
| Snapchat | Lifestyle & interest segments | `TiktokSnapMeta.csv` |
| Yahoo DSP | IAB + 3rd party data providers | `IAB_Categories_All_DSPs_Complete.csv` + `yahoo_*.json` |
| The Trade Desk | IAB contextual + audience segments | `IAB_Categories_All_DSPs_Complete.csv` |
| DV360 | Google affinity & in-market audiences | `IAB_Categories_All_DSPs_Complete.csv` |

## Running Tests

```bash
# Unit tests only (no Qdrant required)
pytest tests/ -m "not integration" -v

# All tests (requires running Qdrant with populated data)
pytest tests/ -v
```

## Requirements

- Python 3.10+
- ~200 MB disk for data + Qdrant storage
- ~512 MB–1 GB RAM per gunicorn worker (BGE model + Qdrant client)
- No GPU needed (CPU inference, <250ms per query)
- OpenAI API key only needed for rebuilding enrichment (not for serving)
