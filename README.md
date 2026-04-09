# AI-Powered Cross-Platform Audience Targeting

Match natural language client briefs to audience segments across 6 ad platforms using embeddings, hierarchical clustering, and a knowledge graph.

## What It Does

You type a client brief like:

> "We're launching a premium SUV campaign targeting affluent families who own luxury vehicles and show interest in travel."

The system returns matched audience segments from **Meta, TikTok, Snapchat, Yahoo DSP, The Trade Desk, and DV360** -- ranked by relevance, with cross-platform equivalents linked together.

## Architecture (v2)

```
Client Brief
    |
    v
Sentence Chunking (split into individual sentences)
    |
    v
BGE Asymmetric Encoder (query prefix)
    |
    v
Layer 0: FAISS search over ~53 super-categories
    |
    v
Layer 1: Filter ~427 sub-categories (children of matched Layer 0)
    |
    v
Layer 2: Expand to ~9,095 platform-specific segments via graph edges
    |
    v
Re-Rank: 70% text similarity + 30% Node2Vec graph similarity + neighbor boost
    |
    v
Results grouped by platform
```

No LLM calls at query time. All intelligence is baked into the embeddings and graph at build time.

## Quick Start

### 1. Install Dependencies

```bash
# Python 3.10+ required
pip install -e .
pip install node2vec
```

### 2. Download Data

The `data/` folder (~171 MB) is too large for Git. Download it from Google Drive:

**[Download data/ folder from Google Drive](https://drive.google.com/drive/folders/1uWwMyEgaut00_TExuwdQh9TR8k0z2Tzh?usp=drive_link)**

Place the downloaded `data/` folder in the project root so the structure looks like:

```
Graph-Contextual-Audience/
├── data/
│   ├── IAB_Categories_All_DSPs_Complete.csv
│   ├── TiktokSnapMeta.csv
│   ├── ttd_top_1000_apps.csv
│   ├── meta_*.json
│   ├── yahoo_*.json
│   ├── enriched/       # LLM-enriched segment descriptions
│   ├── embeddings/     # FAISS indices + Node2Vec embeddings
│   └── graphs/         # NetworkX GraphML
├── app.py
├── ...
```

**All artifacts (embeddings, indices, graph) are pre-built.** You do NOT need an OpenAI API key to run the demo.

### 3. Rebuild Artifacts (only if needed)

If you need to rebuild from raw data:

```bash
# Step 1: Load and enrich data (requires OPENAI_API_KEY for enrichment)
export OPENAI_API_KEY="your-key-here"
python -c "from enrichment import enrich_all; enrich_all()"

# Step 2: Build embeddings, clusters, and graph
python -c "
from data_loader import load_all
from embedder import load_model, embed_segments_v2, save_layer_indices, train_node2vec, save_node2vec
from clustering import cluster_two_level, save_clusters_v2
from graph_builder import build_graph_v2, save_graph

segments = load_all()
model = load_model('v2')
embeddings = embed_segments_v2(segments, model)

super_cats, sub_cats, l0_labels, l1_labels = cluster_two_level(embeddings, segments)
save_clusters_v2(super_cats, sub_cats, l0_labels, l1_labels)

graph = build_graph_v2(segments, super_cats, sub_cats, embeddings, l1_labels)
save_graph(graph, filename='audience_graph_v2.graphml')

save_layer_indices(super_cats, sub_cats, embeddings, l1_labels, model)

n2v = train_node2vec(graph)
save_node2vec(n2v)
"
```

### 4. Run the Demo

```bash
streamlit run app.py
```

Opens at [http://localhost:8501](http://localhost:8501).

### 5. Run Evaluation

```bash
python evaluate.py
```

Runs 15 short keyword tests + 7 long brief tests and prints recall/coverage metrics.

## Project Structure

```
.
├── app.py              # Streamlit web UI
├── config.py           # All configuration parameters
├── data_loader.py      # Load raw data from CSV/JSON
├── enrichment.py       # LLM enrichment of segment descriptions (offline)
├── embedder.py         # BGE encoder, FAISS indices, Node2Vec
├── clustering.py       # Two-level HDBSCAN clustering
├── graph_builder.py    # 3-layer knowledge graph construction
├── query.py            # Search engine (coarse-to-fine + re-ranking)
├── evaluate.py         # Test harness with keyword/brief test cases
├── visualize.py        # Interactive graph and chart visualizations
├── methodology.md      # Full methodology document
├── data/
│   ├── *.csv, *.json   # Raw platform data
│   ├── enriched/       # LLM-enriched segment JSONs
│   ├── embeddings/     # FAISS indices + embeddings + Node2Vec
│   └── graphs/         # NetworkX GraphML
└── notebooks/          # Jupyter exploration notebooks
```

## Key Files to Read

- **[methodology.md](methodology.md)** -- Full technical methodology (architecture, algorithms, evaluation)
- **[config.py](config.py)** -- All tunable parameters in one place
- **[query.py](query.py)** -- The search pipeline (start here to understand the core logic)

## Platforms Covered

| Platform | Segments | Source |
|----------|----------|--------|
| Meta | Interests, Behaviours, Demographics | `TiktokSnapMeta.csv` + `meta_*.json` |
| TikTok | Interest & behaviour categories | `TiktokSnapMeta.csv` |
| Snapchat | Lifestyle & interest segments | `TiktokSnapMeta.csv` |
| Yahoo DSP | IAB + 3rd party data providers | `IAB_Categories_All_DSPs_Complete.csv` + `yahoo_*.json` |
| The Trade Desk | IAB contextual + audience segments | `IAB_Categories_All_DSPs_Complete.csv` |
| DV360 | Google affinity & in-market audiences | `IAB_Categories_All_DSPs_Complete.csv` |

## Evaluation Results (v2)

- **Short keyword queries**: ~85% keyword recall, 100% platform coverage
- **Long brief queries**: ~86% keyword recall, 100% platform coverage
- 22 test cases total (15 short + 7 long briefs)

## Requirements

- Python 3.10+
- ~200 MB disk for data + artifacts
- No GPU needed (CPU inference is fast, <2s per query)
- OpenAI API key only needed for rebuilding enrichment (not for running)
