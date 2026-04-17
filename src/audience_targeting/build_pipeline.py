"""Offline build pipeline: load data -> enrich -> embed -> cluster -> ingest to Qdrant.

Usage:
    python -m audience_targeting.build_pipeline [--skip-node2vec] [--qdrant-url URL]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from qdrant_client import QdrantClient

from audience_targeting import qdrant_store
from audience_targeting.clustering import cluster_two_level
from audience_targeting.data_loader import load_all, print_summary
from audience_targeting.embedder import embed_segments, load_model, train_node2vec
from audience_targeting.enrichment import (
    apply_cached_descriptions,
    enrich_segments,
    print_coverage_report,
)
from audience_targeting.relationships import (
    compute_parent_segment_ids,
    compute_subcategory_map,
    compute_subcategory_relationships,
    compute_super_category_map,
)
from audience_targeting.settings import Settings

logger = logging.getLogger(__name__)


def build(
    settings: Settings,
    skip_node2vec: bool = False,
    run_enrichment: bool = False,
    blue_green: bool = True,
    delete_previous: bool = False,
) -> None:
    """Execute the full build pipeline.

    Parameters
    ----------
    blue_green
        When True (default), creates timestamped collections and atomically
        swaps aliases after a successful ingest — zero downtime for the API.
        When False, uses the legacy ``recreate_collection`` path (destructive).
    delete_previous
        When True *and* ``blue_green`` is True, deletes the collections that
        the aliases previously pointed at.  Default False (keep for rollback).
    """
    t0 = time.time()

    # Step 1: Load raw data
    print("\n[1/7] Loading segments...")
    segments = load_all(settings)
    print_summary(segments)

    # Step 2: Enrichment (apply cached or run LLM)
    print("\n[2/7] Applying enrichment descriptions...")
    if run_enrichment:
        enrich_segments(segments, settings, resume=True)
    else:
        apply_cached_descriptions(segments, settings)

    # Print enrichment coverage so the team can see what's missing.
    print_coverage_report(segments, settings)

    # Step 3: Embed with BGE
    print("\n[3/7] Embedding segments...")
    model = load_model(settings)
    embeddings = embed_segments(segments, model, settings)

    # Step 4: Two-level HDBSCAN clustering
    print("\n[4/7] Clustering...")
    super_cats, sub_cats, l0_labels, l1_labels = cluster_two_level(embeddings, segments, settings)

    # Step 5: Compute relationships
    print("\n[5/7] Computing relationships...")
    parent_segment_map = compute_parent_segment_ids(segments)
    subcategory_map = compute_subcategory_map(segments, l1_labels, sub_cats)
    super_category_map = compute_super_category_map(sub_cats)
    sub_relationships = compute_subcategory_relationships(
        sub_cats,
        similarity_threshold_related=settings.similarity_threshold_related,
        similarity_threshold_equivalent=settings.similarity_threshold_equivalent,
    )
    n_related = sum(len(r["related"]) for r in sub_relationships.values())
    n_broader = sum(len(r["broader"]) for r in sub_relationships.values())
    n_narrower = sum(len(r["narrower"]) for r in sub_relationships.values())
    print(f"  Parent segments: {len(parent_segment_map)}")
    print(f"  Subcategory assignments: {len(subcategory_map)}")
    print(f"  Pre-computed edges: {n_related} related, {n_broader} broader, {n_narrower} narrower")

    # Step 6: Node2Vec (optional)
    node2vec_embeddings: dict | None = None
    if not skip_node2vec:
        print("\n[6/7] Training Node2Vec (temporary graph)...")
        node2vec_embeddings = _train_node2vec_from_scratch(
            segments, super_cats, sub_cats, embeddings, l1_labels, settings
        )
    else:
        print("\n[6/7] Skipping Node2Vec (--skip-node2vec)")

    # Step 7: Ingest into Qdrant
    print("\n[7/7] Ingesting into Qdrant...")
    client = settings.create_qdrant_client()

    if blue_green:
        # Blue/green: create versioned collections, ingest, then swap aliases.
        mapping = qdrant_store.create_versioned_collections(client, settings)

        # Temporarily override collection_name to return versioned names.
        _original_collection_name = settings.collection_name
        def _versioned_collection_name(base: str) -> str:
            alias = _original_collection_name(base)
            return mapping.get(alias, alias)
        settings.collection_name = _versioned_collection_name  # type: ignore[assignment]

        try:
            qdrant_store.ingest_supercategories(client, super_cats, settings)
            qdrant_store.ingest_subcategories(
                client, sub_cats, node2vec_embeddings, settings, sub_relationships
            )
            qdrant_store.ingest_segments(
                client, segments, embeddings,
                subcategory_map, super_category_map, parent_segment_map,
                node2vec_embeddings, settings,
            )
        finally:
            settings.collection_name = _original_collection_name  # type: ignore[assignment]

        # Swap aliases atomically — API starts serving new data.
        qdrant_store.swap_aliases(client, mapping, settings, delete_previous=delete_previous)
        print(f"  Blue/green deploy complete. Aliases swapped: {mapping}")
    else:
        # Legacy destructive path.
        qdrant_store.create_collections(client, settings)
        qdrant_store.ingest_supercategories(client, super_cats, settings)
        qdrant_store.ingest_subcategories(
            client, sub_cats, node2vec_embeddings, settings, sub_relationships
        )
        qdrant_store.ingest_segments(
            client, segments, embeddings,
            subcategory_map, super_category_map, parent_segment_map,
            node2vec_embeddings, settings,
        )

    elapsed = time.time() - t0
    print(f"\nBuild complete in {elapsed:.1f}s")
    print(f"  Super-categories: {len(super_cats)}")
    print(f"  Sub-categories:   {len(sub_cats)}")
    print(f"  Segments:         {len(segments)}")
    print(f"  Node2Vec:         {'trained' if node2vec_embeddings else 'skipped'}")
    if blue_green:
        print(f"  Deploy mode:      blue/green (previous collections {'deleted' if delete_previous else 'kept for rollback'})")


def _train_node2vec_from_scratch(
    segments, super_cats, sub_cats, embeddings, l1_labels, settings
) -> dict | None:
    """Build a temporary NetworkX graph, train Node2Vec, discard the graph.

    Returns None if graph construction or training fails — the build
    pipeline will proceed with text-only re-ranking.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not installed — skipping Node2Vec")
        return None

    from audience_targeting.relationships import compute_parent_segment_ids

    try:
        G = nx.DiGraph()

        # Add Layer 0 nodes
        for sc in super_cats:
            G.add_node(f"super_{sc.id}", node_type="super_category")

        # Add Layer 1 nodes + CONTAINS edges
        for sub in sub_cats:
            G.add_node(f"sub_{sub.id}", node_type="sub_category")
            G.add_edge(f"super_{sub.parent_id}", f"sub_{sub.id}", edge_type="CONTAINS")

        # Add Layer 2 nodes + MEMBER_OF edges
        for i, seg in enumerate(segments):
            G.add_node(seg.id, node_type="segment")
            sub_id = int(l1_labels[i])
            if sub_id >= 0:
                G.add_edge(seg.id, f"sub_{sub_id}", edge_type="MEMBER_OF")

        # Add IS_CHILD_OF edges
        parent_map = compute_parent_segment_ids(segments)
        for child_id, parent_id in parent_map.items():
            if child_id in G and parent_id in G:
                G.add_edge(child_id, parent_id, edge_type="IS_CHILD_OF")

        # Add EQUIVALENT_TO edges (cross-platform, same sub-cat, high similarity)
        id_to_idx = {s.id: i for i, s in enumerate(segments)}
        for sub in sub_cats:
            if len(sub.segment_ids) < 2:
                continue
            from collections import defaultdict
            by_platform: dict[str, list[str]] = defaultdict(list)
            for sid in sub.segment_ids:
                seg = next((s for s in segments if s.id == sid), None)
                if seg:
                    by_platform[seg.platform].append(sid)

            platforms = list(by_platform.keys())
            for pi in range(len(platforms)):
                for pj in range(pi + 1, len(platforms)):
                    for sid_a in by_platform[platforms[pi]]:
                        idx_a = id_to_idx.get(sid_a)
                        if idx_a is None:
                            continue
                        for sid_b in by_platform[platforms[pj]]:
                            idx_b = id_to_idx.get(sid_b)
                            if idx_b is None:
                                continue
                            sim = float(embeddings[idx_a].astype(float) @ embeddings[idx_b].astype(float))
                            if sim >= settings.similarity_threshold_equivalent:
                                G.add_edge(sid_a, sid_b, edge_type="EQUIVALENT_TO")
                                G.add_edge(sid_b, sid_a, edge_type="EQUIVALENT_TO")

        print(f"  Temporary graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Train Node2Vec (returns empty dict on its own failure)
        node2vec_embeddings = train_node2vec(G, settings)

        # Graph is discarded when this function returns
        return node2vec_embeddings or None

    except MemoryError:
        logger.error(
            "Node2Vec graph construction ran out of memory (%d segments). "
            "Continuing without graph embeddings.",
            len(segments),
        )
        return None
    except Exception as exc:
        logger.error(
            "Node2Vec graph construction failed: %s. "
            "Continuing without graph embeddings.",
            exc,
        )
        return None


def main():
    parser = argparse.ArgumentParser(description="Build audience targeting data pipeline")
    parser.add_argument("--skip-node2vec", action="store_true", help="Skip Node2Vec training")
    parser.add_argument("--qdrant-url", type=str, help="Override Qdrant URL")
    parser.add_argument("--qdrant-path", type=str, help="Use local file-based Qdrant (no server needed)")
    parser.add_argument("--enrich", action="store_true", help="Run LLM enrichment (requires OPENAI_API_KEY)")
    parser.add_argument("--data-dir", type=str, help="Override data directory")
    parser.add_argument(
        "--no-blue-green", action="store_true",
        help="Disable blue/green deploy (use destructive recreate_collection instead)",
    )
    parser.add_argument(
        "--delete-previous", action="store_true",
        help="Delete the previous collection set after a successful blue/green swap",
    )
    args = parser.parse_args()

    settings = Settings()
    if args.qdrant_path:
        settings.qdrant_path = args.qdrant_path
    elif args.qdrant_url:
        settings.qdrant_url = args.qdrant_url
    if args.data_dir:
        from pathlib import Path
        settings.data_dir = Path(args.data_dir)
        settings.enriched_dir = Path(args.data_dir) / "enriched"

    logging.basicConfig(level=getattr(logging, settings.log_level), format="%(levelname)s %(name)s: %(message)s")
    build(
        settings,
        skip_node2vec=args.skip_node2vec,
        run_enrichment=args.enrich,
        blue_green=not args.no_blue_green,
        delete_previous=args.delete_previous,
    )


if __name__ == "__main__":
    main()
