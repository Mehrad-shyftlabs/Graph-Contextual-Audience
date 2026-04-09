"""Search pipeline: embed query -> coarse-to-fine FAISS search -> graph re-ranking.

v1: embed query -> FAISS over group centroids -> graph traversal.
v2: sentence chunking -> Layer 0 -> Layer 1 -> Layer 2 -> Node2Vec re-rank -> aggregate.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import faiss
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from clustering import AudienceGroup, SubCategory, SuperCategory
from data_loader import Segment


# ── Dataclasses ───────────────────────────────────────────────────────────


@dataclass
class MatchedGroup:
    """A matched audience group with similarity score (v1 compat)."""
    group: AudienceGroup
    score: float
    segments: list[Segment] = field(default_factory=list)


@dataclass
class MatchedSubCategory:
    """A matched sub-category with similarity score (v2)."""
    sub_category: SubCategory
    super_category: SuperCategory | None
    score: float
    source_sentence: str = ""


@dataclass
class Recommendation:
    """A recommended related group."""
    group: AudienceGroup | SubCategory
    relation: str  # "related", "broader", "narrower", "sibling"
    score: float
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.group.name if hasattr(self.group, 'name') else str(self.group)


@dataclass
class SearchResult:
    """Full search result with matches, recommendations, and scope options."""
    query: str
    matched_groups: list[MatchedGroup] = field(default_factory=list)
    matched_subcategories: list[MatchedSubCategory] = field(default_factory=list)
    segments_by_platform: dict[str, list[tuple[Segment, float]]] = field(default_factory=dict)
    recommendations: list[Recommendation] = field(default_factory=list)
    broadening_options: list[Recommendation] = field(default_factory=list)
    narrowing_options: list[Recommendation] = field(default_factory=list)
    sentence_topics: dict[str, list[str]] = field(default_factory=dict)  # sentence -> matched super-cat names


# ── Sentence chunking ─────────────────────────────────────────────────────


def chunk_brief(text: str) -> list[str]:
    """Split a client brief into individual sentences for independent search.

    Short keyword queries pass through as a single chunk.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return [text.strip()]
    return sentences


# ── v2: Coarse-to-Fine Search Engine ──────────────────────────────────────


class AudienceSearchEngineV2:
    """v2 search engine: 3-layer coarse-to-fine with sentence chunking + graph re-ranking."""

    def __init__(
        self,
        segments: list[Segment],
        super_categories: list[SuperCategory],
        sub_categories: list[SubCategory],
        graph: nx.DiGraph,
        model: SentenceTransformer,
        l0_index: faiss.IndexFlatIP,
        l1_index: faiss.IndexFlatIP,
        l0_centroids: np.ndarray,
        l1_centroids: np.ndarray,
        segment_embeddings: np.ndarray,
        node2vec_embeddings: dict[str, np.ndarray] | None = None,
    ):
        self.segments = segments
        self.id_to_segment = {s.id: s for s in segments}
        self.super_categories = super_categories
        self.sub_categories = sub_categories
        self.id_to_super = {sc.id: sc for sc in super_categories}
        self.id_to_sub = {sub.id: sub for sub in sub_categories}
        self.graph = graph
        self.model = model
        self.l0_index = l0_index
        self.l1_index = l1_index
        self.l0_centroids = l0_centroids
        self.l1_centroids = l1_centroids
        self.segment_embeddings = segment_embeddings
        self.node2vec_embeddings = node2vec_embeddings or {}

        # Build segment ID -> embedding index mapping
        self.seg_id_to_idx = {s.id: i for i, s in enumerate(segments)}

    def search(
        self,
        query: str,
        platforms: list[str] | None = None,
        top_k_segments: int = config.LAYER2_TOP_K,
    ) -> SearchResult:
        """Execute coarse-to-fine search with sentence chunking.

        1. Chunk brief into sentences
        2. For each sentence: Layer 0 -> Layer 1 -> Layer 2
        3. Graph-aware re-ranking
        4. Aggregate across sentences
        """
        from embedder import embed_query

        sentences = chunk_brief(query)
        all_candidates: dict[str, list[tuple[Segment, float]]] = defaultdict(list)
        all_matched_subs: list[MatchedSubCategory] = []
        sentence_topics: dict[str, list[str]] = {}
        matched_seg_ids: set[str] = set()

        for sentence in sentences:
            query_emb = embed_query(sentence, self.model)

            # ── Layer 0: find top super-categories ────────────────────
            l0_scores, l0_indices = self.l0_index.search(query_emb, config.LAYER0_TOP_K)
            l0_scores, l0_indices = l0_scores[0], l0_indices[0]

            matched_super_names = []
            valid_sub_ids: set[int] = set()

            for score, idx in zip(l0_scores, l0_indices):
                if idx < 0 or score < 0.3:
                    continue
                sc = self.super_categories[idx]
                matched_super_names.append(sc.name)
                valid_sub_ids.update(sc.subcategory_ids)

            sentence_topics[sentence] = matched_super_names

            # ── Layer 1: search within children of matched Layer 0 ────
            l1_candidates: list[tuple[SubCategory, float]] = []

            for sub_id in valid_sub_ids:
                sub = self.id_to_sub.get(sub_id)
                if sub is None:
                    continue
                # Early skip if no children on requested platforms
                if platforms and not sub.platforms.intersection(set(platforms)):
                    continue
                sim = float(query_emb @ sub.centroid.reshape(-1))
                l1_candidates.append((sub, sim))

            l1_candidates.sort(key=lambda x: x[1], reverse=True)
            top_subcats = l1_candidates[:config.LAYER1_TOP_K]

            for sub, sub_score in top_subcats:
                sc = self.id_to_super.get(sub.parent_id)
                all_matched_subs.append(MatchedSubCategory(
                    sub_category=sub,
                    super_category=sc,
                    score=sub_score,
                    source_sentence=sentence,
                ))

            # ── Layer 2: expand to segments with platform filter ──────
            for sub, sub_score in top_subcats:
                for seg_id in sub.segment_ids:
                    seg = self.id_to_segment.get(seg_id)
                    if seg is None:
                        continue
                    if platforms and seg.platform not in platforms:
                        continue

                    seg_idx = self.seg_id_to_idx.get(seg_id)
                    if seg_idx is not None:
                        text_sim = float(query_emb @ self.segment_embeddings[seg_idx].reshape(-1))
                    else:
                        text_sim = sub_score

                    # Node2Vec re-ranking
                    n2v_sim = self._compute_node2vec_sim(seg_id, f"sub_{sub.id}")
                    neighbor_boost = 0.0  # computed in aggregation pass

                    score = (
                        config.RERANK_TEXT_WEIGHT * text_sim
                        + config.RERANK_GRAPH_WEIGHT * n2v_sim
                    )

                    all_candidates[seg.platform].append((seg, score))
                    matched_seg_ids.add(seg_id)

        # ── Neighbor boost pass ───────────────────────────────────────
        for platform, seg_scores in all_candidates.items():
            boosted = []
            for seg, score in seg_scores:
                boost = self._compute_neighbor_boost(seg.id, matched_seg_ids)
                boosted.append((seg, score + 0.05 * boost))
            all_candidates[platform] = boosted

        # ── Deduplicate and sort per platform ─────────────────────────
        segments_by_platform: dict[str, list[tuple[Segment, float]]] = {}
        for platform, seg_scores in all_candidates.items():
            seen: dict[str, tuple[Segment, float]] = {}
            for seg, score in seg_scores:
                if seg.id not in seen or score > seen[seg.id][1]:
                    seen[seg.id] = (seg, score)
            sorted_segs = sorted(seen.values(), key=lambda x: x[1], reverse=True)
            segments_by_platform[platform] = sorted_segs[:top_k_segments]

        # ── Deduplicate matched sub-categories ────────────────────────
        seen_subs: dict[int, MatchedSubCategory] = {}
        for ms in all_matched_subs:
            if ms.sub_category.id not in seen_subs or ms.score > seen_subs[ms.sub_category.id].score:
                seen_subs[ms.sub_category.id] = ms
        deduped_subs = sorted(seen_subs.values(), key=lambda x: x.score, reverse=True)

        # ── Get recommendations via graph ─────────────────────────────
        matched_sub_ids = {ms.sub_category.id for ms in deduped_subs[:10]}
        recommendations = self._get_recommendations(matched_sub_ids)
        broadening = self._get_scope_options(matched_sub_ids, "BROADER_THAN")
        narrowing = self._get_scope_options(matched_sub_ids, "NARROWER_THAN")

        # Build v1-compatible matched_groups from sub-categories
        matched_groups = self._subs_to_matched_groups(deduped_subs, segments_by_platform)

        return SearchResult(
            query=query,
            matched_groups=matched_groups,
            matched_subcategories=deduped_subs,
            segments_by_platform=segments_by_platform,
            recommendations=recommendations,
            broadening_options=broadening,
            narrowing_options=narrowing,
            sentence_topics=sentence_topics,
        )

    def _compute_node2vec_sim(self, seg_id: str, sub_node: str) -> float:
        """Compute Node2Vec cosine similarity between a segment and sub-category node."""
        if not self.node2vec_embeddings:
            return 0.0
        seg_emb = self.node2vec_embeddings.get(seg_id)
        sub_emb = self.node2vec_embeddings.get(sub_node)
        if seg_emb is None or sub_emb is None:
            return 0.0
        norm_a = np.linalg.norm(seg_emb)
        norm_b = np.linalg.norm(sub_emb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(seg_emb @ sub_emb / (norm_a * norm_b))

    def _compute_neighbor_boost(self, seg_id: str, matched_ids: set[str]) -> float:
        """Compute neighbor boost: fraction of graph neighbors in the result set."""
        if seg_id not in self.graph:
            return 0.0
        neighbors = set()
        for _, target, data in self.graph.edges(seg_id, data=True):
            if data.get("edge_type") in ("EQUIVALENT_TO", "MEMBER_OF", "IS_CHILD_OF"):
                neighbors.add(target)
        for source, _, data in self.graph.in_edges(seg_id, data=True):
            if data.get("edge_type") in ("EQUIVALENT_TO", "MEMBER_OF", "IS_CHILD_OF"):
                neighbors.add(source)
        if not neighbors:
            return 0.0
        return len(neighbors & matched_ids) / len(neighbors)

    def _get_recommendations(
        self, matched_sub_ids: set[int], max_recs: int = 5
    ) -> list[Recommendation]:
        """Get related sub-categories via RELATED_TO edges."""
        recs: dict[int, float] = {}

        for sub_id in list(matched_sub_ids)[:10]:
            sub_node = f"sub_{sub_id}"
            if sub_node not in self.graph:
                continue
            for _, neighbor, data in self.graph.edges(sub_node, data=True):
                if data.get("edge_type") != "RELATED_TO":
                    continue
                try:
                    neighbor_id = int(neighbor.replace("sub_", ""))
                except ValueError:
                    continue
                if neighbor_id in matched_sub_ids:
                    continue
                weight = data.get("weight", 0.5)
                if neighbor_id not in recs or weight > recs[neighbor_id]:
                    recs[neighbor_id] = weight

        results = []
        for sub_id, score in sorted(recs.items(), key=lambda x: x[1], reverse=True)[:max_recs]:
            sub = self.id_to_sub.get(sub_id)
            if sub:
                results.append(Recommendation(group=sub, relation="related", score=score, name=sub.name))
        return results

    def _get_scope_options(
        self, matched_sub_ids: set[int], edge_type: str, max_opts: int = 3
    ) -> list[Recommendation]:
        """Get broader or narrower sub-category options."""
        opts: dict[int, float] = {}

        for sub_id in list(matched_sub_ids)[:10]:
            sub_node = f"sub_{sub_id}"
            if sub_node not in self.graph:
                continue
            for _, neighbor, data in self.graph.edges(sub_node, data=True):
                if data.get("edge_type") != edge_type:
                    continue
                try:
                    neighbor_id = int(neighbor.replace("sub_", ""))
                except ValueError:
                    continue
                if neighbor_id in matched_sub_ids:
                    continue
                weight = data.get("weight", 0.5)
                if neighbor_id not in opts or weight > opts[neighbor_id]:
                    opts[neighbor_id] = weight

        relation = "broader" if edge_type == "BROADER_THAN" else "narrower"
        results = []
        for sub_id, score in sorted(opts.items(), key=lambda x: x[1], reverse=True)[:max_opts]:
            sub = self.id_to_sub.get(sub_id)
            if sub:
                results.append(Recommendation(group=sub, relation=relation, score=score, name=sub.name))
        return results

    def _subs_to_matched_groups(
        self,
        matched_subs: list[MatchedSubCategory],
        segments_by_platform: dict[str, list[tuple[Segment, float]]],
    ) -> list[MatchedGroup]:
        """Convert matched sub-categories to MatchedGroup for v1 compatibility."""
        groups = []
        for ms in matched_subs[:15]:
            sub = ms.sub_category
            # Create a lightweight AudienceGroup wrapper
            ag = AudienceGroup(
                id=sub.id,
                name=sub.name,
                segment_ids=sub.segment_ids,
                centroid=sub.centroid,
                platforms=sub.platforms,
                member_count=sub.member_count,
            )
            mg = MatchedGroup(group=ag, score=ms.score)
            # Attach segments that are in results
            all_result_seg_ids = set()
            for segs in segments_by_platform.values():
                for seg, _ in segs:
                    all_result_seg_ids.add(seg.id)
            mg.segments = [
                self.id_to_segment[sid]
                for sid in sub.segment_ids
                if sid in all_result_seg_ids and sid in self.id_to_segment
            ]
            groups.append(mg)
        return groups


# ── v1: Flat Search Engine (kept for comparison) ──────────────────────────


class AudienceSearchEngine:
    """v1 search engine: FAISS over group centroids + flat graph traversal."""

    def __init__(
        self,
        segments: list[Segment],
        groups: list[AudienceGroup],
        graph: nx.DiGraph,
        model: SentenceTransformer,
        group_index: faiss.IndexFlatIP | None = None,
        group_centroids: np.ndarray | None = None,
    ):
        self.segments = segments
        self.id_to_segment = {s.id: s for s in segments}
        self.groups = groups
        self.id_to_group = {g.id: g for g in groups}
        self.graph = graph
        self.model = model

        if group_centroids is not None:
            self.group_centroids = group_centroids
        else:
            self.group_centroids = np.array([g.centroid for g in groups], dtype=np.float32)

        if group_index is not None:
            self.group_index = group_index
        else:
            self.group_index = faiss.IndexFlatIP(self.group_centroids.shape[1])
            self.group_index.add(self.group_centroids)

    def search(
        self,
        query: str,
        platforms: list[str] | None = None,
        top_k_groups: int = config.FAISS_TOP_K,
        top_k_segments: int = 10,
    ) -> SearchResult:
        """Execute v1 flat search."""
        query_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.group_index.search(query_emb, top_k_groups)
        scores, indices = scores[0], indices[0]

        matched_groups: list[MatchedGroup] = []
        all_segments_scored: dict[str, list[tuple[Segment, float]]] = defaultdict(list)

        for score, idx in zip(scores, indices):
            if idx < 0 or score < 0.3:
                continue
            group = self.groups[idx]
            mg = MatchedGroup(group=group, score=float(score))

            for seg_id in group.segment_ids:
                seg = self.id_to_segment.get(seg_id)
                if seg is None:
                    continue
                if platforms and seg.platform not in platforms:
                    continue
                if seg.embedding is not None:
                    seg_score = float(query_emb @ seg.embedding.reshape(-1, 1))
                else:
                    seg_score = float(score)
                mg.segments.append(seg)
                all_segments_scored[seg.platform].append((seg, seg_score))

            matched_groups.append(mg)

        segments_by_platform: dict[str, list[tuple[Segment, float]]] = {}
        for platform, seg_scores in all_segments_scored.items():
            seen: dict[str, tuple[Segment, float]] = {}
            for seg, score in seg_scores:
                if seg.id not in seen or score > seen[seg.id][1]:
                    seen[seg.id] = (seg, score)
            sorted_segs = sorted(seen.values(), key=lambda x: x[1], reverse=True)
            segments_by_platform[platform] = sorted_segs[:top_k_segments]

        recommendations = self._get_recommendations(matched_groups)
        broadening = self._get_scope_options(matched_groups, "BROADER_THAN")
        narrowing = self._get_scope_options(matched_groups, "NARROWER_THAN")

        return SearchResult(
            query=query,
            matched_groups=matched_groups,
            segments_by_platform=segments_by_platform,
            recommendations=recommendations,
            broadening_options=broadening,
            narrowing_options=narrowing,
        )

    def _get_recommendations(self, matched_groups, max_recs=5):
        matched_ids = {mg.group.id for mg in matched_groups}
        recs: dict[int, float] = {}
        for mg in matched_groups[:5]:
            group_node = f"group_{mg.group.id}"
            if group_node not in self.graph:
                continue
            for _, neighbor, data in self.graph.edges(group_node, data=True):
                if data.get("edge_type") != "RELATED_TO":
                    continue
                try:
                    neighbor_id = int(neighbor.replace("group_", ""))
                except ValueError:
                    continue
                if neighbor_id in matched_ids:
                    continue
                weight = data.get("weight", 0.5)
                if neighbor_id not in recs or weight > recs[neighbor_id]:
                    recs[neighbor_id] = weight
        results = []
        for gid, score in sorted(recs.items(), key=lambda x: x[1], reverse=True)[:max_recs]:
            group = self.id_to_group.get(gid)
            if group:
                results.append(Recommendation(group=group, relation="related", score=score, name=group.name))
        return results

    def _get_scope_options(self, matched_groups, edge_type, max_opts=3):
        matched_ids = {mg.group.id for mg in matched_groups}
        opts: dict[int, float] = {}
        for mg in matched_groups[:5]:
            group_node = f"group_{mg.group.id}"
            if group_node not in self.graph:
                continue
            for _, neighbor, data in self.graph.edges(group_node, data=True):
                if data.get("edge_type") != edge_type:
                    continue
                try:
                    neighbor_id = int(neighbor.replace("group_", ""))
                except ValueError:
                    continue
                if neighbor_id in matched_ids:
                    continue
                weight = data.get("weight", 0.5)
                if neighbor_id not in opts or weight > opts[neighbor_id]:
                    opts[neighbor_id] = weight
        relation = "broader" if edge_type == "BROADER_THAN" else "narrower"
        results = []
        for gid, score in sorted(opts.items(), key=lambda x: x[1], reverse=True)[:max_opts]:
            group = self.id_to_group.get(gid)
            if group:
                results.append(Recommendation(group=group, relation=relation, score=score, name=group.name))
        return results


# ── Formatting ────────────────────────────────────────────────────────────


def format_result(result: SearchResult) -> str:
    """Format search result for display."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"Query: \"{result.query}\"")
    lines.append(f"{'='*70}")

    # Sentence topics (v2)
    if result.sentence_topics:
        lines.append(f"\n--- Detected Topics ---")
        for sentence, topics in result.sentence_topics.items():
            topic_str = ", ".join(topics[:3]) if topics else "no match"
            lines.append(f"  \"{sentence[:60]}...\" -> {topic_str}")

    # Matched groups / sub-categories
    lines.append(f"\n--- Matched Audience Groups ---")
    for i, mg in enumerate(result.matched_groups[:10], 1):
        platforms_str = ", ".join(sorted(mg.group.platforms))
        lines.append(
            f"  {i}. [{mg.group.name}] (score: {mg.score:.3f}) "
            f"-- {mg.group.member_count} segments, platforms: {platforms_str}"
        )

    # Segments by platform
    lines.append(f"\n--- Segments by Platform ---")
    for platform in sorted(result.segments_by_platform):
        segs = result.segments_by_platform[platform]
        lines.append(f"\n  {platform.upper()}:")
        for seg, score in segs[:5]:
            size_str = ""
            if seg.audience_size:
                if seg.audience_size >= 1_000_000:
                    size_str = f" | reach: {seg.audience_size/1_000_000:.1f}M"
                else:
                    size_str = f" | reach: {seg.audience_size:,}"
            hierarchy_str = " > ".join(seg.hierarchy[:3])
            lines.append(
                f"    - {seg.name} (score: {score:.3f}){size_str}"
                f"\n      [{seg.segment_type}] {hierarchy_str}"
            )

    # Recommendations
    if result.recommendations:
        lines.append(f"\n--- Also Consider ---")
        for rec in result.recommendations:
            lines.append(
                f"  - [{rec.name}] ({rec.relation}, similarity: {rec.score:.2f}) "
                f"-- {rec.group.member_count} segments"
            )

    if result.broadening_options:
        lines.append(f"\n--- Broaden Reach ---")
        for opt in result.broadening_options:
            lines.append(
                f"  - [{opt.name}] (similarity: {opt.score:.2f}) "
                f"-- {opt.group.member_count} segments"
            )

    if result.narrowing_options:
        lines.append(f"\n--- Narrow / More Specific ---")
        for opt in result.narrowing_options:
            lines.append(
                f"  - [{opt.name}] (similarity: {opt.score:.2f}) "
                f"-- {opt.group.member_count} segments"
            )

    return "\n".join(lines)


# ── Factory ───────────────────────────────────────────────────────────────


def create_engine(version: str = "v2") -> AudienceSearchEngineV2 | AudienceSearchEngine:
    """Create a search engine from saved artifacts."""
    from data_loader import load_all
    from embedder import load_model

    if version == "v2":
        return _create_engine_v2()
    return _create_engine_v1()


def _create_engine_v2() -> AudienceSearchEngineV2:
    """Create v2 engine from saved artifacts."""
    from data_loader import load_all
    from embedder import load_model, load_layer_indices, load_node2vec
    from clustering import load_clusters_v2
    from graph_builder import load_graph

    print("Loading v2 search engine components...")
    segments = load_all()

    # Apply enrichment descriptions
    from enrichment import _create_batches, _load_cache, _apply_descriptions
    batches = _create_batches(segments, config.ENRICHMENT_BATCH_SIZE)
    for batch_key, batch_segments in batches.items():
        cache_path = config.ENRICHED_DIR / f"{batch_key}.json"
        if cache_path.exists():
            cached = _load_cache(cache_path)
            _apply_descriptions(batch_segments, cached)

    super_cats, sub_cats, l0_labels, l1_labels = load_clusters_v2()
    l0_index, l1_index, l2_index, l0_centroids, l1_centroids, l2_embeddings = load_layer_indices()
    graph = load_graph(filename="audience_graph_v2.graphml")
    model = load_model("v2")

    # Assign embeddings back to segments
    for i, seg in enumerate(segments):
        seg.embedding = l2_embeddings[i]

    # Load Node2Vec if available
    try:
        node2vec_embs = load_node2vec()
        print(f"  Node2Vec: {len(node2vec_embs)} embeddings loaded")
    except FileNotFoundError:
        node2vec_embs = None
        print("  Node2Vec: not available (will skip graph re-ranking)")

    engine = AudienceSearchEngineV2(
        segments=segments,
        super_categories=super_cats,
        sub_categories=sub_cats,
        graph=graph,
        model=model,
        l0_index=l0_index,
        l1_index=l1_index,
        l0_centroids=l0_centroids,
        l1_centroids=l1_centroids,
        segment_embeddings=l2_embeddings,
        node2vec_embeddings=node2vec_embs,
    )

    print(f"v2 Engine ready: {len(segments)} segments, {len(super_cats)} super-cats, "
          f"{len(sub_cats)} sub-cats, {graph.number_of_nodes()} graph nodes")
    return engine


def _create_engine_v1(prefix: str = "enriched") -> AudienceSearchEngine:
    """Create v1 engine from saved artifacts."""
    from data_loader import load_all
    from embedder import load_artifacts, load_model
    from clustering import load_clusters
    from graph_builder import load_graph

    print("Loading v1 search engine components...")
    segments = load_all()

    if prefix == "enriched":
        from enrichment import _create_batches, _load_cache, _apply_descriptions
        batches = _create_batches(segments, config.ENRICHMENT_BATCH_SIZE)
        for batch_key, batch_segments in batches.items():
            cache_path = config.ENRICHED_DIR / f"{batch_key}.json"
            if cache_path.exists():
                cached = _load_cache(cache_path)
                _apply_descriptions(batch_segments, cached)

    embeddings, seg_index, segment_ids = load_artifacts(prefix=prefix)
    groups, labels, centroids = load_clusters(prefix=prefix)
    graph = load_graph()
    model = load_model("v1")

    id_to_emb_idx = {sid: i for i, sid in enumerate(segment_ids)}
    for seg in segments:
        idx = id_to_emb_idx.get(seg.id)
        if idx is not None:
            seg.embedding = embeddings[idx]

    engine = AudienceSearchEngine(
        segments=segments,
        groups=groups,
        graph=graph,
        model=model,
        group_centroids=centroids,
    )
    print(f"v1 Engine ready: {len(segments)} segments, {len(groups)} groups, "
          f"{graph.number_of_nodes()} graph nodes")
    return engine


# ── CLI ───────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    engine = create_engine("v2")

    test_queries = [
        "luxury SUV shoppers",
        "pet food buyers",
        "fitness app users",
        "first-time homebuyers",
        "basketball fans",
        # Long brief test
        ("We're launching a premium SUV campaign targeting affluent families. "
         "The ideal audience owns luxury vehicles, has high household income, "
         "and shows interest in family activities and travel."),
    ]

    for q in test_queries:
        result = engine.search(q)
        print(format_result(result))
