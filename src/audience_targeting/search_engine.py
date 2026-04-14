"""Qdrant-backed search engine — coarse-to-fine retrieval with re-ranking.

Replaces the NetworkX + FAISS based AudienceSearchEngineV2.
"""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from audience_targeting import qdrant_store
from audience_targeting.embedder import embed_query
from audience_targeting.models import (
    MatchedSubCategory,
    Recommendation,
    SearchResult,
    Segment,
    SubCategory,
    SuperCategory,
)
from audience_targeting.settings import Settings


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


# ── Cosine similarity helper ─────────────────────────────────────────────


def _cosine_sim(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(a @ b / (norm_a * norm_b))


# ── Search engine ─────────────────────────────────────────────────────────


class AudienceSearchEngine:
    """Qdrant-backed audience search engine with 3-layer coarse-to-fine retrieval."""

    def __init__(
        self,
        client: QdrantClient,
        model: SentenceTransformer,
        settings: Settings,
    ):
        self.client = client
        self.model = model
        self.settings = settings

    def search(
        self,
        query: str,
        platforms: list[str] | None = None,
        top_k_segments: int | None = None,
    ) -> SearchResult:
        """Execute coarse-to-fine search with sentence chunking.

        1. Chunk brief into sentences
        2. For each sentence: Layer 0 -> Layer 1 -> Layer 2
        3. Re-rank with Node2Vec + cohesion boost
        4. Aggregate across sentences
        5. Compute recommendations + broaden/narrow
        """
        top_k = top_k_segments or self.settings.layer2_top_k

        sentences = chunk_brief(query)
        all_candidates: dict[str, list[tuple[dict, float]]] = defaultdict(list)
        all_matched_subs: list[dict] = []
        sentence_topics: dict[str, list[str]] = {}
        subcategory_hit_counts: Counter = Counter()

        # Cache sub-category Node2Vec vectors to avoid re-fetching
        sub_n2v_cache: dict[int, list[float] | None] = {}

        for sentence in sentences:
            query_emb = embed_query(sentence, self.model, self.settings)
            query_vec = query_emb.flatten().tolist()

            # ── Layer 0: search super-categories ─────────────────────
            l0_results = qdrant_store.search_supercategories(
                self.client, query_vec, self.settings
            )

            matched_super_ids = [r["super_id"] for r in l0_results]
            sentence_topics[sentence] = [r["name"] for r in l0_results]

            if not matched_super_ids:
                continue

            # ── Layer 1: search sub-categories within matched supers ─
            fetch_n2v = self.settings.use_node2vec
            l1_results = qdrant_store.search_subcategories(
                self.client, query_vec, matched_super_ids, self.settings,
                with_vectors=fetch_n2v,
            )

            matched_sub_ids = [r["sub_id"] for r in l1_results]

            for r in l1_results:
                all_matched_subs.append({
                    "sub_id": r["sub_id"],
                    "name": r["name"],
                    "parent_super_id": r["parent_super_id"],
                    "score": r["score"],
                    "platforms": r.get("platforms", []),
                    "member_count": r.get("member_count", 0),
                    "source_sentence": sentence,
                })
                # Cache Node2Vec vector for this sub-category
                if fetch_n2v and r.get("vectors"):
                    sub_n2v_cache[r["sub_id"]] = r["vectors"].get("node2vec")

            if not matched_sub_ids:
                continue

            # ── Layer 2: search segments within matched sub-cats ──────
            l2_results = qdrant_store.search_segments(
                self.client, query_vec, matched_sub_ids, self.settings,
                platforms=platforms,
                top_k=top_k * 6,
                with_vectors=fetch_n2v,
            )

            for seg_result in l2_results:
                text_sim = seg_result["score"]
                seg_sub_id = seg_result.get("subcategory_id", -1)

                # Node2Vec re-ranking
                n2v_sim = 0.0
                if fetch_n2v and seg_result.get("vectors"):
                    seg_n2v = seg_result["vectors"].get("node2vec")
                    sub_n2v = sub_n2v_cache.get(seg_sub_id)
                    if seg_n2v and sub_n2v:
                        n2v_sim = _cosine_sim(seg_n2v, sub_n2v)

                score = (
                    self.settings.rerank_text_weight * text_sim
                    + self.settings.rerank_graph_weight * n2v_sim
                )

                platform = seg_result["platform"]
                all_candidates[platform].append((seg_result, score))
                subcategory_hit_counts[seg_sub_id] += 1

        # ── Cohesion boost (replaces neighbor boost) ─────────────────
        total_hits = sum(subcategory_hit_counts.values())
        if total_hits > 0:
            for platform, candidates in all_candidates.items():
                boosted = []
                for seg_result, score in candidates:
                    cohesion = subcategory_hit_counts[seg_result.get("subcategory_id", -1)] / total_hits
                    boosted.append((seg_result, score + 0.05 * cohesion))
                all_candidates[platform] = boosted

        # ── Deduplicate and sort per platform ────────────────────────
        segments_by_platform: dict[str, list[tuple[Segment, float]]] = {}
        for platform, candidates in all_candidates.items():
            seen: dict[str, tuple[dict, float]] = {}
            for seg_result, score in candidates:
                sid = seg_result["segment_id"]
                if sid not in seen or score > seen[sid][1]:
                    seen[sid] = (seg_result, score)

            sorted_segs = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:top_k]
            segments_by_platform[platform] = [
                (_to_segment(sr), score) for sr, score in sorted_segs
            ]

        # ── Deduplicate matched sub-categories ───────────────────────
        seen_subs: dict[int, dict] = {}
        for ms in all_matched_subs:
            sid = ms["sub_id"]
            if sid not in seen_subs or ms["score"] > seen_subs[sid]["score"]:
                seen_subs[sid] = ms
        deduped_subs = sorted(seen_subs.values(), key=lambda x: x["score"], reverse=True)

        matched_subcategories = [
            MatchedSubCategory(
                sub_category=SubCategory(
                    id=ms["sub_id"], name=ms["name"],
                    parent_id=ms["parent_super_id"],
                    platforms=set(ms.get("platforms", [])),
                    member_count=ms.get("member_count", 0),
                ),
                super_category=None,
                score=ms["score"],
                source_sentence=ms.get("source_sentence", ""),
            )
            for ms in deduped_subs
        ]

        # ── Recommendations via vector similarity ────────────────────
        matched_sub_ids = [ms["sub_id"] for ms in deduped_subs[:10]]
        recommendations = self._get_recommendations(deduped_subs[:10], matched_sub_ids)
        broadening = self._get_scope_options(deduped_subs[:10], matched_sub_ids, "broader")
        narrowing = self._get_scope_options(deduped_subs[:10], matched_sub_ids, "narrower")

        return SearchResult(
            query=query,
            matched_subcategories=matched_subcategories,
            segments_by_platform=segments_by_platform,
            recommendations=recommendations,
            broadening_options=broadening,
            narrowing_options=narrowing,
            sentence_topics=sentence_topics,
        )

    def _get_recommendations(
        self, matched_subs: list[dict], matched_sub_ids: list[int], max_recs: int = 5
    ) -> list[Recommendation]:
        """Find related sub-categories via vector similarity — replaces RELATED_TO edges."""
        all_recs: dict[int, dict] = {}

        for ms in matched_subs:
            results = qdrant_store.get_related_subcategories(
                self.client,
                centroid_vector=self._get_sub_centroid(ms["sub_id"]),
                exclude_sub_ids=matched_sub_ids + list(all_recs.keys()),
                exclude_parent_id=ms.get("parent_super_id"),
                settings=self.settings,
                max_results=max_recs,
            )
            for r in results:
                rid = r["sub_id"]
                if rid not in all_recs or r["score"] > all_recs[rid]["score"]:
                    all_recs[rid] = r

        sorted_recs = sorted(all_recs.values(), key=lambda x: x["score"], reverse=True)[:max_recs]
        return [
            Recommendation(
                sub_id=r["sub_id"], name=r["name"], relation="related",
                score=r["score"], member_count=r.get("member_count", 0),
                platforms=r.get("platforms", []),
            )
            for r in sorted_recs
        ]

    def _get_scope_options(
        self, matched_subs: list[dict], matched_sub_ids: list[int], direction: str, max_opts: int = 3
    ) -> list[Recommendation]:
        """Find broader or narrower sub-categories — replaces BROADER_THAN/NARROWER_THAN edges."""
        opts: dict[int, dict] = {}

        for ms in matched_subs:
            siblings = qdrant_store.get_siblings(
                self.client,
                parent_super_id=ms["parent_super_id"],
                exclude_sub_ids=matched_sub_ids,
                settings=self.settings,
                with_vectors=True,
            )

            ms_centroid = self._get_sub_centroid(ms["sub_id"])
            ms_count = ms.get("member_count", 0)

            for sib in siblings:
                sib_count = sib.get("member_count", 0)
                sib_id = sib["sub_id"]

                # Compute centroid similarity
                sib_centroid = sib.get("vectors", {}).get("bge", [])
                if not sib_centroid or not ms_centroid:
                    continue
                sim = _cosine_sim(ms_centroid, sib_centroid)
                if sim < 0.6:
                    continue

                if direction == "broader" and sib_count >= ms_count * 1.5:
                    if sib_id not in opts or sim > opts[sib_id].get("score", 0):
                        opts[sib_id] = {**sib, "score": sim}
                elif direction == "narrower" and ms_count > 0 and sib_count <= ms_count * 0.67:
                    if sib_id not in opts or sim > opts[sib_id].get("score", 0):
                        opts[sib_id] = {**sib, "score": sim}

        sorted_opts = sorted(opts.values(), key=lambda x: x["score"], reverse=True)[:max_opts]
        return [
            Recommendation(
                sub_id=o["sub_id"], name=o["name"], relation=direction,
                score=o["score"], member_count=o.get("member_count", 0),
                platforms=o.get("platforms", []),
            )
            for o in sorted_opts
        ]

    def _get_sub_centroid(self, sub_id: int) -> list[float]:
        """Retrieve a sub-category's BGE centroid vector from Qdrant."""
        results, _ = self.client.scroll(
            collection_name=self.settings.collection_name("subcategories"),
            scroll_filter=Filter(must=[
                FieldCondition(key="sub_id", match=MatchValue(value=sub_id)),
            ]),
            limit=1,
            with_vectors=True,
        )
        if results and results[0].vector:
            vec = results[0].vector
            if isinstance(vec, dict):
                return vec.get("bge", [])
            return vec
        return []


# Import needed for _get_sub_centroid
from qdrant_client.models import FieldCondition, Filter, MatchValue


# ── Helper ───────────────────────────────────────────────────────────────


def _to_segment(result: dict) -> Segment:
    """Convert a Qdrant result dict back into a Segment dataclass."""
    return Segment(
        id=result.get("segment_id", ""),
        name=result.get("name", ""),
        platform=result.get("platform", ""),
        source_file="",
        hierarchy=result.get("hierarchy", []),
        segment_type=result.get("segment_type", ""),
        audience_size=result.get("audience_size"),
        description=result.get("description"),
    )


# ── Factory ──────────────────────────────────────────────────────────────


def create_engine(settings: Settings | None = None) -> AudienceSearchEngine:
    """Create a search engine connected to Qdrant."""
    if settings is None:
        settings = Settings()

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    model = SentenceTransformer(settings.embedding_model)

    return AudienceSearchEngine(client=client, model=model, settings=settings)
