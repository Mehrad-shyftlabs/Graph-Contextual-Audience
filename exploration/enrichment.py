"""Enrich audience segments with LLM-generated descriptions using OpenAI GPT-4o-mini."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import config
from data_loader import Segment


def _load_env():
    """Load environment variables from .env file."""
    env_path = config.ROOT_DIR / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


_load_env()

from openai import OpenAI

SYSTEM_PROMPT = """You are an ad-tech audience taxonomy expert. You deeply understand how digital advertising platforms categorize users into audience segments for targeting.

For each category below, generate a 1-2 sentence description of:
1. What audience this category represents (who are these people?)
2. What advertising intent it signals (what are they likely to buy or do?)

Be specific about the types of people, their behaviors, and purchase signals. Do not be generic."""

USER_PROMPT_TEMPLATE = """Platform: {platform}

Categories:
{categories}

Return a JSON array with exactly one object per category:
[{{"name": "original name", "description": "your description"}}, ...]

Return ONLY the JSON array, no other text."""


def enrich_segments(
    segments: list[Segment],
    batch_size: int = config.ENRICHMENT_BATCH_SIZE,
    resume: bool = True,
    max_concurrent: int = 20,
) -> list[Segment]:
    """Enrich segments with LLM-generated descriptions.

    Groups segments by platform + top-level category and sends batches to GPT-4o-mini.
    Uses concurrent requests for speed. Results are cached for resumability.
    """
    import concurrent.futures

    client = OpenAI()
    config.ENRICHED_DIR.mkdir(parents=True, exist_ok=True)

    # Group segments by platform + top-level hierarchy
    batches = _create_batches(segments, batch_size)
    total_batches = len(batches)

    # Separate cached vs pending
    pending: dict[str, list[Segment]] = {}
    skipped_count = 0

    for batch_key, batch_segments in batches.items():
        cache_path = config.ENRICHED_DIR / f"{batch_key}.json"
        if resume and cache_path.exists():
            cached = _load_cache(cache_path)
            _apply_descriptions(batch_segments, cached)
            skipped_count += len(batch_segments)
        else:
            pending[batch_key] = batch_segments

    print(f"Enrichment: {len(segments)} segments, {total_batches} batches "
          f"({skipped_count} cached, {len(pending)} to process)")

    if not pending:
        print("All batches already cached!")
        return segments

    # Process pending batches concurrently
    enriched_count = 0
    errors = 0

    def _process_batch(item):
        batch_key, batch_segments = item
        cache_path = config.ENRICHED_DIR / f"{batch_key}.json"
        descriptions = _call_llm(client, batch_segments)
        if descriptions:
            _save_cache(cache_path, descriptions)
            _apply_descriptions(batch_segments, descriptions)
            return len(descriptions)
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(_process_batch, item): item[0]
            for item in pending.items()
        }
        done = 0
        for future in concurrent.futures.as_completed(futures):
            done += 1
            try:
                count = future.result()
                enriched_count += count
            except Exception as e:
                errors += 1
                print(f"  ERROR: {e}")

            if done % 50 == 0 or done == len(futures):
                print(f"  Progress: {done}/{len(futures)} batches "
                      f"(enriched: {enriched_count}, errors: {errors})")

    print(f"Enrichment complete: {enriched_count} new, {skipped_count} from cache, {errors} errors")
    return segments


def _create_batches(
    segments: list[Segment],
    batch_size: int,
) -> dict[str, list[Segment]]:
    """Group segments into batches by platform + top-level category."""
    groups: dict[str, list[Segment]] = defaultdict(list)
    for seg in segments:
        top_cat = seg.hierarchy[0] if seg.hierarchy else "general"
        # Clean the key for filesystem safety
        key = f"{seg.platform}_{top_cat}".replace("/", "_").replace(" ", "_").replace("&", "and")
        key = "".join(c for c in key if c.isalnum() or c in "_-")
        groups[key].append(seg)

    # Split large groups into sub-batches
    batches: dict[str, list[Segment]] = {}
    for key, segs in groups.items():
        if len(segs) <= batch_size:
            batches[key] = segs
        else:
            for i in range(0, len(segs), batch_size):
                sub_key = f"{key}_part{i // batch_size}"
                batches[sub_key] = segs[i : i + batch_size]

    return batches


def _call_llm(
    client: OpenAI,
    segments: list[Segment],
) -> list[dict]:
    """Call GPT-4o-mini to generate descriptions for a batch of segments."""
    platform = segments[0].platform
    categories_text = "\n".join(
        f"{i+1}. {seg.name} (hierarchy: {' > '.join(seg.hierarchy)})"
        for i, seg in enumerate(segments)
    )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        platform=platform,
        categories=categories_text,
    )

    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4000,
        )

        content = response.choices[0].message.content.strip()
        # Parse JSON from response (handle markdown code blocks)
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]

        descriptions = json.loads(content)
        return descriptions

    except Exception as e:
        print(f"  ERROR in LLM call: {e}")
        return []


def _apply_descriptions(
    segments: list[Segment],
    descriptions: list[dict],
) -> None:
    """Apply LLM descriptions back to segments."""
    desc_map = {d["name"]: d["description"] for d in descriptions if "name" in d and "description" in d}
    for seg in segments:
        if seg.name in desc_map:
            seg.description = desc_map[seg.name]


def _save_cache(path: Path, descriptions: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump(descriptions, f, indent=2)


def _load_cache(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import load_all

    segments = load_all()
    enrich_segments(segments, resume=True)

    # Show some enriched examples
    enriched = [s for s in segments if s.description]
    print(f"\n{'='*60}")
    print(f"Enriched {len(enriched)}/{len(segments)} segments")
    print(f"{'='*60}")

    for s in enriched[:10]:
        print(f"\n[{s.platform}] {s.name}")
        print(f"  {s.description}")
