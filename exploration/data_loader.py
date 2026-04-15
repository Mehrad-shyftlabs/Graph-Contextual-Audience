"""Load and normalize audience segments from all platform data sources."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config


@dataclass
class Segment:
    """Unified representation of an audience segment across all platforms."""

    id: str
    name: str
    platform: str  # meta, tiktok, snapchat, yahoo_dsp, ttd, dv360
    source_file: str
    hierarchy: list[str] = field(default_factory=list)
    segment_type: str = ""  # interest, behavior, iab_content, lifestyle, app, ...
    audience_size: int | None = None
    metadata: dict = field(default_factory=dict)
    description: str | None = None  # filled by enrichment
    embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def embed_text(self) -> str:
        """Text used for embedding: name + hierarchy context."""
        if self.description:
            return f"{self.name} | {' > '.join(self.hierarchy)} | {self.description}"
        if self.hierarchy:
            return f"{self.name} | {' > '.join(self.hierarchy)}"
        return self.name


# ── IAB CSV loader ─────────────────────────────────────────────────────────


def _normalize_platform_name(raw: str) -> str:
    mapping = {
        "Yahoo DSP": "yahoo_dsp",
        "The Trade Desk": "ttd",
        "DV360": "dv360",
    }
    return mapping.get(raw, raw.lower().replace(" ", "_"))


def load_iab_csv(path: Path = config.IAB_CSV) -> list[Segment]:
    """Load IAB categories from the DSP CSV.

    Builds parent-child hierarchy from Tier 1 / Tier 2 relationships.
    Filters out Sensitive/Exclusion rows and rows missing Category ID.
    """
    segments: list[Segment] = []
    # First pass: build a map of Category ID -> Category Name for hierarchy lookup
    id_to_name: dict[str, str] = {}
    rows: list[dict] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat_id = (row.get("Category ID") or "").strip()
            cat_name = (row.get("Category Name") or "").strip()
            if not cat_id or not cat_name:
                continue
            # Skip sensitive/exclusion categories
            support = (row.get("IAB Support Level") or "").strip()
            if "Sensitive" in support or "Exclusion" in support:
                continue
            id_to_name[cat_id] = cat_name
            rows.append(row)

    for row in rows:
        cat_id = row["Category ID"].strip()
        cat_name = row["Category Name"].strip()
        platform = _normalize_platform_name(row["DSP Platform"].strip())
        tier = (row.get("Tier") or "").strip()
        channel = (row.get("Channel") or "").strip()

        # Build hierarchy
        hierarchy = _build_iab_hierarchy(cat_id, cat_name, id_to_name, tier)

        # Determine segment type from tier
        segment_type = "iab_content"
        if "Topic" in tier:
            segment_type = "google_topic"
        elif "App" in tier:
            segment_type = "app_category"
        elif "Genre" in tier:
            segment_type = "genre"

        # Include channel in ID — same category on different channels are separate segments
        channel_slug = channel.replace(" ", "_").replace("/", "_").lower()
        seg = Segment(
            id=f"iab_{platform}_{cat_id}_{channel_slug}",
            name=cat_name,
            platform=platform,
            source_file=path.name,
            hierarchy=hierarchy,
            segment_type=segment_type,
            metadata={"category_id": cat_id, "channel": channel, "tier": tier},
        )
        segments.append(seg)

    return segments


def _build_iab_hierarchy(
    cat_id: str, cat_name: str, id_to_name: dict[str, str], tier: str
) -> list[str]:
    """Build hierarchy path from IAB category ID.

    IAB2-10 -> ["Automotive", "Electric Vehicle"]
    TOPIC-1 -> ["Arts & Entertainment"]
    """
    # For standard IAB tiers: parent is the prefix before the dash
    if "-" in cat_id and tier == "Tier 2":
        parent_id = cat_id.rsplit("-", 1)[0]
        parent_name = id_to_name.get(parent_id, parent_id)
        return [parent_name, cat_name]
    return [cat_name]


# ── TikTok / Snapchat / Meta CSV loader ───────────────────────────────────


def load_social_csv(path: Path = config.SOCIAL_CSV) -> list[Segment]:
    """Load TikTok, Snapchat, and Meta segments from TiktokSnapMeta.csv."""
    segments: list[Segment] = []

    platform_map = {
        "TikTok": "tiktok",
        "Snapchat": "snapchat",
        "Meta": "meta",
    }

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            raw_platform = (row.get("Platform") or "").strip()
            platform = platform_map.get(raw_platform, raw_platform.lower())
            category = (row.get("Category") or "").strip()
            sub_segment = (row.get("Sub-Segment") or "").strip()
            section = (row.get("Section") or "").strip()
            targeting_type = (row.get("Targeting Type") or "").strip()

            if not category:
                continue

            # Build name and hierarchy
            name = sub_segment if sub_segment else category
            hierarchy = [category, sub_segment] if sub_segment else [category]

            seg = Segment(
                id=f"social_{platform}_{i}",
                name=name,
                platform=platform,
                source_file=path.name,
                hierarchy=hierarchy,
                segment_type=targeting_type.lower() if targeting_type else "interest",
                metadata={"section": section, "targeting_type": targeting_type},
            )
            segments.append(seg)

    return segments


# ── Meta JSON loader ───────────────────────────────────────────────────────


def load_meta_json(paths: list[Path] | None = None) -> list[Segment]:
    """Load Meta audience segments from the vertical-specific JSON files.

    Each file has structure: {"data": [{"id", "name", "type", "path", "audience_size_*"}]}
    """
    if paths is None:
        paths = config.META_JSON_FILES

    segments: list[Segment] = []
    seen_ids: set[str] = set()

    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        items = data.get("data", data if isinstance(data, list) else [])
        vertical = path.stem.replace("meta_", "")  # e.g., "automotive"

        for item in items:
            raw_id = str(item["id"])
            # Deduplicate across JSON files (behaviors/demographics repeat)
            if raw_id in seen_ids:
                continue
            seen_ids.add(raw_id)

            name = item["name"]
            path_arr = item.get("path", [])
            seg_type = item.get("type", "interest")
            lower = item.get("audience_size_lower_bound")
            upper = item.get("audience_size_upper_bound")
            audience_size = (lower + upper) // 2 if lower and upper else None

            seg = Segment(
                id=f"meta_{raw_id}",
                name=name,
                platform="meta",
                source_file=path.name,
                hierarchy=path_arr if path_arr else [name],
                segment_type=seg_type,
                audience_size=audience_size,
                metadata={
                    "vertical": vertical,
                    "audience_lower": lower,
                    "audience_upper": upper,
                },
            )
            segments.append(seg)

    return segments


# ── Yahoo DSP JSON loader ─────────────────────────────────────────────────


def load_yahoo_json(paths: list[Path] | None = None) -> list[Segment]:
    """Load Yahoo DSP audience segments from the vertical-specific JSON files.

    Each file has structure: {"response": [{"id", "name", "hierarchy", "reachCount", ...}]}
    The hierarchy array goes from leaf toward root (reversed). We normalize it.
    """
    if paths is None:
        paths = config.YAHOO_JSON_FILES

    segments: list[Segment] = []
    seen_ids: set[int] = set()

    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        items = data.get("response", data if isinstance(data, list) else [])
        vertical = path.stem.replace("yahoo_", "")

        for item in items:
            raw_id = item["id"]
            if raw_id in seen_ids:
                continue
            seen_ids.add(raw_id)

            name = item["name"]
            raw_hierarchy = item.get("hierarchy", [])
            # Hierarchy is an array of {"name": ...} dicts, leaf-to-root order
            hierarchy_names = [
                h["name"] if isinstance(h, dict) else str(h) for h in raw_hierarchy
            ]
            # Reverse to get root-to-leaf, and drop generic top levels
            hierarchy_names.reverse()
            # Remove generic top-level nodes that don't add semantic value
            skip_prefixes = {"All", "3rd Party Data", "1st Party Data"}
            hierarchy_clean = [h for h in hierarchy_names if h not in skip_prefixes]
            # Prepend the segment name if it's not already in the hierarchy
            if hierarchy_clean and name not in hierarchy_clean:
                hierarchy_clean.append(name)
            elif not hierarchy_clean:
                hierarchy_clean = [name]

            reach = item.get("reachCount")

            seg = Segment(
                id=f"yahoo_{raw_id}",
                name=name,
                platform="yahoo_dsp",
                source_file=path.name,
                hierarchy=hierarchy_clean,
                segment_type="audience",
                audience_size=reach if reach and reach > 0 else None,
                metadata={
                    "vertical": vertical,
                    "status": item.get("status"),
                    "audience_type": item.get("audienceType"),
                    "created_at": item.get("createdAt"),
                },
            )
            segments.append(seg)

    return segments


# ── TTD Apps loader ────────────────────────────────────────────────────────


def load_ttd_apps(path: Path = config.TTD_APPS_CSV) -> list[Segment]:
    """Load The Trade Desk top apps as app-targeting segments."""
    segments: list[Segment] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            app = (row.get("App") or "").strip()
            category = (row.get("Category") or "").strip()
            rank = (row.get("Rank") or "").strip()
            notes = (row.get("Relevance / Notes") or "").strip()

            if not app or not category:
                continue

            # Category format: "Travel - OTA" -> hierarchy ["Travel", "OTA"]
            hierarchy = [part.strip() for part in category.split(" - ")]

            seg = Segment(
                id=f"ttd_app_{rank}",
                name=app,
                platform="ttd",
                source_file=path.name,
                hierarchy=hierarchy,
                segment_type="app",
                metadata={"rank": int(rank) if rank.isdigit() else None, "notes": notes, "category": category},
            )
            segments.append(seg)

    return segments


# ── Orchestrator ───────────────────────────────────────────────────────────


def load_all() -> list[Segment]:
    """Load all segments from all data sources with deduplication.

    Meta and Yahoo appear in both CSV (TiktokSnapMeta.csv / IAB CSV) and JSON files.
    JSON versions are richer (audience sizes, deeper hierarchy), so we prefer them.
    CSV entries for Meta are kept since they cover different targeting types
    (the CSV has behavioral/demographic segments not in the vertical JSONs).
    """
    # Load all sources
    iab_segments = load_iab_csv()
    social_segments = load_social_csv()
    meta_json_segments = load_meta_json()
    yahoo_json_segments = load_yahoo_json()
    ttd_app_segments = load_ttd_apps()

    all_segments = (
        iab_segments
        + social_segments
        + meta_json_segments
        + yahoo_json_segments
        + ttd_app_segments
    )

    return all_segments


def print_summary(segments: list[Segment]) -> None:
    """Print a summary of loaded segments."""
    from collections import Counter

    platform_counts = Counter(s.platform for s in segments)
    type_counts = Counter(s.segment_type for s in segments)
    source_counts = Counter(s.source_file for s in segments)

    print(f"\n{'='*60}")
    print(f"Total segments loaded: {len(segments)}")
    print(f"{'='*60}")

    print("\nBy platform:")
    for platform, count in sorted(platform_counts.items()):
        print(f"  {platform:15s} {count:6d}")

    print("\nBy segment type:")
    for stype, count in sorted(type_counts.items()):
        print(f"  {stype:20s} {count:6d}")

    print("\nBy source file:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source:45s} {count:6d}")

    # Hierarchy depth distribution
    depths = [len(s.hierarchy) for s in segments]
    print(f"\nHierarchy depth: min={min(depths)}, max={max(depths)}, avg={sum(depths)/len(depths):.1f}")

    # Segments with audience sizes
    with_size = sum(1 for s in segments if s.audience_size is not None)
    print(f"Segments with audience size: {with_size} ({with_size/len(segments)*100:.1f}%)")


if __name__ == "__main__":
    segments = load_all()
    print_summary(segments)

    # Show a few samples per platform
    from collections import defaultdict
    by_platform = defaultdict(list)
    for s in segments:
        by_platform[s.platform].append(s)

    print(f"\n{'='*60}")
    print("Sample segments per platform:")
    print(f"{'='*60}")
    for platform in sorted(by_platform):
        print(f"\n--- {platform} ---")
        for s in by_platform[platform][:3]:
            print(f"  [{s.id}] {s.name}")
            print(f"    hierarchy: {' > '.join(s.hierarchy)}")
            print(f"    type: {s.segment_type}, audience: {s.audience_size}")
