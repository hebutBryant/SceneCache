"""Validate manual region selections and map logical positions to scheduler slots."""

from dataclasses import dataclass
from functools import lru_cache
import json
import math
import os
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .schema import image_sha256

MODE_ENV = "SGLANG_VISUAL_KV_MODE"
MEMORY_ENV = "SGLANG_VISUAL_KV_MEMORY"
SELECTION_ENV = "SGLANG_VISUAL_KV_SELECTION"
METRICS_ENV = "SGLANG_VISUAL_KV_METRICS"

REGION_MODES = {"region_reference", "region_pipeline", "region_pipeline_triton"}
MODES = REGION_MODES | {"pipelined_full_kv"}


@dataclass(frozen=True)
class SelectionPlan:
    memory_fingerprint: str
    seed_region_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    expand_hops: int
    source_visual_count: int
    source_visual_indices: tuple[int, ...]
    logical_positions: tuple[int, ...]
    compact_visual_start: int
    source_visual_start: int
    nonvisual_logical_positions: tuple[int, ...]

    @property
    def dropped(self) -> int:
        return self.source_visual_count - len(self.source_visual_indices)


def mode():
    return os.environ.get(MODE_ENV, "legacy")


@lru_cache(maxsize=1)
def descriptor(root: str):
    directory = Path(root)
    manifest = json.loads((directory / "manifest.json").read_text())
    submitted_size = tuple(manifest["geometry"]["steps"][1]["output_size"])
    with Image.open(directory / "image.png") as stored:
        image = stored.convert("RGB")
        if image_sha256(image) != manifest["image_hash"]:
            raise ValueError("The memory image does not match the manifest")
        image.thumbnail(submitted_size, Image.Resampling.LANCZOS)
        submitted_hash = image_sha256(image)
    return manifest, submitted_size, submitted_hash


def prepare_image(image, root: str):
    """Match the exact saved image or its exact pre-resize representation."""
    manifest, submitted_size, submitted_hash = descriptor(root)
    image = ImageOps.exif_transpose(image).convert("RGB")
    digest = image_sha256(image)
    if digest == manifest["image_hash"]:
        image.thumbnail(submitted_size, Image.Resampling.LANCZOS)
        if image_sha256(image) == submitted_hash:
            return image
    elif digest == submitted_hash:
        return image
    raise ValueError("Region backend requires the exact image in the selected Scene Graph memory")


@lru_cache(maxsize=32)
def _selection(root: str, selection_json: bytes):
    manifest, _, _ = descriptor(root)
    config = json.loads(selection_json)
    if not isinstance(config, dict):
        raise ValueError("Region selection must be a JSON object")
    by_id = {region["region_id"]: region for region in manifest["regions"]}
    raw_regions = config.get("region_ids", [])
    if not isinstance(raw_regions, list) or any(not isinstance(value, str) for value in raw_regions):
        raise ValueError("region_ids must be a JSON array of strings")
    ids = set(raw_regions)
    seed_ids = tuple(sorted(ids))
    if unknown := ids - by_id.keys():
        raise ValueError(f"Unknown regions {sorted(unknown)}")
    hops = config.get("expand_hops", 0)
    if type(hops) is not int or hops < 0:
        raise ValueError("expand_hops must be a nonnegative integer")
    count = len(manifest["tokens"])
    if "source_visual_indices" in config:
        if config.get("memory_fingerprint") != manifest["fingerprint"]:
            raise ValueError("Explicit visual selection requires the exact source memory fingerprint")
        if hops or any(config.get(option) for option in ("all_visual", "include_unassigned", "active_budget")):
            raise ValueError("Explicit visual selections already include all expanded and background rows")
        indices = config["source_visual_indices"]
        if (
            not isinstance(indices, list) or not indices
            or any(type(index) is not int or not 0 <= index < count for index in indices)
            or indices != sorted(set(indices))
        ):
            raise ValueError("Explicit visual indices must be a nonempty sorted set of source rows")
        return seed_ids, seed_ids, 0, tuple(indices)
    for _ in range(hops):
        expanded = ids | {
            neighbor
            for edge in manifest["edges"]
            for node, neighbor in ((edge["region_a"], edge["region_b"]), (edge["region_b"], edge["region_a"]))
            if node in ids
        }
        if expanded == ids:
            break
        ids = expanded
    active = {idx for name in ids for idx in by_id[name]["token_indices"]}
    if config.get("include_unassigned"):
        active.update(manifest["unassigned_token_indices"])
    if config.get("all_visual"):
        active.update(range(count))
    if not active or not active <= set(range(count)):
        raise ValueError("Region selection has no valid active visual tokens")
    budget = config.get("active_budget")
    if budget is not None:
        if type(budget) not in (float, int) or not 0 < budget <= 1:
            raise ValueError("active_budget must be in (0, 1]")
        max_rows = max(1, math.ceil(count * budget))
        if len(active) > max_rows:
            raise ValueError("Required selected regions exceed the active budget")
        # The benchmark may fill a manually chosen budget around its seed
        # regions. This is spatial context expansion, not query retrieval.
        tokens = manifest["tokens"]
        coords = [(token["grid_y"], token["grid_x"]) for token in tokens]
        centers = [coords[index] for index in active]

        def distance(index):
            y, x = coords[index]
            return min(abs(cy - y) + abs(cx - x) for cy, cx in centers), index

        active.update(sorted(set(range(count)) - active, key=distance)[:max_rows - len(active)])
    return seed_ids, tuple(sorted(ids)), hops, tuple(sorted(active))


def compact_inputs(item, token_ids, mrope, delta, root, path):
    """Keep a subsequence of slots; preserve positions and advance decode at original MRoPE."""
    manifest, _, _ = descriptor(root)
    seeds, ids, hops, active = _selection(root, Path(path).read_bytes())
    if len(item.offsets) != 1:
        raise ValueError("Region backend requires one image span")
    start, last = item.offsets[0]
    n_all = last - start + 1
    source_tokens = manifest["tokens"]
    if item.image_grid_thw.tolist() != [manifest["geometry"]["grid_thw"]] or n_all != len(source_tokens):
        raise ValueError("Visual grid or span differs from the selected source memory")
    source_start = source_tokens[0]["source_index"]
    source_prefix = manifest["source_prefix_token_ids"]
    if start != source_start or token_ids[:start].tolist() != source_prefix:
        raise ValueError("Region backend requires the source image-first prefix with no extra pre-image text")
    if mrope[:, start:last + 1].T.tolist() != [token["mrope_position"] for token in source_tokens]:
        raise ValueError("Visual MRoPE positions differ from the saved source memory")
    nonvisual = (*range(start), *range(last + 1, token_ids.numel()))
    logical = (*range(start), *(start + idx for idx in active), *range(last + 1, token_ids.numel()))
    plan = SelectionPlan(
        manifest["fingerprint"], seeds, ids, hops, n_all, active, logical, start, source_start, nonvisual,
    )
    index = torch.tensor(logical, dtype=torch.long)
    item.offsets = [(start, start + len(active) - 1)]
    item["visual_kv_selection"] = plan
    return token_ids.index_select(0, index), mrope.index_select(1, index), delta + plan.dropped
