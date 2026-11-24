"""Map display-oriented region masks onto Qwen's post-merge LLM sequence."""

from __future__ import annotations

from operator import index as integer_index
from typing import Iterable

import numpy as np

from .graph import build_contact_edges, mask_bbox
from .schema import (
    ImageGeometry,
    ImageMemory,
    MaskInput,
    RegionNode,
    SourceKVMetadata,
    TokenFootprint,
    image_sha256,
    json_bytes,
    mask_sha256,
)


def build_token_provenance(
    geometry: ImageGeometry,
    source_indices: Iterable[int],
    mrope_positions: Iterable[Iterable[int]],
) -> list[TokenFootprint]:
    geometry.validate()
    source = tuple(map(integer_index, source_indices))
    positions = tuple(tuple(map(integer_index, position)) for position in mrope_positions)
    rows, cols = geometry.token_grid
    count = rows * cols
    if len(source) != count or len(positions) != count:
        raise ValueError(f"Expected {count} post-merge positions and source sequence indices")
    if source[0] < 0 or source != tuple(range(source[0], source[0] + count)):
        raise ValueError("Expected a complete and contiguous single-image source span")
    if any(len(position) != 3 for position in positions):
        raise ValueError("Each image token must have temporal, height and width MRoPE positions")
    original_width, original_height = geometry.original_size
    processed_width, processed_height = geometry.processed_size
    step = geometry.patch_size * geometry.spatial_merge_size
    tokens: list[TokenFootprint] = []
    for index in range(count):
        row, col = divmod(index, cols)
        x, y = col * step, row * step
        x_end, y_end = x + step, y + step
        tokens.append(
            TokenFootprint(
                visual_index=index,
                source_index=source[index],
                grid_y=row,
                grid_x=col,
                processed_xyxy=(x, y, x_end, y_end),
                original_xyxy=(
                    x * original_width / processed_width,
                    y * original_height / processed_height,
                    x_end * original_width / processed_width,
                    y_end * original_height / processed_height,
                ),
                mrope_position=positions[index],
            )
        )
    return tokens


def _mask_coverage_numerators(mask: np.ndarray, geometry: ImageGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Exact area intersections of bool unit pixels and inverse-scaled rectangles.

    Keep integer numerators with common denominator processed_width *
    processed_height. This avoids losing small intersections by rounding a
    token footprint or sampling only its center.
    """
    width, height = geometry.original_size
    processed_width, processed_height = geometry.processed_size
    if width * height * processed_width * processed_height > np.iinfo(np.int64).max:
        raise ValueError("Image geometry exceeds exact int64 coverage range")
    step = geometry.patch_size * geometry.spatial_merge_size
    rows, cols = geometry.token_grid
    x_numerator = np.arange(cols + 1, dtype=np.int64) * step * width
    y_numerator = np.arange(rows + 1, dtype=np.int64) * step * height
    ix, rx = np.divmod(x_numerator, processed_width)
    iy, ry = np.divmod(y_numerator, processed_height)
    nx = np.minimum(ix + 1, width)
    ny = np.minimum(iy + 1, height)

    integral = np.zeros((height + 1, width + 1), dtype=np.int64)
    np.cumsum(mask, axis=0, dtype=np.int64, out=integral[1:, 1:])
    np.cumsum(integral[1:, 1:], axis=1, out=integral[1:, 1:])

    base = integral[np.ix_(iy, ix)]
    right = integral[np.ix_(iy, nx)] - base
    bottom = integral[np.ix_(ny, ix)] - base
    corner = integral[np.ix_(ny, nx)] - (base + right + bottom)
    area = (
        base * (processed_width * processed_height)
        + right * rx[None, :] * processed_height
        + bottom * ry[:, None] * processed_width
        + corner * ry[:, None] * rx[None, :]
    )
    intersections = np.diff(np.diff(area, axis=0), axis=1)
    footprint_areas = np.multiply.outer(np.diff(y_numerator), np.diff(x_numerator))
    return intersections.ravel(), footprint_areas.ravel()


def mask_token_membership(
    mask: np.ndarray, geometry: ImageGeometry, overlap_threshold: float = 0.0
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("mask-token overlap threshold must be between zero and one")
    geometry.validate()
    if mask.dtype != np.bool_ or mask.shape != tuple(reversed(geometry.original_size)):
        raise ValueError("Expected an original-coordinate bool mask")
    overlaps, areas = _mask_coverage_numerators(mask, geometry)
    weights = overlaps / areas
    selected = np.flatnonzero((overlaps > 0) & (weights >= overlap_threshold))
    return tuple(selected.tolist()), tuple(weights[selected].tolist())


def assemble_memory(
    image,
    geometry: ImageGeometry,
    source_indices: Iterable[int],
    mrope_positions: Iterable[Iterable[int]],
    masks: list[MaskInput],
    source_kv,
    kv_metadata: SourceKVMetadata,
    fingerprints: dict,
    config: dict,
    *,
    overlap_threshold: float = 0.0,
    contact_mode: str = "pixel_cells",
    segmentation: dict | None = None,
) -> ImageMemory:
    """Index a complete source bank without re-encoding regions or duplicating K/V."""
    tokens = build_token_provenance(geometry, source_indices, mrope_positions)
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("mask-token overlap threshold must be between zero and one")
    if image.mode != "RGB" or image.size != tuple(geometry.original_size):
        raise ValueError("Image must be the normalized RGB image used by both SAM3 and Qwen")
    shape = tuple(reversed(geometry.original_size))
    candidates = []
    filtered = []
    for index, supplied in enumerate(masks):
        if not isinstance(supplied.mask, np.ndarray) or supplied.mask.dtype != np.bool_ or supplied.mask.shape != shape:
            raise ValueError(f"Mask {index} must be a bool array with original (height, width) shape")
        box = mask_bbox(supplied.mask)
        if box is None:
            filtered.append({"input_index": index, "label": supplied.label, "reason": "empty_mask"})
            continue
        mask = supplied.mask.copy(order="C")
        digest = mask_sha256(mask)
        order = json_bytes([supplied.label_source, supplied.score, supplied.metadata])
        candidates.append(((supplied.label or "", box, digest, order), supplied, mask))

    regions: list[RegionNode] = []
    region_masks = {}
    assigned = set()
    for label_and_box, supplied, mask in sorted(candidates, key=lambda candidate: candidate[0]):
        _, box, digest, _ = label_and_box
        region_id = f"r{len(regions) + 1:04d}"
        indices, weights = mask_token_membership(mask, geometry, overlap_threshold)
        assigned.update(indices)
        regions.append(
            RegionNode(
                region_id=region_id,
                mask_ref=f"masks/{region_id}.png",
                bbox_xyxy=box,
                area=int(np.count_nonzero(mask)),
                token_indices=indices,
                token_weights=weights,
                mask_sha256=digest,
                score=supplied.score,
                label=supplied.label,
                label_source=supplied.label_source,
                metadata=supplied.metadata,
            )
        )
        region_masks[region_id] = mask

    memory = ImageMemory(
        image=image.copy(),
        image_hash=image_sha256(image),
        geometry=geometry,
        tokens=tokens,
        regions=regions,
        edges=build_contact_edges(regions, region_masks, contact_mode),
        masks=region_masks,
        source_kv=source_kv,
        kv_metadata=kv_metadata,
        fingerprints=fingerprints,
        config={**config, "mask_token_overlap_threshold": overlap_threshold, "contact_mode": contact_mode},
        segmentation={
            **(segmentation or {}),
            "mapping_input_masks": len(masks),
            "mapping_retained_masks": len(regions),
            "mapping_filtered_masks": filtered,
        },
        unassigned_token_indices=tuple(sorted(set(range(len(tokens))) - assigned)),
    )
    memory.validate()
    return memory
