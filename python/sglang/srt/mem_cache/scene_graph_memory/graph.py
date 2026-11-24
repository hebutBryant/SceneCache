"""Reproducible geometric contacts between original-coordinate pixel masks."""

from __future__ import annotations

import numpy as np

from .schema import ContactEdge, RegionNode

CONTACT_CONVENTIONS = {
    "pixel_cells": "closed_unit_pixel_cells_8_neighborhood_v1",
    "overlap_only": "foreground_pixel_centers_v1",
}


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    columns = np.flatnonzero(mask.any(axis=0))
    rows = np.flatnonzero(mask.any(axis=1))
    if not columns.size or not rows.size:
        return None
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def _intersection(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    if left >= right or top >= bottom:
        return None
    return slice(top, bottom), slice(left, right)


def _expanded_box(box, shape):
    height, width = shape
    left, top, right, bottom = box
    return max(0, left - 1), max(0, top - 1), min(width, right + 1), min(height, bottom + 1)


def _dilate8(mask, box):
    """Dilate only the first mask, once, within its one-pixel expanded bbox."""
    left, top, right, bottom = box
    local = mask[top:bottom, left:right]
    height, width = local.shape
    padded = np.pad(local, 1)
    dilated = np.zeros_like(local)
    for dy in range(3):
        for dx in range(3):
            dilated |= padded[dy : dy + height, dx : dx + width]
    return dilated


def mask_contact_kind(
    mask_a: np.ndarray, mask_b: np.ndarray, contact_mode: str = "pixel_cells"
) -> str | None:
    if contact_mode not in CONTACT_CONVENTIONS:
        raise ValueError(f"Unknown contact mode {contact_mode!r}")
    if mask_a.shape != mask_b.shape or mask_a.ndim != 2:
        raise ValueError("Contact masks must have the same 2D shape")
    if mask_a.dtype != np.bool_ or mask_b.dtype != np.bool_:
        raise ValueError("Contact masks must be bool arrays")
    a = mask_bbox(mask_a)
    b = mask_bbox(mask_b)
    if a is None or b is None:
        return None
    shared = _intersection(a, b)
    if shared is not None and np.any(mask_a[shared] & mask_b[shared]):
        return "overlap"
    if contact_mode == "overlap_only":
        return None
    expanded = _expanded_box(a, mask_a.shape)
    return "touch" if _dilated_intersects(mask_a, mask_b, b, expanded) else None


def _dilated_intersects(mask_a, mask_b, box_b, expanded):
    shared = _intersection(expanded, box_b)
    if shared is None:
        return False
    y, x = shared
    left, top = expanded[:2]
    dilated = _dilate8(mask_a, expanded)
    return bool(
        np.any(
            dilated[y.start - top : y.stop - top, x.start - left : x.stop - left]
            & mask_b[shared]
        )
    )


def build_contact_edges(
    regions: list[RegionNode], masks: dict[str, np.ndarray], contact_mode: str = "pixel_cells"
) -> list[ContactEdge]:
    if contact_mode not in CONTACT_CONVENTIONS:
        raise ValueError(f"Unknown contact mode {contact_mode!r}")
    ordered = sorted(regions, key=lambda region: region.region_id)
    edges: list[ContactEdge] = []
    for index, region_a in enumerate(ordered):
        mask_a = masks[region_a.region_id]
        expanded = _expanded_box(region_a.bbox_xyxy, mask_a.shape)
        dilated = None
        for region_b in ordered[index + 1 :]:
            mask_b = masks[region_b.region_id]
            shared = _intersection(region_a.bbox_xyxy, region_b.bbox_xyxy)
            kind = None
            if shared is not None and np.any(mask_a[shared] & mask_b[shared]):
                kind = "overlap"
            elif contact_mode == "pixel_cells":
                shared = _intersection(expanded, region_b.bbox_xyxy)
                if shared is not None:
                    if dilated is None:
                        dilated = _dilate8(mask_a, expanded)
                    y, x = shared
                    left, top = expanded[:2]
                    if np.any(
                        dilated[y.start - top : y.stop - top, x.start - left : x.stop - left]
                        & mask_b[shared]
                    ):
                        kind = "touch"
            if kind is not None:
                edges.append(
                    ContactEdge(region_a.region_id, region_b.region_id, kind, CONTACT_CONVENTIONS[contact_mode])
                )
    return edges
