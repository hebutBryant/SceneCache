"""CPU representation of an image, its spatial regions, and one source prefill."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import numpy as np
    import torch
    from PIL import Image


SCHEMA_VERSION = 1
TOKEN_MAPPING_VERSION = "qwen_single_image_row_major_resize_v1"
BANK_REF = "source_visual_kv"


def json_bytes(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def content_fingerprint(value) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def image_sha256(image: Image.Image) -> str:
    """Hash normalized RGB pixels plus dimensions. Paths and file metadata are excluded."""
    digest = hashlib.sha256(json_bytes([image.mode, *image.size]))
    digest.update(image.tobytes())
    return digest.hexdigest()


def mask_sha256(mask: np.ndarray) -> str:
    import numpy as np

    digest = hashlib.sha256(json_bytes(mask.shape))
    # Boolean arrays can use 0xff for true after PNG decoding. Hash the values.
    digest.update(np.packbits(mask, bitorder="big").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ImageGeometry:
    """Measured resize-only geometry, in display-oriented pixel coordinates.

    Size tuples are (width, height). The patch grid is (temporal, height, width)
    before spatial merge. The current schema accepts one temporal grid cell
    and an extent occupied entirely by the image.
    """

    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    grid_thw: tuple[int, int, int]
    patch_size: int
    spatial_merge_size: int
    steps: list[dict] = field(default_factory=list)

    def validate(self) -> None:
        if len(self.original_size) != 2 or len(self.processed_size) != 2:
            raise ValueError("Geometry sizes must be (width, height)")
        sizes = (*self.original_size, *self.processed_size)
        if any(not isinstance(size, int) or size < 1 for size in sizes):
            raise ValueError("Geometry sizes must be positive integers")
        if len(self.grid_thw) != 3:
            raise ValueError("Expected an image grid shaped (temporal, height, width)")
        temporal, height, width = self.grid_thw
        if temporal != 1 or any(
            not isinstance(size, int) or size < 1 for size in (height, width)
        ):
            raise ValueError("Expected one temporal cell and a positive image grid")
        if not isinstance(self.patch_size, int) or self.patch_size < 1:
            raise ValueError("Expected a positive scalar patch size")
        merge = self.spatial_merge_size
        if not isinstance(merge, int) or merge < 1 or width % merge or height % merge:
            raise ValueError("Spatial merge must divide both patch grid dimensions")
        if tuple(self.processed_size) != (width * self.patch_size, height * self.patch_size):
            raise ValueError("Processed image dimensions differ from patch grid extent")
        for step in self.steps:
            if step.get("operation") in {"crop", "pad", "padding", "resize_with_padding"}:
                raise ValueError("This token mapper requires a resize-only image transform")

    @property
    def token_grid(self) -> tuple[int, int]:
        _, height, width = self.grid_thw
        return height // self.spatial_merge_size, width // self.spatial_merge_size


@dataclass(frozen=True)
class TokenFootprint:
    visual_index: int
    source_index: int
    grid_y: int
    grid_x: int
    processed_xyxy: tuple[int, int, int, int]
    original_xyxy: tuple[float, float, float, float]
    mrope_position: tuple[int, int, int]


@dataclass(frozen=True)
class SourceKVMetadata:
    key_representation_stage: str
    value_representation_stage: str
    source_token_ids: list[int]
    chat_template: str
    qk_norm: dict
    rope: dict
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MaskInput:
    mask: np.ndarray
    score: float | None = None
    label: str | None = None
    label_source: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RegionNode:
    region_id: str
    mask_ref: str
    bbox_xyxy: tuple[int, int, int, int]
    area: int
    token_indices: tuple[int, ...]
    token_weights: tuple[float, ...]
    mask_sha256: str
    score: float | None = None
    label: str | None = None
    label_source: str | None = None
    metadata: dict = field(default_factory=dict)
    kv_bank_ref: str = BANK_REF


@dataclass(frozen=True)
class ContactEdge:
    region_a: str
    region_b: str
    contact_kind: str
    distance_convention: str
    relation_type: str = "contact_candidate"
    distance: int = 0


@dataclass(frozen=True)
class KVSlice:
    """Token-major per-layer rows from one source prefill, in source order."""

    layers: dict[int, tuple[torch.Tensor, torch.Tensor]]
    region_ids: tuple[str, ...]
    visual_indices: tuple[int, ...]
    source_indices: tuple[int, ...]
    mrope_positions: tuple[tuple[int, int, int], ...]
    key_representation_stage: str
    value_representation_stage: str


@dataclass
class ImageMemory:
    image: Image.Image
    image_hash: str
    geometry: ImageGeometry
    tokens: list[TokenFootprint]
    regions: list[RegionNode]
    edges: list[ContactEdge]
    masks: dict[str, np.ndarray]
    source_kv: dict[int, tuple[torch.Tensor, torch.Tensor]]
    kv_metadata: SourceKVMetadata
    fingerprints: dict
    config: dict
    segmentation: dict
    unassigned_token_indices: tuple[int, ...]

    @property
    def image_id(self) -> str:
        return self.image_hash

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(
            {
                "schema_version": SCHEMA_VERSION,
                "mapping_version": TOKEN_MAPPING_VERSION,
                "image_hash": self.image_hash,
                "fingerprints": self.fingerprints,
                "geometry": asdict(self.geometry),
                "config": self.config,
                "segmentation": self.segmentation,
                "kv_metadata": asdict(self.kv_metadata),
                "layers": self.layer_specs(),
                "tokens": [asdict(token) for token in self.tokens],
                "regions": [asdict(region) for region in self.regions],
                "edges": [asdict(edge) for edge in self.edges],
                "unassigned_token_indices": self.unassigned_token_indices,
            }
        )

    @property
    def unassigned_tokens(self) -> tuple[int, ...]:
        return self.unassigned_token_indices

    @property
    def source_prefix_token_ids(self) -> list[int]:
        return self.kv_metadata.source_token_ids[: self.tokens[0].source_index]

    @property
    def non_visual_source_indices(self) -> tuple[int, ...]:
        start = self.tokens[0].source_index
        end = self.tokens[-1].source_index + 1
        return (*range(start), *range(end, len(self.kv_metadata.source_token_ids)))

    def layer_specs(self) -> dict[str, dict]:
        return {
            str(layer): {
                "key": {"shape": list(key.shape), "dtype": str(key.dtype)},
                "value": {"shape": list(value.shape), "dtype": str(value.dtype)},
            }
            for layer, (key, value) in sorted(self.source_kv.items())
        }

    def validate(self) -> None:
        """Check structure and every index before serving or persisting a memory."""
        import numpy as np

        from .graph import CONTACT_CONVENTIONS, mask_bbox
        from .mapping import build_token_provenance

        self.geometry.validate()
        if self.image.mode != "RGB" or self.image.size != tuple(self.geometry.original_size):
            raise ValueError("Source image is not the expected display-oriented RGB image")
        if image_sha256(self.image) != self.image_hash:
            raise ValueError("Source image hash mismatch")
        if self.kv_metadata.key_representation_stage == "" or self.kv_metadata.value_representation_stage == "":
            raise ValueError("K/V capture stages must be explicit")
        if not self.tokens or not self.source_kv:
            raise ValueError("Memory must contain image tokens and at least one layer")
        source_indices = [token.source_index for token in self.tokens]
        if source_indices[-1] >= len(self.kv_metadata.source_token_ids):
            raise ValueError("Visual indices extend beyond the source input ID array")
        expected_tokens = build_token_provenance(
            self.geometry, source_indices, [t.mrope_position for t in self.tokens]
        )
        if self.tokens != expected_tokens:
            raise ValueError("Visual token provenance differs from the declared geometry")
        total = len(self.tokens)
        if set(self.source_kv) != set(range(len(self.source_kv))):
            raise ValueError("The source K/V bank must contain every layer starting at zero")
        expected_layers = self.kv_metadata.details.get("num_hidden_layers")
        if expected_layers is not None and len(self.source_kv) != expected_layers:
            raise ValueError("The source bank does not cover the declared decoder layer count")
        for layer, (key, value) in self.source_kv.items():
            for name, tensor in (("K", key), ("V", value)):
                if tensor.ndim < 2 or tensor.shape[0] != total or tensor.shape[1:].numel() == 0:
                    raise ValueError(f"Layer {layer} {name} does not cover the complete visual span")
                if tensor.device.type != "cpu" or tensor.requires_grad or not tensor.is_contiguous():
                    raise ValueError(f"Layer {layer} {name} is not a contiguous detached CPU tensor")

        region_ids = [region.region_id for region in self.regions]
        expected_ids = [f"r{index:04d}" for index in range(1, len(region_ids) + 1)]
        if region_ids != expected_ids or set(region_ids) != self.masks.keys():
            raise ValueError("Region IDs or mask references do not match")
        original_shape = tuple(reversed(self.geometry.original_size))
        covered: set[int] = set()
        for region in self.regions:
            mask = self.masks[region.region_id]
            if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != original_shape:
                raise ValueError(f"{region.region_id} is not an original-coordinate bool mask")
            if (
                region.mask_ref != f"masks/{region.region_id}.png"
                or region.kv_bank_ref != BANK_REF
                or region.mask_sha256 != mask_sha256(mask)
                or region.bbox_xyxy != mask_bbox(mask)
                or region.area != int(np.count_nonzero(mask))
                or region.area <= 0
            ):
                raise ValueError(f"{region.region_id} mask provenance mismatch")
            indices = region.token_indices
            if tuple(sorted(set(indices))) != indices or any(i < 0 or i >= total for i in indices):
                raise ValueError(f"{region.region_id} has invalid visual token indices")
            if len(region.token_weights) != len(indices) or any(
                not math.isfinite(weight) or not 0 < weight <= 1 for weight in region.token_weights
            ):
                raise ValueError(f"{region.region_id} has invalid coverage weights")
            covered.update(indices)
        if tuple(sorted(set(range(total)) - covered)) != self.unassigned_token_indices:
            raise ValueError("Region and unassigned token indices do not partition the visual span")

        pairs = [(edge.region_a, edge.region_b) for edge in self.edges]
        if pairs != sorted(set(pairs)):
            raise ValueError("Duplicated or unordered contact edges")
        mode = self.config.get("contact_mode")
        if mode not in CONTACT_CONVENTIONS:
            raise ValueError(f"Unknown contact mode {mode!r}")
        for edge in self.edges:
            if (
                edge.region_a not in self.masks
                or edge.region_b not in self.masks
                or edge.region_a >= edge.region_b
                or edge.relation_type != "contact_candidate"
                or edge.contact_kind not in {"overlap", "touch"}
                or (mode == "overlap_only" and edge.contact_kind != "overlap")
                or edge.distance != 0
                or edge.distance_convention != CONTACT_CONVENTIONS[mode]
            ):
                raise ValueError("Invalid contact edge")

    def get_neighbors(self, region_id: str) -> tuple[str, ...]:
        if region_id not in self.masks:
            raise KeyError(f"Unknown region {region_id!r}")
        adjacent = {
            (edge.region_b if edge.region_a == region_id else edge.region_a)
            for edge in self.edges
            if region_id in (edge.region_a, edge.region_b)
        }
        return tuple(sorted(adjacent))

    def get_region_kv(self, region_id: str, layers: Iterable[int] | None = None) -> KVSlice:
        return self.get_regions_kv([region_id], layers=layers)

    def get_regions_kv(
        self,
        region_ids: Iterable[str],
        expand_hops: int = 0,
        *,
        layers: Iterable[int] | None = None,
    ) -> KVSlice:
        import torch

        if not isinstance(expand_hops, int) or expand_hops < 0:
            raise ValueError("expand_hops must be a nonnegative integer")
        if isinstance(region_ids, str):
            raise TypeError("Use a sequence of region IDs or call get_region_kv")
        regions = {region.region_id: region for region in self.regions}
        selected = set(region_ids)
        unknown = selected - regions.keys()
        if unknown:
            raise KeyError(f"Unknown regions: {sorted(unknown)}")
        frontier = selected
        for _ in range(expand_hops):
            frontier = {
                neighbor for region_id in frontier for neighbor in self.get_neighbors(region_id)
            } - selected
            if not frontier:
                break
            selected.update(frontier)
        visual_indices = tuple(
            sorted({index for region_id in selected for index in regions[region_id].token_indices})
        )
        index_tensor = torch.tensor(visual_indices, dtype=torch.long)
        requested_layers = sorted(self.source_kv if layers is None else set(layers))
        sliced = {
            layer: (
                self.source_kv[layer][0].index_select(0, index_tensor),
                self.source_kv[layer][1].index_select(0, index_tensor),
            )
            for layer in requested_layers
        }
        return KVSlice(
            layers=sliced,
            region_ids=tuple(sorted(selected)),
            visual_indices=visual_indices,
            source_indices=tuple(self.tokens[i].source_index for i in visual_indices),
            mrope_positions=tuple(self.tokens[i].mrope_position for i in visual_indices),
            key_representation_stage=self.kv_metadata.key_representation_stage,
            value_representation_stage=self.kv_metadata.value_representation_stage,
        )

    def export_region(self, region_id: str, output_path: str | Path) -> Path:
        from .store import export_region

        return export_region(self, region_id, output_path)
