"""Verified, atomic, tensor-only storage for a source visual K/V bank."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from .schema import (
    BANK_REF,
    SCHEMA_VERSION,
    TOKEN_MAPPING_VERSION,
    ContactEdge,
    ImageGeometry,
    ImageMemory,
    RegionNode,
    SourceKVMetadata,
    TokenFootprint,
    json_bytes,
)

MANIFEST = "manifest.json"
BANK = "source_kv.safetensors"
IMAGE = "image.png"


def _file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    length = 0
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
            length += len(block)
    return {"bytes": length, "sha256": digest.hexdigest()}


def _sync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _bank_tensors(layers: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        f"layer.{layer}.{kind}": tensor
        for layer, pair in sorted(layers.items())
        for kind, tensor in zip(("k", "v"), pair)
    }


def inspect_summary(memory: ImageMemory) -> dict:
    memberships = Counter(index for region in memory.regions for index in region.token_indices)
    index_data = {
        "regions": {
            region.region_id: {"indices": region.token_indices, "weights": region.token_weights}
            for region in memory.regions
        },
        "unassigned": memory.unassigned_token_indices,
    }
    first_layer = memory.source_kv[min(memory.source_kv)]
    return {
        "schema_version": memory.schema_version,
        "fingerprint": memory.fingerprint,
        "image_hash": memory.image_hash,
        "image_size": list(memory.geometry.original_size),
        "fingerprints": memory.fingerprints,
        "regions": len(memory.regions),
        "edges": len(memory.edges),
        "edge_kinds": dict(Counter(edge.contact_kind for edge in memory.edges)),
        "visual_tokens": len(memory.tokens),
        "assigned_tokens": len(memberships),
        "multiply_assigned_tokens": sum(count > 1 for count in memberships.values()),
        "unassigned_tokens": len(memory.unassigned_token_indices),
        "per_region": [
            {
                "region_id": region.region_id,
                "label": region.label,
                "area": region.area,
                "score": region.score,
                "tokens": len(region.token_indices),
                "neighbors": list(memory.get_neighbors(region.region_id)),
            }
            for region in memory.regions
        ],
        "layers": len(memory.source_kv),
        "layer_specs": memory.layer_specs(),
        "first_layer_key_shape": list(first_layer[0].shape),
        "first_layer_value_shape": list(first_layer[1].shape),
        "first_layer_key_dtype": str(first_layer[0].dtype),
        "first_layer_value_dtype": str(first_layer[1].dtype),
        "key_representation_stage": memory.kv_metadata.key_representation_stage,
        "value_representation_stage": memory.kv_metadata.value_representation_stage,
        "kv_bank_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in _bank_tensors(memory.source_kv).values()
        ),
        "membership_index_json_bytes": len(json_bytes(index_data)),
        "segmentation": memory.segmentation,
    }


def _manifest(memory: ImageMemory, files: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "token_mapping_version": TOKEN_MAPPING_VERSION,
        "fingerprint": memory.fingerprint,
        "image_hash": memory.image_hash,
        "geometry": asdict(memory.geometry),
        "fingerprints": memory.fingerprints,
        "config": memory.config,
        "segmentation": memory.segmentation,
        "kv_metadata": asdict(memory.kv_metadata),
        "source_prefix_token_ids": memory.source_prefix_token_ids,
        "non_visual_source_indices": memory.non_visual_source_indices,
        "source_bank": {"id": BANK_REF, "file": BANK, "layers": memory.layer_specs()},
        "tokens": [asdict(token) for token in memory.tokens],
        "regions": [asdict(region) for region in memory.regions],
        "edges": [asdict(edge) for edge in memory.edges],
        "unassigned_token_indices": memory.unassigned_token_indices,
        "summary": inspect_summary(memory),
        "files": files,
    }


def _json_dataclass(cls, fields, tuple_fields):
    return cls(**{
        key: tuple(value) if key in tuple_fields else value for key, value in fields.items()
    })


def load_memory(
    output_dir: str | Path, *, expected_fingerprint: str | None = None
) -> ImageMemory:
    """Load only committed directories and validate disk bytes and structural provenance."""
    root = Path(output_dir)
    try:
        manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exception:
        raise ValueError(f"Invalid or incomplete memory at {root}: {exception}") from exception
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("token_mapping_version") != TOKEN_MAPPING_VERSION:
        raise ValueError("Unsupported memory schema or token mapping version")
    if expected_fingerprint is not None and manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"Memory fingerprint mismatch at {root}")
    if manifest["source_bank"]["id"] != BANK_REF or manifest["source_bank"]["file"] != BANK:
        raise ValueError("Invalid source bank reference")

    region_data = manifest["regions"]
    for index, region in enumerate(region_data, 1):
        region_id = f"r{index:04d}"
        if region["region_id"] != region_id or region["mask_ref"] != f"masks/{region_id}.png":
            raise ValueError("Invalid region mask filename")
    mask_files = [region["mask_ref"] for region in region_data]
    expected_files = {BANK, IMAGE, *mask_files}
    if manifest["files"].keys() != expected_files:
        raise ValueError("Memory file table does not match required payloads")
    for name in sorted(expected_files):
        try:
            actual = _file_record(root / name)
        except OSError as exception:
            raise ValueError(f"Missing payload {name} in {root}") from exception
        if actual != manifest["files"][name]:
            raise ValueError(f"Memory payload hash or size mismatch: {name}")

    layers = manifest["source_bank"]["layers"]
    if set(layers) != {str(index) for index in range(len(layers))}:
        raise ValueError("Layer spec does not cover a contiguous set of decoder layers")
    loaded = load_file(root / BANK, device="cpu")
    if set(loaded) != {f"layer.{layer}.{kind}" for layer in layers for kind in ("k", "v")}:
        raise ValueError("Unexpected or incomplete layer tensors in source bank")
    source_kv = {
        index: (loaded[f"layer.{index}.k"], loaded[f"layer.{index}.v"])
        for index in range(len(layers))
    }
    masks = {}
    for region in region_data:
        with Image.open(root / region["mask_ref"]) as encoded:
            if encoded.mode != "1":
                raise ValueError(f"Mask {region['region_id']} must be a 1-bit image")
            masks[region["region_id"]] = np.array(encoded, dtype=np.bool_)
    with Image.open(root / IMAGE) as encoded_image:
        image = encoded_image.copy()

    memory = ImageMemory(
        image=image,
        image_hash=manifest["image_hash"],
        geometry=_json_dataclass(
            ImageGeometry, manifest["geometry"], {"original_size", "processed_size", "grid_thw"}
        ),
        tokens=[
            _json_dataclass(
                TokenFootprint, token, {"processed_xyxy", "original_xyxy", "mrope_position"}
            )
            for token in manifest["tokens"]
        ],
        regions=[
            _json_dataclass(RegionNode, region, {"bbox_xyxy", "token_indices", "token_weights"})
            for region in region_data
        ],
        edges=[ContactEdge(**edge) for edge in manifest["edges"]],
        masks=masks,
        source_kv=source_kv,
        kv_metadata=SourceKVMetadata(**manifest["kv_metadata"]),
        fingerprints=manifest["fingerprints"],
        config=manifest["config"],
        segmentation=manifest["segmentation"],
        unassigned_token_indices=tuple(manifest["unassigned_token_indices"]),
    )
    memory.validate()
    if (
        memory.fingerprint != manifest["fingerprint"]
        or memory.layer_specs() != layers
        or memory.source_prefix_token_ids != manifest["source_prefix_token_ids"]
        or memory.non_visual_source_indices != tuple(manifest["non_visual_source_indices"])
    ):
        raise ValueError("Memory metadata or fingerprint does not match the payloads")
    return memory


def _same_source_values(memory: ImageMemory, reopened: ImageMemory) -> None:
    if memory.fingerprint != reopened.fingerprint:
        raise ValueError("Loaded memory does not match its source provenance")
    for layer, pair in memory.source_kv.items():
        for name, source, disk in zip(("K", "V"), pair, reopened.source_kv[layer]):
            if source.dtype != disk.dtype or source.shape != disk.shape or not torch.equal(
                source.view(torch.uint8), disk.view(torch.uint8)
            ):
                raise ValueError(f"Round trip changed layer {layer} {name} values")


def _accept_existing(memory: ImageMemory, target: Path) -> Path:
    try:
        existing = load_memory(target, expected_fingerprint=memory.fingerprint)
        _same_source_values(memory, existing)
    except (OSError, ValueError, RuntimeError) as exception:
        raise FileExistsError(f"Existing output differs from this memory: {target}") from exception
    return target


def save_memory(memory: ImageMemory, output_dir: str | Path) -> Path:
    """Commit a validated directory atomically, refusing stale or partial outputs."""
    memory.validate()
    target = Path(output_dir).absolute()
    if target.exists():
        return _accept_existing(memory, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.staging-", dir=target.parent) as staging:
        stage = Path(staging)
        memory.image.save(stage / IMAGE)
        save_file(_bank_tensors(memory.source_kv), stage / BANK, metadata={"format": "pt"})
        (stage / "masks").mkdir()
        for region in memory.regions:
            Image.fromarray(memory.masks[region.region_id]).save(stage / region.mask_ref)
        files = {
            name: _file_record(stage / name)
            for name in [IMAGE, BANK, *(region.mask_ref for region in memory.regions)]
        }
        manifest = _manifest(memory, files)
        (stage / MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        reopened = load_memory(stage, expected_fingerprint=memory.fingerprint)
        _same_source_values(memory, reopened)
        del reopened
        for name in [*files, MANIFEST, "masks"]:
            _sync(stage / name)
        _sync(stage)
        try:
            stage.rename(target)
        except OSError as exception:
            if exception.errno in (errno.EEXIST, errno.ENOTEMPTY):
                return _accept_existing(memory, target)
            raise
    _sync(target.parent)
    return target


def export_region(memory: ImageMemory, region_id: str, output_path: str | Path) -> Path:
    """Save selected rows and their positions in one standalone safetensors file."""
    selection = memory.get_region_kv(region_id)
    region = next(region for region in memory.regions if region.region_id == region_id)
    target = Path(output_path).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": memory.schema_version,
        "memory_fingerprint": memory.fingerprint,
        "image_hash": memory.image_hash,
        "region_id": region_id,
        "mask_ref": region.mask_ref,
        "token_weights": region.token_weights,
        "visual_indices": selection.visual_indices,
        "source_indices": selection.source_indices,
        "mrope_positions": selection.mrope_positions,
        "key_representation_stage": selection.key_representation_stage,
        "value_representation_stage": selection.value_representation_stage,
        "qk_norm": memory.kv_metadata.qk_norm,
        "rope": memory.kv_metadata.rope,
    }
    with tempfile.TemporaryDirectory(prefix=".region-", dir=target.parent) as staging:
        pending = Path(staging) / "region.safetensors"
        save_file(
            _bank_tensors(selection.layers),
            pending,
            metadata={"format": "pt", "scene_graph_memory": json_bytes(metadata).decode("utf-8")},
        )
        _sync(pending)
        pending.replace(target)
    _sync(target.parent)
    return target
