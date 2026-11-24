"""Compact, persist and resume HF 4.57 Qwen2.5-VL's native dynamic K/V cache.

This bank stores post-MRoPE K, unlike the raw projection bank in SceneCache.
Logical MRoPE coordinates remain external to DynamicCache, which only tracks
the physical sequence length after compaction.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file, save_file
from transformers.cache_utils import DynamicCache, DynamicLayer

from .schema import content_fingerprint, json_bytes
from .store import _file_record, _sync

BANK = "terminal_kv.safetensors"
MANIFEST = "manifest.json"
FORMAT = "transformers_4_57_qwen2_5_vl_full_dynamic_cache"
STAGES = {"key": "post_mrope", "value": "raw_projection"}


def _cache_pairs(cache: DynamicCache):
    if not isinstance(cache, DynamicCache) or cache.offloading or not cache.layers:
        raise ValueError("Expected a populated, resident Transformers DynamicCache")
    length = cache.get_seq_length()
    if not length:
        raise ValueError("Cannot use an empty source cache for terminal reuse")
    pairs = []
    device = cache.layers[0].keys.device
    for layer in cache.layers:
        if type(layer) is not DynamicLayer or not layer.is_initialized:
            raise ValueError(
                "Only initialized full-attention DynamicLayer entries are supported"
            )
        key, value = layer.keys, layer.values
        if (
            key.ndim != 4
            or key.shape != value.shape
            or key.shape[0] != 1
            or key.shape[-2] != length
            or not key.numel()
            or key.device != device
            or value.device != device
            or key.dtype != value.dtype
        ):
            raise ValueError(
                "Expected one uniform-device batch with all source K/V rows in every layer"
            )
        pairs.append((key, value))
    return pairs


def _adopt_pairs(pairs):
    """Initialize native layers with owned tensors without concatenating a duplicate."""
    cache = DynamicCache()
    for key, value in pairs:
        layer = DynamicLayer()
        layer.lazy_initialization(key)
        layer.keys = key
        layer.values = value
        cache.layers.append(layer)
    _cache_pairs(cache)
    return cache


def _validate_indices(indices: Iterable[int] | torch.Tensor, length: int):
    if isinstance(indices, torch.Tensor):
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("Cache row indices must be a one-dimensional LongTensor")
        indices = indices.detach().cpu().tolist()
    else:
        indices = list(indices)
    if (
        not indices
        or any(
            not isinstance(index, int) or isinstance(index, bool) for index in indices
        )
        or not (0 <= indices[0] <= indices[-1] < length)
        or any(before >= after for before, after in zip(indices, indices[1:]))
    ):
        raise ValueError(
            "Cache row indices must be nonempty, unique, in range and in source order"
        )
    return indices


@torch.no_grad()
def compact_cache(
    source_cache: DynamicCache, indices: Iterable[int] | torch.Tensor
) -> DynamicCache:
    """Gather exact source rows into disjoint freshly allocated K/V tensors."""
    pairs = _cache_pairs(source_cache)
    rows = torch.tensor(
        _validate_indices(indices, source_cache.get_seq_length()),
        dtype=torch.long,
        device=pairs[0][0].device,
    )
    return _adopt_pairs(
        (key.detach().index_select(-2, rows), value.detach().index_select(-2, rows))
        for key, value in pairs
    )


def branch_cache(cache: DynamicCache) -> DynamicCache:
    """Create a separate append-only layer container over the same immutable rows.

    Transformers 4.57 DynamicLayer updates replace K/V via torch.cat. This
    branch owns its layer containers and shares only the initial tensor rows.
    """
    return _adopt_pairs(_cache_pairs(cache))


@torch.no_grad()
def copy_cache(cache: DynamicCache, *, device=None) -> DynamicCache:
    """Make an independent native cache, optionally offloading or restoring it."""
    pairs = _cache_pairs(cache)
    target = pairs[0][0].device if device is None else torch.device(device)
    return _adopt_pairs(
        tuple(
            tensor.detach().to(target, copy=True, memory_format=torch.contiguous_format)
            for tensor in pair
        )
        for pair in pairs
    )


def continuation_inputs(cache: DynamicCache, position_ids: torch.Tensor) -> dict:
    """Pair physical append positions with unchanged logical MRoPE suffix positions."""
    pairs = _cache_pairs(cache)
    if (
        position_ids.ndim != 3
        or position_ids.shape[:2] != (3, 1)
        or not position_ids.shape[-1]
        or position_ids.dtype != torch.long
    ):
        raise ValueError(
            "Expected original Qwen MRoPE suffix positions with shape [3, 1, new_tokens]"
        )
    device = pairs[0][0].device
    past = cache.get_seq_length()
    end = past + position_ids.shape[-1]
    return {
        "cache_position": torch.arange(past, end, dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, end), dtype=torch.long, device=device),
        "position_ids": position_ids.to(device),
    }


def cache_stats(cache: DynamicCache) -> dict:
    """Report logical tensor bytes and unique live storage bytes, including pointers."""
    pairs = _cache_pairs(cache)
    layers, storages = [], {}
    logical_bytes = 0
    for layer, pair in enumerate(pairs):
        layer_info = {"layer": layer}
        for name, tensor in zip(("key", "value"), pair):
            storage = tensor.untyped_storage()
            allocated = storage.nbytes()
            used = tensor.numel() * tensor.element_size()
            pointer = storage.data_ptr()
            device = str(tensor.device)
            logical_bytes += used
            storages[(device, pointer)] = allocated
            layer_info[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": device,
                "logical_bytes": used,
                "storage_bytes": allocated,
                "storage_ptr": pointer,
                "storage_offset": tensor.storage_offset(),
            }
        layers.append(layer_info)
    return {
        "sequence_length": cache.get_seq_length(),
        "layer_count": len(layers),
        "logical_bytes": logical_bytes,
        "storage_bytes": sum(storages.values()),
        "storage_count": len(storages),
        "layers": layers,
    }


def _tensor_name(layer, kind):
    return f"layer.{layer}.{kind}"


def _spec(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def _payload(cache):
    return {
        _tensor_name(layer, kind): tensor.detach().to("cpu").contiguous()
        for layer, pair in enumerate(_cache_pairs(cache))
        for kind, tensor in zip(("k", "v"), pair)
    }


def save_cache(directory, cache: DynamicCache, *, metadata: dict) -> Path:
    """Commit tensor-only data and source compatibility metadata together."""
    if not isinstance(metadata, dict):
        raise ValueError("Terminal cache compatibility metadata must be an object")
    target = Path(directory)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.writing-", dir=target.parent)
    )
    try:
        tensors = _payload(cache)
        save_file(tensors, scratch / BANK)
        manifest = {
            "format": FORMAT,
            "schema_version": 1,
            "representation_stages": STAGES,
            "sequence_length": cache.get_seq_length(),
            "layer_count": len(cache.layers),
            "metadata": metadata,
            "tensors": {
                name: _spec(tensor) for name, tensor in sorted(tensors.items())
            },
            "bank": _file_record(scratch / BANK),
        }
        manifest["fingerprint"] = content_fingerprint(manifest)
        (scratch / MANIFEST).write_bytes(json_bytes(manifest))
        for path in (scratch / BANK, scratch / MANIFEST, scratch):
            _sync(path)
        os.rename(scratch, target)
        _sync(target.parent)
    finally:
        if scratch.exists():
            for path in (scratch / BANK, scratch / MANIFEST):
                path.unlink(missing_ok=True)
            scratch.rmdir()
    return target


def load_cache(directory, *, device, expected_metadata: dict | None = None):
    """Verify compatibility and exact payload bytes before restoring a native cache."""
    target = Path(directory)
    manifest = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
    fingerprint = manifest.pop("fingerprint", None)
    if (
        manifest.get("format") != FORMAT
        or manifest.get("schema_version") != 1
        or manifest.get("representation_stages") != STAGES
        or fingerprint != content_fingerprint(manifest)
    ):
        raise ValueError("Terminal cache manifest format or fingerprint mismatch")
    if expected_metadata is not None and json_bytes(expected_metadata) != json_bytes(
        manifest["metadata"]
    ):
        raise ValueError("Terminal cache compatibility metadata mismatch")
    if _file_record(target / BANK) != manifest["bank"]:
        raise ValueError("Terminal K/V bank payload hash or length mismatch")
    tensors = load_file(target / BANK, device="cpu")
    if {name: _spec(tensor) for name, tensor in tensors.items()} != manifest["tensors"]:
        raise ValueError("Terminal K/V tensor metadata mismatch")
    layers = manifest["layer_count"]
    if (
        not isinstance(layers, int)
        or isinstance(layers, bool)
        or layers < 1
        or set(tensors)
        != {_tensor_name(layer, kind) for layer in range(layers) for kind in ("k", "v")}
    ):
        raise ValueError("Terminal cache does not contain every native layer pair")
    pairs = (
        tuple(tensors[_tensor_name(layer, kind)].to(device) for kind in ("k", "v"))
        for layer in range(layers)
    )
    cache = _adopt_pairs(pairs)
    if cache.get_seq_length() != manifest["sequence_length"]:
        raise ValueError("Terminal cache sequence length mismatch")
    return cache, manifest["metadata"]
