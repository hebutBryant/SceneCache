"""Opt-in tracing for an isolated report server; leaves model calculations intact.

Loaded by a task-owned sitecustomize only when SCENECACHE_REPORT_DIR is set.
The control file can clear the demo process's cache between completed requests.
"""

import collections
import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import threading
import time


def install():
    directory = os.environ.get("SCENECACHE_REPORT_DIR")
    if not directory:
        return
    root = Path(directory)
    control_path = root / "control.json"
    event_path = root / "trace.jsonl"
    lock = threading.Lock()
    current = {"request_id": "startup", "reset": False}
    last_stamp = None

    def emit(event, **data):
        record = {"event": event, "request_id": current["request_id"],
                  "pid": os.getpid(), "time_ns": time.time_ns(), **data}
        encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode()
        with lock:
            fd = os.open(event_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, encoded)
            finally:
                os.close(fd)

    from sglang.srt.managers.mock_kv_manager import mock_kv_manager

    def refresh():
        nonlocal last_stamp, current
        try:
            stamp = control_path.stat().st_mtime_ns
            if stamp == last_stamp:
                return
            next_control = json.loads(control_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return
        last_stamp = stamp
        current = next_control
        if current.get("reset"):
            mock_kv_manager.release_all()
            mm_utils = sys.modules.get("sglang.srt.managers.mm_utils")
            embedding_cache = getattr(mm_utils, "embedding_cache", None)
            keep_embedding = current.get("keep_embedding", False)
            if embedding_cache is not None and not keep_embedding:
                embedding_cache.clear()
            emit("cache_reset", keep_embedding=keep_embedding)

    class ObservedDict(collections.OrderedDict):
        def __contains__(self, key):
            refresh()
            return super().__contains__(key)

    mock_kv_manager.uuid_to_pace = ObservedDict(mock_kv_manager.uuid_to_pace)
    manager_cls = type(mock_kv_manager)
    for method_name in ("write_kv", "get_kv"):
        original = getattr(manager_cls, method_name)

        def wrap(method, name):
            @functools.wraps(method)
            def traced(self, tensor, uuid, *args, **kwargs):
                refresh()
                started = time.perf_counter()
                result = method(self, tensor, uuid, *args, **kwargs)
                emit(name, key=str(uuid), kind="vision" if str(uuid).endswith("_vit_emb") else "kv",
                     shape=list(tensor.shape), dtype=str(tensor.dtype),
                     bytes=tensor.numel() * tensor.element_size(),
                     elapsed_ms=(time.perf_counter() - started) * 1000)
                return result
            return traced

        setattr(manager_cls, method_name, wrap(original, method_name))

    from sglang.srt.mem_cache.multimodal_cache import MultiModalCache
    original_get = MultiModalCache.get
    original_put = MultiModalCache.put

    def embedding_get(self, key):
        refresh()
        result = original_get(self, key)
        emit("embedding_cache_get", hit=result is not None,
             shape=list(result.shape) if result is not None else None,
             bytes=result.numel() * result.element_size() if result is not None else 0,
             device=str(result.device) if result is not None else None)
        return result

    def embedding_put(self, key, tensor):
        result = original_put(self, key, tensor)
        emit("embedding_cache_put", stored=result, shape=list(tensor.shape),
             bytes=tensor.numel() * tensor.element_size(), device=str(tensor.device))
        return result

    MultiModalCache.get = embedding_get
    MultiModalCache.put = embedding_put

    target = "sglang.srt.layers.attention.flashinfer_backend"

    class TraceLoader(importlib.abc.Loader):
        def __init__(self, loader):
            self.loader = loader

        def create_module(self, spec):
            return self.loader.create_module(spec) if hasattr(self.loader, "create_module") else None

        def exec_module(self, module):
            self.loader.exec_module(module)
            cls = module.FlashInferIndicesUpdaterPrefill
            original = cls.update_variable_block_wrapper

            @functools.wraps(original)
            def traced(instance, *args, **kwargs):
                refresh()
                result = original(instance, *args, **kwargs)
                batch = kwargs.get("forward_batch")
                if batch is None:
                    batch = args[-1]
                if not batch.forward_mode.is_extend():
                    return result
                total = batch.input_ids.numel()
                masks = getattr(batch, "compute_mask", None) or {}
                mask_lists = {str(layer): mask.detach().cpu().tolist() for layer, mask in masks.items()}
                image_ranges = []
                image_grids = []
                for mm in batch.mm_inputs or []:
                    if mm is not None:
                        for item in mm.mm_items:
                            image_ranges.extend([list(pair) for pair in item.offsets])
                            grid = getattr(item, "image_grid_thw", None)
                            if grid is not None:
                                image_grids.extend(grid.tolist())
                emit("prefill", total_input_tokens=total,
                     image_ranges=image_ranges, image_grids=image_grids,
                     first_layer_compute_mask=mask_lists.get("0"),
                     computed_tokens_per_layer={layer: sum(mask) for layer, mask in mask_lists.items()},
                     cached_layers=len(getattr(batch, "recompute_info", None) or {}))
                return result

            cls.update_variable_block_wrapper = traced

    class TraceFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "sglang.srt.layers.attention.flashinfer_backend":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.loader:
                spec.loader = TraceLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, TraceFinder())

