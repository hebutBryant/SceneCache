"""Single-request raw visual KV restore for the Qwen3-VL decoder."""

from contextlib import contextmanager
from functools import lru_cache
import json
import os
from pathlib import Path
from time import perf_counter_ns

import torch

from .kv_transfer import KVLayerTransfer
from .selection import MEMORY_ENV, METRICS_ENV, MODES, REGION_MODES, SELECTION_ENV, mode

PROFILE_ENV = "SGLANG_VISUAL_KV_PROFILE"
VERIFY_ENV = "SGLANG_VISUAL_KV_VERIFY_DIR"


@contextmanager
def profile_range(name):
    if os.getenv(PROFILE_ENV):
        with torch.autograd.profiler.record_function(name), torch.cuda.nvtx.range(name):
            yield
    else:
        yield


def emit_metrics(**record):
    if path := os.getenv(METRICS_ENV):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time_ns": perf_counter_ns(), **record}) + "\n")


def observe_legacy_layer(batch, layer, **tensors):
    """Save representative legacy activations only in an explicit correctness run."""
    if not os.getenv(VERIFY_ENV) or layer not in (0, 18, 35):
        return
    if not hasattr(batch, "_legacy_visual_probes"):
        batch._legacy_visual_probes = {}
    batch._legacy_visual_probes.setdefault(layer, {}).update(
        {name: tensor.detach().cpu().clone() for name, tensor in tensors.items()}
    )


def finish_legacy_probes(batch, output):
    if hasattr(batch, "_legacy_visual_probes"):
        destination = Path(os.environ[VERIFY_ENV])
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"layers": batch._legacy_visual_probes, "logits": output.next_token_logits.detach().cpu().clone()},
            destination / "legacy_full_kv.pt",
        )


def observe_region_decode(batch):
    if not os.getenv(METRICS_ENV):
        return
    for mm in batch.mm_inputs or []:
        if mm is None:
            continue
        for item in mm.mm_items:
            plan = getattr(item, "visual_kv_selection", None)
            if plan is not None:
                emit_metrics(
                    event="optimized_decode", mode=mode(), backend="flashinfer_paged",
                    n_all=plan.source_visual_count, n_active=len(plan.source_visual_indices),
                    n_recompute=0, kv_length=batch.seq_lens_sum, h2d_bytes=0,
                )


def validate_server(model_runner):
    selected_mode = mode()
    if selected_mode == "legacy":
        return
    if selected_mode not in MODES:
        raise ValueError(f"Unknown visual KV mode {selected_mode}")
    args = model_runner.server_args
    if (
        args.tp_size != 1 or args.pp_size != 1 or args.dp_size != 1
        or args.max_running_requests != 1 or not args.disable_cuda_graph
        or not args.disable_radix_cache or not args.disable_overlap_schedule
        or (args.chunked_prefill_size or 0) > 0 or args.speculative_algorithm is not None
        or args.enable_dp_attention or args.page_size != 1
        or os.getenv("SGLANG_SCENE_GRAPH_CAPTURE_DIR")
        or os.getenv("SGLANG_VISUAL_KV_DISABLE_DECODER_CACHE") == "1"
    ):
        raise ValueError(
            "Visual KV optimization requires TP/PP/DP=1, max-running-requests=1, page-size=1, "
            "CUDA graph, radix cache, overlap scheduling, chunking, speculation and capture disabled"
        )
    if (
        model_runner.model_config.hf_config.architectures != ["Qwen3VLForConditionalGeneration"]
        or model_runner.dtype != torch.bfloat16
        or model_runner.kv_cache_dtype != model_runner.dtype or args.quantization is not None
    ):
        raise ValueError("Visual KV optimization requires dense Qwen3-VL with BF16 model and KV dtype")
    if selected_mode in REGION_MODES:
        if not (os.getenv(MEMORY_ENV) and os.getenv(SELECTION_ENV)):
            raise ValueError(f"Region mode requires {MEMORY_ENV} and {SELECTION_ENV}")
        started = perf_counter_ns()
        _load_source(os.environ[MEMORY_ENV], str(Path(args.model_path).resolve()))
        emit_metrics(event="visual_kv_memory_load", elapsed_ms=(perf_counter_ns() - started) / 1e6)


@lru_cache(maxsize=1)
def _load_source(directory, model_path):
    from .builder import _fingerprint_files, _model_files
    from .store import load_memory

    memory = load_memory(directory)
    weights, processor = _model_files(Path(model_path))
    if (
        memory.fingerprints["architecture"] != "Qwen3VLForConditionalGeneration"
        or memory.fingerprints["model_weights_sha256"] != _fingerprint_files(weights)
        or memory.fingerprints["processor_files_sha256"] != _fingerprint_files(processor)
        or memory.kv_metadata.key_representation_stage != "qkv_projection_raw_k_pre_qk_norm_pre_mrope"
        or memory.kv_metadata.value_representation_stage != "qkv_projection_raw_v"
    ):
        raise ValueError("Region KV memory does not match the running Qwen model, processor or KV stage")
    return memory


def start_optimized_prefill(updater, batch, wrappers, use_ragged):
    """Return whether the request uses the optimized prefill."""
    selected_mode = mode()
    if selected_mode not in MODES or not batch.forward_mode.is_extend():
        return False
    inputs = [item for mm in batch.mm_inputs if mm for item in mm.mm_items]
    if not inputs:
        return False
    if (
        len(batch.mm_inputs) != 1 or len(inputs) != 1 or len(inputs[0].offsets) != 1
        or any(batch.extend_prefix_lens_cpu) or use_ragged
    ):
        raise ValueError("Visual KV optimization requires one unchunked image and no cached text prefix")
    if len(set(updater.recompute_ratio_in_layer)) != 1:
        raise ValueError("Visual KV optimization requires a uniform recompute ratio")
    item = inputs[0]
    start, stop = item.offsets[0]
    stop += 1
    total = batch.input_ids.numel()
    if not 0 < start < stop < total:
        raise ValueError("Expected an image span surrounded by a nonempty text prefix and suffix")
    n_active = stop - start
    n_recompute = max(1, int(n_active * updater.recompute_ratio_in_layer[0]))
    source_indices = None
    source_type = "cpu_mock_kv"
    plan = getattr(item, "visual_kv_selection", None)

    lookup_start = perf_counter_ns()
    with profile_range("scene_kv.lookup"):
        if selected_mode in REGION_MODES:
            memory = _load_source(os.environ[MEMORY_ENV], updater.visual_kv_model_path)
            if plan is None or plan.memory_fingerprint != memory.fingerprint:
                raise ValueError("Region selection does not match the loaded memory")
            all_source = memory.source_kv
            n_all = plan.source_visual_count
            source_indices = torch.tensor(plan.source_visual_indices[n_recompute:], dtype=torch.long)
            source_type = "cpu_scene_graph_bank"
        else:
            if plan is not None:
                raise ValueError("Full KV mode received compacted inputs")
            all_source = updater.mock_kv_manager.snapshot_layer_pairs(item.split_hash[0], updater.layer_ids)
            if all_source is None:
                emit_metrics(event="visual_kv_fallback", mode=selected_mode, reason="cpu_cache_miss")
                return False
            n_all = n_active
    lookup_ms = (perf_counter_ns() - lookup_start) / 1e6

    expected_shape = (n_all, updater.kv_size)
    if set(all_source) != set(updater.layer_ids) or any(
        tensor.shape != expected_shape or tensor.dtype != updater.data_type
        or tensor.device.type != "cpu" for pair in all_source.values() for tensor in pair
    ):
        raise ValueError(f"Raw visual KV must have per-layer shape {expected_shape} and model dtype")

    device = batch.input_ids.device
    state = OptimizedVisualPrefill(
        selected_mode, plan, all_source, source_type, source_indices, device,
        start, stop, n_all, n_recompute, total, lookup_ms,
    )
    try:
        if plan is not None:
            item["visual_kv_active_device"] = state.device_index(plan.source_visual_indices)
        if selected_mode == "region_pipeline_triton":
            state.logical_key_positions = state.device_index(plan.logical_positions).to(torch.int32)
            state.logical_query_positions = state.logical_key_positions.index_select(0, state.compute_indices)
        else:
            block_columns = [state.reuse_start, state.reuse_rows, total - stop]
            block_rows = [state.reuse_start, total - stop]
            block_mask = [[[True, False, False], [True, True, True]]]
            if state.reuse_rows == 0:
                block_columns = [total]
                block_rows = [total]
                block_mask = [[[True]]]
            block_mask = state.device_index(block_mask, dtype=torch.bool).repeat(updater.num_kv_heads, 1, 1)
            rows = state.device_index([block_rows], dtype=torch.int32).repeat(updater.num_kv_heads, 1)
            columns = state.device_index([block_columns], dtype=torch.int32).repeat(updater.num_kv_heads, 1)
            with profile_range("scene_kv.flashinfer_plan"):
                wrappers[updater.recompute_ratio_in_layer[0]].plan(
                    block_mask, rows, columns, updater.num_qo_heads, updater.num_kv_heads,
                    updater.head_dim, causal=True, non_blocking=True,
                    q_data_type=updater.q_data_type, kv_data_type=updater.data_type,
                )
        batch.actual_extend_seq_len = state.device_index([total - state.reuse_rows], dtype=torch.int32)
        batch.compute_mask = {}
        batch.recompute_info = {}
        batch.write_info = {}
        batch.optimized_visual_kv = state
    except BaseException:
        state.close()
        raise
    return True


class OptimizedVisualPrefill:
    def __init__(
        self, selected_mode, plan, source, source_type, source_indices, device,
        start, stop, n_all, n_recompute, total, lookup_ms,
    ):
        self.mode = selected_mode
        self.plan = plan
        self.device = device
        self.source_type = source_type
        self.lookup_ms = lookup_ms
        self.n_all = n_all
        self.n_active = stop - start
        self.n_recompute = n_recompute
        self.reuse_start = start + n_recompute
        self.reuse_stop = stop
        self.reuse_rows = stop - self.reuse_start
        self.total = total
        self.h2d_index_bytes = 0
        self.index_copy_calls = 0
        self.reference_copy_calls = 0
        self.reference_gather_ms = 0.0
        self.reference_copy_ms = 0.0
        self.transfer = None
        self.verify = Path(os.environ[VERIFY_ENV]) if os.getenv(VERIFY_ENV) else None
        self.probes = {}
        self.compute_indices = torch.cat((
            torch.arange(self.reuse_start, device=device),
            torch.arange(stop, total, device=device),
        ))
        self.source_cpu_bytes = sum(
            tensor.numel() * tensor.element_size() for pair in source.values() for tensor in pair
        )
        key, value = next(iter(source.values()))
        self.scratch = (key.new_empty((total, key.shape[-1]), device=device),
                        value.new_empty((total, value.shape[-1]), device=device))
        self.layers = tuple(sorted(source))
        self.indices_cpu = source_indices
        if source_indices is None:
            self.source = {layer: tuple(tensor[n_recompute:] for tensor in pair) for layer, pair in source.items()}
        else:
            self.source = source
        self.bytes_per_reused_layer = self.reuse_rows * sum(t.shape[-1] * t.element_size() for t in (key, value))
        if self.reuse_rows and self.mode != "region_reference":
            budget = int(os.getenv("SGLANG_VISUAL_KV_STAGING_BYTES", 32 * 1024**2))
            depth = int(os.getenv("SGLANG_VISUAL_KV_PREFETCH_DEPTH", 2))
            self.transfer = KVLayerTransfer(
                self.source, device, source_indices=source_indices, prefetch_depth=depth,
                max_pinned_bytes=budget, max_gpu_bytes=budget, telemetry=bool(os.getenv(PROFILE_ENV)),
            )
            try:
                for layer in self.layers[:depth]:
                    self.transfer.schedule(layer)
            except BaseException:
                self.close()
                raise

    def device_index(self, values, dtype=torch.long):
        index = torch.tensor(values, dtype=dtype)
        self.h2d_index_bytes += index.numel() * index.element_size()
        self.index_copy_calls += 1
        return index.to(self.device)

    def raw_pair(self, layer):
        if self.transfer:
            return self.transfer.consume(layer)
        started = perf_counter_ns()
        pair = self.source[layer]
        if self.indices_cpu is not None:
            with profile_range("scene_kv.cpu_gather"):
                pair = tuple(t.index_select(0, self.indices_cpu) for t in pair)
        packed = perf_counter_ns()
        with profile_range("scene_kv.sync_h2d"):
            pair = tuple(t.to(self.device) for t in pair)
        self.reference_copy_calls += 2
        self.reference_gather_ms += (packed - started) / 1e6
        self.reference_copy_ms += (perf_counter_ns() - packed) / 1e6
        return pair

    def release(self, layer):
        if self.transfer:
            self.transfer.release(layer)
            future = layer + self.transfer.prefetch_depth
            if future < len(self.layers):
                self.transfer.schedule(future)

    def observe(self, layer, **tensors):
        if self.verify and layer in (0, len(self.layers) // 2, len(self.layers) - 1):
            self.probes.setdefault(layer, {}).update(
                {name: tensor.detach().cpu().clone() for name, tensor in tensors.items()}
            )

    def finish(self, output):
        if self.verify:
            self.verify.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "layers": self.probes,
                    "logits": output.next_token_logits.detach().cpu().clone(),
                    "logical_positions": self.plan.logical_positions if self.plan is not None else tuple(range(self.total)),
                    "compute_indices": self.compute_indices.detach().cpu(),
                },
                self.verify / f"{self.mode}.pt",
            )

    def close(self):
        if self.transfer:
            self.transfer.close()
            transfer = self.transfer.metrics()
        else:
            transfer = {}
        h2d = transfer.get("h2d_bytes", self.reference_copy_calls // 2 * self.bytes_per_reused_layer)
        prefix_queries = self.reuse_start
        suffix_queries = self.total - self.reuse_stop
        causal_pairs_per_head = (
            prefix_queries * (prefix_queries + 1) // 2
            + suffix_queries * self.reuse_stop
            + suffix_queries * (suffix_queries + 1) // 2
        )
        emit_metrics(
            event="optimized_prefill", mode=self.mode, source=self.source_type,
            backend="triton" if self.mode == "region_pipeline_triton" else "flashinfer_variable_block",
            n_all=self.n_all, n_active=self.n_active, n_recompute=self.n_recompute,
            layers=len(self.layers), query_rows=self.total - self.reuse_rows, kv_length=self.total,
            causal_qk_pairs_per_head=causal_pairs_per_head,
            seed_region_ids=list(self.plan.seed_region_ids) if self.plan is not None else [],
            region_ids=list(self.plan.region_ids) if self.plan is not None else [],
            expand_hops=self.plan.expand_hops if self.plan is not None else 0,
            h2d_bytes=h2d, h2d_index_bytes=self.h2d_index_bytes,
            copy_calls=2 * transfer.get("scheduled_layers", 0) + self.reference_copy_calls + self.index_copy_calls,
            sync_calls=self.reference_copy_calls + self.index_copy_calls + transfer.get("host_wait_count", 0)
                + transfer.get("close_wait_count", int(bool(transfer))),
            cpu_lookup_ms=self.lookup_ms, reference_gather_ms=self.reference_gather_ms,
            reference_copy_ms=self.reference_copy_ms, pinned_bytes=transfer.get("pinned_bytes", 0),
            gpu_staging_bytes=transfer.get("device_buffer_bytes", 0), source_cpu_bytes=self.source_cpu_bytes,
            gpu_kv_bytes=self.total * len(self.layers) * sum(buffer.shape[-1] * buffer.element_size() for buffer in self.scratch),
            gpu_restore_scratch_bytes=sum(buffer.numel() * buffer.element_size() for buffer in self.scratch),
            transfer=transfer,
        )
