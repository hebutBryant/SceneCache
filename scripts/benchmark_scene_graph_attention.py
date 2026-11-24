"""Microbenchmark compact Triton GQA against preplanned FlashInfer.

Use synthetic compact tensors or pass --probe-file with an actual same-selection
prefill capture. Timings exclude model execution, packing, route construction,
transfers, planning and JIT compilation.
"""

import argparse
import json

import torch
import triton.testing
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
from flashinfer.sparse import VariableBlockSparseAttentionWrapper

from sglang.srt.mem_cache.scene_graph_memory.triton_attention import (
    torch_attention_reference,
    triton_attention,
)


def median_us(fn, rep_ms):
    return triton.testing.do_bench_cudagraph(fn, rep=rep_ms, return_mode="median") * 1000


def measure(q_len, k_len, rep_ms):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(103)
    query = torch.randn(q_len, 32, 128, device=device, dtype=torch.bfloat16, generator=generator)
    key = torch.randn(k_len, 8, 128, device=device, dtype=torch.bfloat16, generator=generator)
    value = torch.randn(key.shape, device=device, dtype=key.dtype, generator=generator)
    # Strictly increasing positions model grouped query rows and the gathered
    # original frame. Their gaps need not line up with the compact K/V indices.
    key_positions = torch.arange(k_len, device=device, dtype=torch.int32) * 3
    query_positions = torch.linspace(0, k_len * 3 - 1, q_len, device=device).to(torch.int32)
    if q_len == 1:
        query_positions.fill_(k_len * 3 - 1)
    visibility = key_positions[None, :] <= query_positions[:, None]
    workspace = torch.empty(64 * 1024 * 1024, device=device, dtype=torch.uint8)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD", backend="fa2")
    wrapper.plan(
        torch.tensor([0, q_len], device=device, dtype=torch.int32),
        torch.tensor([0, k_len], device=device, dtype=torch.int32),
        32, 8, 128, custom_mask=visibility, q_data_type=torch.bfloat16,
    )

    def run_triton():
        return triton_attention(
            query, key, value, query_positions=query_positions, key_positions=key_positions,
            key_positions_sorted=True,
        )

    def run_flashinfer():
        return wrapper.run(query, key, value)

    reference = torch_attention_reference(
        query, key, value, query_positions=query_positions, key_positions=key_positions
    )
    triton_output = run_triton()
    flashinfer_output = run_flashinfer()
    torch.testing.assert_close(triton_output, reference, atol=0.012, rtol=0.012)
    torch.testing.assert_close(flashinfer_output, reference, atol=0.012, rtol=0.012)

    return {
        "q_rows": q_len, "kv_rows": k_len, "q_heads": 32, "kv_heads": 8, "head_dim": 128,
        "triton_us": median_us(run_triton, rep_ms),
        "flashinfer_fa2_us": median_us(run_flashinfer, rep_ms),
        "triton_max_abs_error": (triton_output.float() - reference.float()).abs().max().item(),
        "flashinfer_max_abs_error": (flashinfer_output.float() - reference.float()).abs().max().item(),
    }


def profile_device_ops(fn, calls):
    """Average CUDA operation times per invocation, separate from wrapper events."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    times = {}
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            times[event.name] = times.get(event.name, 0.0) + event.device_time_total
    if not times:
        raise RuntimeError("CUDA profiler captured no device operations")
    return {name: total / calls for name, total in times.items()}


def measure_probe(path, layer_id, rep_ms, profile_calls):
    probe = torch.load(path, map_location="cpu", weights_only=True)
    tensors = probe["layers"][layer_id]
    query, key, value = (
        tensors[name].reshape(tensors[name].shape[0], -1, 128).cuda()
        for name in ("query", "key", "value")
    )
    q_len, q_heads, head_dim = query.shape
    k_len, kv_heads, _ = key.shape
    logical_positions = torch.as_tensor(probe["logical_positions"], dtype=torch.int32)
    compute_indices = probe["compute_indices"].long()
    if logical_positions.numel() != k_len or compute_indices.numel() != q_len:
        raise ValueError("Probe logical positions do not match query and key lengths")
    if not torch.all(logical_positions.diff() > 0):
        raise ValueError("Expected strictly ordered compact key positions")
    query_positions = logical_positions.index_select(0, compute_indices).cuda()
    key_positions = logical_positions.cuda()
    gaps = torch.nonzero(compute_indices.diff() != 1).flatten().tolist()
    if compute_indices[0] != 0 or compute_indices[-1] != k_len - 1 or len(gaps) > 1:
        raise ValueError("Expected prefix and suffix queries with at most one intervening reuse block")
    if gaps:
        prefix = gaps[0] + 1
        suffix_start = compute_indices[prefix].item()
        block_rows = [prefix, q_len - prefix]
        block_columns = [prefix, suffix_start - prefix, k_len - suffix_start]
        block_mask = [[True, False, False], [True, True, True]]
    else:
        block_rows, block_columns, block_mask = [q_len], [k_len], [[True]]

    workspace = torch.empty(128 * 1024**2, device=query.device, dtype=torch.uint8)
    wrapper = VariableBlockSparseAttentionWrapper(workspace)
    wrapper.plan(
        torch.tensor(block_mask, device=query.device, dtype=torch.bool).repeat(kv_heads, 1, 1),
        torch.tensor([block_rows], device=query.device, dtype=torch.int32).repeat(kv_heads, 1),
        torch.tensor([block_columns], device=query.device, dtype=torch.int32).repeat(kv_heads, 1),
        q_heads, kv_heads, head_dim, causal=True, q_data_type=query.dtype, kv_data_type=key.dtype,
    )

    def run_triton():
        return triton_attention(
            query, key, value, query_positions=query_positions, key_positions=key_positions,
            key_positions_sorted=True,
        )

    def run_flashinfer():
        return wrapper.run(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)).transpose(0, 1).contiguous()

    reference = torch_attention_reference(
        query, key, value, query_positions=query_positions, key_positions=key_positions
    )
    triton_output = run_triton()
    flashinfer_output = run_flashinfer()
    torch.testing.assert_close(triton_output, reference, atol=0.012, rtol=0.012)
    torch.testing.assert_close(flashinfer_output, reference, atol=0.012, rtol=0.012)
    result = {
        "probe": str(path), "layer": layer_id,
        "q_rows": q_len, "kv_rows": k_len, "q_heads": q_heads, "kv_heads": kv_heads,
        "head_dim": head_dim, "row_blocks": block_rows, "key_blocks": block_columns,
        "triton_wrapper_us": median_us(run_triton, rep_ms),
        "flashinfer_variable_block_wrapper_us": median_us(run_flashinfer, rep_ms),
    }
    for label, output in (("triton", triton_output), ("flashinfer", flashinfer_output)):
        delta = (output.float() - reference.float()).abs()
        result[label + "_vs_fp32_oracle"] = {
            "max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
            "rmse": delta.square().mean().sqrt().item(),
        }
        result[label + "_bitwise_matches_capture"] = torch.equal(output.cpu(), tensors["attention"].reshape(query.shape))
    if profile_calls:
        result["profile_calls"] = profile_calls
        for label, operation in (("triton", run_triton), ("flashinfer", run_flashinfer)):
            times = profile_device_ops(operation, profile_calls)
            result[label + "_device_ops_us_per_call"] = times
            result[label + "_device_total_us_per_call"] = sum(times.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", help="Q rows and K/V rows, e.g. --shape 128,882")
    parser.add_argument("--rep-ms", type=int, default=10)
    parser.add_argument("--probe-file", help="Actual compact .pt probe with query, K/V and positions")
    parser.add_argument("--layer", type=int, default=0, help="Captured layer to replay")
    parser.add_argument("--profile-calls", type=int, default=40, help="CUDA operation profile repetitions with --probe-file")
    args = parser.parse_args()
    if args.rep_ms <= 0 or args.profile_calls < 0:
        parser.error("Invalid measurement repetitions")
    if args.probe_file and args.shape:
        parser.error("Choose --probe-file or --shape")
    torch.set_float32_matmul_precision("highest")
    if args.probe_file:
        result = measure_probe(args.probe_file, args.layer, args.rep_ms, args.profile_calls)
        print(json.dumps({
            "device": torch.cuda.get_device_name(),
            "measurement": "warm CUDA graph wrapper latency and separate CUDA operation profile",
            "results": [result],
        }, indent=2))
        return
    shapes = args.shape or ["1,882", "96,256", "256,882", "512,882"]
    results = []
    for shape in shapes:
        q_len, k_len = map(int, shape.split(","))
        if min(q_len, k_len, args.rep_ms) <= 0:
            parser.error("shape dimensions and repetitions must be positive")
        results.append(measure(q_len, k_len, args.rep_ms))
    print(json.dumps({
        "device": torch.cuda.get_device_name(),
        "measurement": "warm CUDA graph, prepacked K/V and preplanned FlashInfer mask",
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
