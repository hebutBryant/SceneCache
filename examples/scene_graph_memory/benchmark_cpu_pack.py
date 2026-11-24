"""Compare CPU packing layouts for the real source KV bank and saved visual selections."""

import argparse
from itertools import pairwise
import json
from pathlib import Path

import torch
from torch.utils.benchmark import Timer

from sglang.srt.mem_cache.scene_graph_memory.selection import _selection
from sglang.srt.mem_cache.scene_graph_memory.store import load_memory


def runs(indices):
    boundaries = [0]
    boundaries.extend(i for i, (left, right) in enumerate(pairwise(indices), 1) if right != left + 1)
    boundaries.append(len(indices))
    return [
        (slice(begin, end), slice(indices[begin], indices[begin] + end - begin))
        for begin, end in pairwise(boundaries)
    ]


def compare_pairs(pair, indices, threads):
    offsets = runs(indices)
    index = torch.tensor(indices)
    output = tuple(torch.empty((len(indices), *src.shape[1:]), dtype=src.dtype, pin_memory=True) for src in pair)

    def gather():
        for destination, source in zip(output, pair):
            torch.index_select(source, 0, index, out=destination)

    def copy_runs():
        for destination, source in zip(output, pair):
            for target, origin in offsets:
                destination[target].copy_(source[origin])

    gather()
    expected = tuple(tensor.clone() for tensor in output)
    copy_runs()
    assert all(torch.equal(actual, reference) for actual, reference in zip(output, expected))
    return {
        "runs": len(offsets),
        "pair_bytes": sum(tensor.numel() * tensor.element_size() for tensor in output),
        **{
            name: Timer("fn()", globals={"fn": fn}, num_threads=threads).blocked_autorange(
                min_run_time=0.25
            ).median * 1e6
            for name, fn in (("index_select_us", gather), ("contiguous_runs_us", copy_runs))
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-dir", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--recompute-ratio", required=True, type=float)
    args = parser.parse_args()
    if not 0 <= args.recompute_ratio < 1:
        raise ValueError("Expected a recompute ratio in [0, 1)")
    torch.set_num_threads(args.threads)
    memory = load_memory(args.memory_dir)
    middle_layer = len(memory.source_kv) // 2
    cases = json.loads(args.cases.read_text())
    results = []
    for case in cases:
        _, _, _, active = _selection(str(args.memory_dir), json.dumps(case).encode())
        recompute = max(1, int(args.recompute_ratio * len(active)))
        if recompute == len(active):
            continue
        results.append({
            "case": case["id"], "n_active": len(active), "n_recompute": recompute,
            **compare_pairs(memory.source_kv[middle_layer], active[recompute:], args.threads),
        })
    args.output.write_text(json.dumps({
        "threads": args.threads, "layer": middle_layer,
        "memory_fingerprint": memory.fingerprint, "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
