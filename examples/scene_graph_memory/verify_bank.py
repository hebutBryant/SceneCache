"""Verify persisted region slices against all layers of their source bank on CPU."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from sglang.srt.mem_cache.scene_graph_memory import load_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("memory_dir", type=Path)
    parser.add_argument("--export", type=Path, help="Verify an optional export-region file too")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    memory = load_memory(args.memory_dir)

    def verify_slice(selection, expected):
        assert selection.visual_indices == expected
        assert selection.source_indices == tuple(memory.tokens[i].source_index for i in expected)
        assert selection.mrope_positions == tuple(memory.tokens[i].mrope_position for i in expected)
        assert set(selection.layers) == set(memory.source_kv)
        index = torch.tensor(expected, dtype=torch.long)
        for layer, values in selection.layers.items():
            for tensor, source in zip(values, memory.source_kv[layer], strict=True):
                assert tensor.dtype == source.dtype
                assert torch.equal(tensor, source.index_select(0, index))

    assigned = set()
    for region in memory.regions:
        verify_slice(memory.get_region_kv(region.region_id), region.token_indices)
        assigned.update(region.token_indices)
    if memory.regions:
        union = memory.get_regions_kv([region.region_id for region in memory.regions])
        verify_slice(union, tuple(sorted(assigned)))
        first = memory.regions[0].region_id
        group = {first, *memory.get_neighbors(first)}
        expanded = tuple(sorted({
            index for region in memory.regions if region.region_id in group for index in region.token_indices
        }))
        verify_slice(memory.get_regions_kv([first], expand_hops=1), expanded)
    unassigned = set(memory.unassigned_token_indices)
    assert not (assigned & unassigned)
    assert assigned | unassigned == set(range(len(memory.tokens)))
    checked_export = None
    if args.export:
        with safe_open(args.export, framework="pt", device="cpu") as exported:
            metadata = json.loads(exported.metadata()["scene_graph_memory"])
            assert metadata["memory_fingerprint"] == memory.fingerprint
            assert metadata["image_hash"] == memory.image_hash
            selection = memory.get_region_kv(metadata["region_id"])
            assert metadata["source_indices"] == list(selection.source_indices)
            assert metadata["mrope_positions"] == [list(pos) for pos in selection.mrope_positions]
            for layer, (key, value) in selection.layers.items():
                assert torch.equal(exported.get_tensor(f"layer.{layer}.k"), key)
                assert torch.equal(exported.get_tensor(f"layer.{layer}.v"), value)
            checked_export = metadata["region_id"]
    print(json.dumps({
        "status": "passed", "fingerprint": memory.fingerprint,
        "regions_verified": len(memory.regions), "layers_verified": len(memory.source_kv),
        "visual_tokens": len(memory.tokens), "assigned": len(assigned), "unassigned": len(unassigned),
        "export_verified": checked_export,
    }, indent=2))


if __name__ == "__main__":
    main()
