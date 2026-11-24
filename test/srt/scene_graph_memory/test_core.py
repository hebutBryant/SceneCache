"""CPU-only correctness checks for the Scene Graph K/V memory format."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

from sglang.srt.mem_cache.scene_graph_memory import store
from sglang.srt.mem_cache.scene_graph_memory.graph import mask_contact_kind
from sglang.srt.mem_cache.scene_graph_memory.mapping import (
    assemble_memory,
    build_token_provenance,
    mask_token_membership,
)
from sglang.srt.mem_cache.scene_graph_memory.schema import (
    ImageGeometry,
    MaskInput,
    SourceKVMetadata,
)


def pixel_mask(shape, *points):
    result = np.zeros(shape, dtype=np.bool_)
    for x, y in points:
        result[y, x] = True
    return result


def source_fixture(threshold=0.0):
    geometry = ImageGeometry((9, 6), (9, 6), (1, 6, 9), 1, 3)
    image = Image.new("RGB", geometry.original_size, "#076279")
    mask_a = pixel_mask((6, 9), (0, 0), (8, 0))
    mask_b = pixel_mask((6, 9), (1, 0), (8, 5))
    mask_c = pixel_mask((6, 9), (7, 5))
    mask_d = pixel_mask((6, 9), (0, 3))
    empty = pixel_mask((6, 9))
    source_kv = {
        layer: (
            torch.arange(18, dtype=dtype).reshape(6, 3) + 23 * layer,
            torch.arange(24, dtype=dtype).reshape(6, 2, 2) + 49 * layer,
        )
        for layer, dtype in enumerate((torch.bfloat16, torch.float16, torch.float32))
    }
    metadata = SourceKVMetadata(
        key_representation_stage="raw_pre_qknorm_pre_rope",
        value_representation_stage="raw_projection",
        source_token_ids=[1, 2, 3, 3, 3, 3, 3, 3, 4, 5],
        chat_template="test-only canonical prefix",
        qk_norm={"enabled": True},
        rope={"rope_theta": 1000000},
        details={"num_hidden_layers": 3},
    )
    return assemble_memory(
        image,
        geometry,
        range(2, 8),
        [(2, 2 + index // 3, 2 + index % 3) for index in range(6)],
        [
            MaskInput(mask_d, label="d"),
            MaskInput(mask_a, score=0.8, label="a", label_source="fixture"),
            MaskInput(empty, label="empty"),
            MaskInput(mask_c, label="c"),
            MaskInput(mask_b, label="b"),
        ],
        source_kv,
        metadata,
        {"model": "unit-model", "processor": "unit-processor"},
        {"fixture": "synthetic CPU unit test"},
        overlap_threshold=threshold,
    )


class GeometryTest(unittest.TestCase):
    def test_contact_uses_unit_cells_and_one_dilation(self):
        origin = pixel_mask((7, 7), (2, 2))
        cases = [
            ("overlap", pixel_mask((7, 7), (2, 2)), "overlap"),
            ("horizontal", pixel_mask((7, 7), (3, 2)), "touch"),
            ("vertical", pixel_mask((7, 7), (2, 3)), "touch"),
            ("diagonal", pixel_mask((7, 7), (3, 3)), "touch"),
            ("one-pixel gap", pixel_mask((7, 7), (4, 2)), None),
            ("far", pixel_mask((7, 7), (6, 6)), None),
            ("empty", pixel_mask((7, 7)), None),
            ("empty center of touching bbox", pixel_mask((7, 7), (0, 2), (6, 2)), None),
        ]
        for label, mask, expected in cases:
            with self.subTest(label=label):
                self.assertEqual(mask_contact_kind(origin, mask), expected)
                self.assertEqual(mask_contact_kind(mask, origin), expected)
                overlap_only = "overlap" if expected == "overlap" else None
                self.assertEqual(mask_contact_kind(origin, mask, "overlap_only"), overlap_only)
        # Edges of the image still support corner contact, no wrap across the image.
        origin = pixel_mask((3, 3), (0, 0))
        self.assertEqual(mask_contact_kind(origin, pixel_mask((3, 3), (1, 1))), "touch")
        self.assertIsNone(mask_contact_kind(origin, pixel_mask((3, 3), (2, 0))))

    def test_resize_merge_and_subpixel_coverage(self):
        geometry = ImageGeometry((7, 3), (12, 8), (1, 4, 6), 2, 2)
        positions = [(2, 2 + index // 3, 2 + index % 3) for index in range(6)]
        tokens = build_token_provenance(geometry, range(1, 7), positions)
        self.assertEqual(tokens[4].source_index, 5)
        self.assertEqual(tokens[4].processed_xyxy, (4, 4, 8, 8))
        self.assertEqual(tokens[4].original_xyxy, (7 / 3, 1.5, 14 / 3, 3.0))
        self.assertEqual(tokens[4].mrope_position, (2, 3, 3))
        one_pixel = pixel_mask((3, 7), (2, 1))
        indices, weights = mask_token_membership(one_pixel, geometry)
        self.assertEqual(indices, (0, 1, 3, 4))
        np.testing.assert_allclose(weights, [1 / 21, 2 / 21, 1 / 21, 2 / 21])
        self.assertEqual(mask_token_membership(one_pixel, geometry, 0.06)[0], (1, 4))
        self.assertEqual(
            mask_token_membership(np.ones((3, 7), dtype=np.bool_), geometry)[1],
            (1.0,) * 6,
        )
        self.assertEqual(mask_token_membership(pixel_mask((3, 7)), geometry), ((), ()))

    def test_rejects_missing_tokens_and_unimplemented_geometry(self):
        geometry = ImageGeometry((7, 3), (12, 8), (1, 4, 6), 2, 2)
        with self.assertRaisesRegex(ValueError, "Expected 6 post-merge"):
            build_token_provenance(geometry, range(5), [(0, 0, 0)] * 5)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            build_token_provenance(geometry, [0, 1, 2, 3, 4, 8], [(0, 0, 0)] * 6)
        with self.assertRaisesRegex(ValueError, "resize-only"):
            replace(geometry, steps=[{"operation": "crop"}]).validate()


class MemoryTest(unittest.TestCase):
    def test_regions_refs_union_hops_and_gather(self):
        memory = source_fixture()
        self.assertEqual(
            [(node.region_id, node.label, node.token_indices) for node in memory.regions],
            [("r0001", "a", (0, 2)), ("r0002", "b", (0, 5)),
             ("r0003", "c", (5,)), ("r0004", "d", (3,))],
        )
        self.assertEqual(memory.regions[0].bbox_xyxy, (0, 0, 9, 1))
        self.assertEqual(memory.regions[0].area, 2)
        self.assertEqual(memory.unassigned_token_indices, (1, 4))
        self.assertEqual(memory.segmentation["mapping_filtered_masks"][0]["reason"], "empty_mask")
        self.assertEqual(memory.source_prefix_token_ids, [1, 2])
        self.assertEqual(memory.non_visual_source_indices, (0, 1, 8, 9))
        self.assertEqual(memory.get_neighbors("r0001"), ("r0002",))
        self.assertEqual(memory.get_neighbors("r0004"), ())
        self.assertEqual(
            [(edge.region_a, edge.region_b, edge.contact_kind) for edge in memory.edges],
            [("r0001", "r0002", "touch"), ("r0002", "r0003", "touch")],
        )
        detail = memory.get_region_kv("r0001")
        self.assertEqual(detail.visual_indices, (0, 2))
        self.assertEqual(detail.source_indices, (2, 4))
        self.assertEqual(detail.mrope_positions, ((2, 2, 2), (2, 2, 4)))
        for layer, (keys, values) in detail.layers.items():
            source_keys, source_values = memory.source_kv[layer]
            self.assertTrue(torch.equal(keys, torch.stack([source_keys[0], source_keys[2]])))
            self.assertTrue(torch.equal(values, torch.stack([source_values[0], source_values[2]])))
            self.assertEqual(keys.dtype, source_keys.dtype)
            self.assertNotEqual(keys.data_ptr(), source_keys.data_ptr())
        union = memory.get_regions_kv(["r0002", "r0001", "r0002"], layers=[2, 0, 2])
        self.assertEqual(tuple(union.layers), (0, 2))
        self.assertEqual(union.visual_indices, (0, 2, 5))
        self.assertEqual(memory.get_regions_kv(["r0001"], 1).region_ids, ("r0001", "r0002"))
        self.assertEqual(memory.get_regions_kv(["r0001"], 2).region_ids, ("r0001", "r0002", "r0003"))
        self.assertEqual(
            memory.get_regions_kv([node.region_id for node in memory.regions]).visual_indices,
            (0, 2, 3, 5),
        )
        with self.assertRaises(KeyError):
            memory.get_regions_kv(["invalid"])
        with self.assertRaises(ValueError):
            memory.get_regions_kv(["r0001"], expand_hops=-1)
        bad_region = replace(memory.regions[0], token_indices=(0, 7))
        with self.assertRaisesRegex(ValueError, "invalid visual token indices"):
            replace(memory, regions=[bad_region, *memory.regions[1:]]).validate()

    def test_round_trip_stale_fingerprint_and_independent_export(self):
        original = source_fixture()
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "memory"
            self.assertEqual(store.save_memory(original, target), target)
            self.assertEqual(store.save_memory(original, target), target)
            reopened = store.load_memory(target, expected_fingerprint=original.fingerprint)
            for layer, pair in original.source_kv.items():
                for before, after in zip(pair, reopened.source_kv[layer]):
                    self.assertEqual(before.shape, after.shape)
                    self.assertEqual(before.dtype, after.dtype)
                    self.assertTrue(torch.equal(before, after))
            self.assertEqual(reopened.tokens, original.tokens)
            self.assertEqual(reopened.edges, original.edges)
            self.assertEqual(reopened.fingerprint, original.fingerprint)
            for region in reopened.regions:
                np.testing.assert_array_equal(reopened.masks[region.region_id], original.masks[region.region_id])
            changed_inputs = [
                replace(original, fingerprints={**original.fingerprints, "model": "different"}),
                replace(original, fingerprints={**original.fingerprints, "processor": "different"}),
                replace(original, kv_metadata=replace(original.kv_metadata, source_token_ids=[99, *original.kv_metadata.source_token_ids[1:]])),
                replace(original, kv_metadata=replace(original.kv_metadata, key_representation_stage="after_rope")),
                source_fixture(threshold=0.9),
            ]
            for altered in changed_inputs:
                with self.subTest(fingerprint=altered.fingerprint):
                    self.assertNotEqual(original.fingerprint, altered.fingerprint)
                    with self.assertRaises(FileExistsError):
                        store.save_memory(altered, target)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                store.load_memory(target, expected_fingerprint="bad")
            del original
            out_file = reopened.export_region("r0001", Path(temp) / "region.safetensors")
            selection = reopened.get_region_kv("r0001")
            del reopened
            exported = load_file(out_file, device="cpu")
            self.assertEqual(len(exported), 6)
            for layer, pair in selection.layers.items():
                self.assertTrue(torch.equal(exported[f"layer.{layer}.k"], pair[0]))
                self.assertTrue(torch.equal(exported[f"layer.{layer}.v"], pair[1]))
            with safe_open(out_file, framework="pt") as exported_file:
                metadata = json.loads(exported_file.metadata()["scene_graph_memory"])
            self.assertEqual(metadata["region_id"], "r0001")
            self.assertEqual(metadata["source_indices"], [2, 4])
            self.assertEqual(metadata["key_representation_stage"], "raw_pre_qknorm_pre_rope")

    def test_incomplete_payload_does_not_publish(self):
        def failed_serialization(tensors, output, metadata):
            Path(output).write_bytes(b"incomplete")
            raise OSError("test failure")

        memory = source_fixture()
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "memory"
            with patch.object(store, "save_file", side_effect=failed_serialization):
                with self.assertRaisesRegex(OSError, "test failure"):
                    store.save_memory(memory, target)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(temp).glob(".memory.staging-*")), [])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                store.load_memory(target)
            store.save_memory(memory, target)
            with (target / store.BANK).open("ab") as corrupt:
                corrupt.write(b"not in manifest")
            with self.assertRaisesRegex(ValueError, "payload hash or size mismatch"):
                store.load_memory(target)

    def test_high_threshold_preserves_regions_and_allows_empty_kv_slice(self):
        memory = source_fixture(threshold=0.95)
        self.assertEqual(len(memory.regions), 4)
        self.assertTrue(all(not node.token_indices for node in memory.regions))
        self.assertEqual(memory.unassigned_token_indices, tuple(range(len(memory.tokens))))
        self.assertEqual(len(memory.edges), 2)
        with tempfile.TemporaryDirectory() as temp:
            exported = memory.export_region("r0001", Path(temp) / "empty.safetensors")
            tensors = load_file(exported, device="cpu")
            self.assertEqual(tensors["layer.0.k"].shape, (0, 3))
            self.assertEqual(tensors["layer.0.k"].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
