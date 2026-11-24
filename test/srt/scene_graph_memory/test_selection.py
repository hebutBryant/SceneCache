import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.scene_graph_memory.selection import _selection, compact_inputs


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "fingerprint": "image-source",
            "geometry": {"grid_thw": [1, 4, 6]},
            "source_prefix_token_ids": [11, 12],
            "tokens": [
                {
                    "source_index": index + 2, "mrope_position": [2, y + 2, x + 2],
                    "grid_y": y, "grid_x": x,
                }
                for index, (y, x) in enumerate((y, x) for y in range(2) for x in range(3))
            ],
            "regions": [
                {"region_id": "left", "token_indices": [0, 3]},
                {"region_id": "bottom", "token_indices": [3, 5]},
            ],
            "edges": [{"region_a": "left", "region_b": "bottom"}],
            "unassigned_token_indices": [1, 2, 4],
        }
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = self.directory.name
        self.path = Path(self.root) / "selection.json"
        self.addCleanup(_selection.cache_clear)
        descriptor = patch(
            "sglang.srt.mem_cache.scene_graph_memory.selection.descriptor",
            return_value=(self.manifest, None, None),
        )
        descriptor.start()
        self.addCleanup(descriptor.stop)

    def compact(self, selection):
        self.path.write_text(json.dumps(selection))
        token_ids = torch.tensor([11, 12, *([13] * 6), 14, 15])
        positions = torch.tensor([
            [0, 0, 0], [1, 1, 1],
            *(token["mrope_position"] for token in self.manifest["tokens"]),
            [5, 5, 5], [6, 6, 6],
        ]).T
        item = FakeItem(offsets=[(2, 7)], image_grid_thw=torch.tensor([[1, 4, 6]]))
        result = compact_inputs(item, token_ids, positions, torch.tensor([[-3]]), self.root, str(self.path))
        return item, token_ids, positions, result

    def test_compaction_preserves_position_and_next_decode(self):
        item, original, original_mrope, (ids, mrope, delta) = self.compact(
            {"region_ids": ["left", "left"], "expand_hops": 1}
        )
        self.assertEqual(item.offsets, [(2, 4)])
        plan = item.visual_kv_selection
        self.assertEqual(plan.seed_region_ids, ("left",))
        self.assertEqual(plan.region_ids, ("bottom", "left"))
        self.assertEqual(plan.source_visual_indices, (0, 3, 5))
        self.assertEqual(plan.nonvisual_logical_positions, (0, 1, 8, 9))
        expected = torch.tensor([0, 1, 2, 5, 7, 8, 9])
        self.assertTrue(torch.equal(ids, original[expected]))
        self.assertTrue(torch.equal(mrope, original_mrope[:, expected]))
        self.assertEqual((delta + ids.numel()).item(), original.numel() - 3)

    def test_full_visual_selection_does_not_change_sequence(self):
        item, original, original_mrope, (ids, mrope, delta) = self.compact(
            {"region_ids": ["left", "bottom"], "include_unassigned": True}
        )
        self.assertEqual(item.visual_kv_selection.source_visual_indices, tuple(range(6)))
        self.assertTrue(torch.equal(ids, original))
        self.assertTrue(torch.equal(mrope, original_mrope))
        self.assertEqual(delta.item(), -3)

    def test_budget_and_invalid_selections(self):
        item, *_ = self.compact({"region_ids": ["left"], "active_budget": 0.5})
        self.assertEqual(len(item.visual_kv_selection.source_visual_indices), 3)
        for selection in ({}, {"region_ids": ["unknown"]}, {"region_ids": ["left"], "active_budget": 0.1}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.compact(selection)

    def test_explicit_indices_bind_to_the_bank_and_keep_source_order(self):
        selection = {
            "memory_fingerprint": "image-source",
            "region_ids": ["left"],
            "source_visual_indices": [0, 3, 4],
        }
        item, original, positions, (ids, mrope, _) = self.compact(selection)
        self.assertEqual(item.visual_kv_selection.source_visual_indices, (0, 3, 4))
        expected = torch.tensor([0, 1, 2, 5, 6, 8, 9])
        self.assertTrue(torch.equal(ids, original[expected]))
        self.assertTrue(torch.equal(mrope, positions[:, expected]))
        for invalid in (
            {"memory_fingerprint": "another-image"},
            {"source_visual_indices": [0, 0, 3]},
            {"source_visual_indices": [3, 0]},
            {"source_visual_indices": [0, 6]},
            {"source_visual_indices": [True, 3]},
            {"expand_hops": 1},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.compact({**selection, **invalid})


class FakeItem(SimpleNamespace):
    def __setitem__(self, key, value):
        setattr(self, key, value)
