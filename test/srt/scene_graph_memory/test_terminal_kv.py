"""Check storage allocation, persistence and positions using native HF cache APIs."""

import gc
import tempfile
import unittest
import weakref
from pathlib import Path

import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import DynamicCache

from sglang.srt.mem_cache.scene_graph_memory.terminal_kv import (
    BANK,
    cache_stats,
    compact_cache,
    continuation_inputs,
    copy_cache,
    load_cache,
    save_cache,
)


def fixture():
    rows = torch.arange(40, dtype=torch.float32).reshape(1, 2, 5, 4)
    return DynamicCache([(rows + 10, rows + 20), (rows + 30, rows + 40)])


class TerminalCacheTest(unittest.TestCase):
    def test_compact_is_exact_and_does_not_retain_source_storage(self):
        source = fixture()
        source_rows = [
            (weakref.ref(layer.keys), weakref.ref(layer.values))
            for layer in source.layers
        ]
        compact = compact_cache(source, [0, 2, 4])
        source_stats = cache_stats(source)
        compact_stats = cache_stats(compact)
        self.assertEqual(compact_stats["sequence_length"], 3)
        self.assertEqual(compact_stats["storage_bytes"], compact_stats["logical_bytes"])
        self.assertEqual(
            compact_stats["storage_bytes"] * 5, source_stats["storage_bytes"] * 3
        )
        for layer, source_layer in zip(compact.layers, source.layers):
            for tensor, original in (
                (layer.keys, source_layer.keys),
                (layer.values, source_layer.values),
            ):
                self.assertTrue(torch.equal(tensor, original[:, :, [0, 2, 4], :]))
                self.assertNotEqual(
                    tensor.untyped_storage().data_ptr(),
                    original.untyped_storage().data_ptr(),
                )
        del source, source_layer, original
        gc.collect()
        self.assertTrue(all(ref() is None for pair in source_rows for ref in pair))
        self.assertEqual(compact.layers[0].keys[0, 0, :, 0].tolist(), [10, 18, 26])

    def test_stats_distinguish_views_and_actual_owned_bytes(self):
        source = fixture()
        # Only one tensor per layer is referenced on purpose. Each K and V
        # reports its view bytes, while unique storage reflects actual memory.
        for layer in source.layers:
            layer.keys = layer.keys[:, :, 1:4, :]
            layer.values = layer.keys
        stats = cache_stats(source)
        self.assertEqual(stats["storage_count"], 2)
        self.assertEqual(stats["layers"][0]["key"]["storage_offset"], 4)
        self.assertEqual(stats["layers"][0]["key"]["logical_bytes"], 96)
        self.assertEqual(stats["layers"][0]["key"]["storage_bytes"], 160)
        compact = compact_cache(source, [0, 2])
        self.assertEqual(cache_stats(compact)["storage_count"], 4)

    def test_positions_are_not_renumbered_with_physical_cache(self):
        compact = compact_cache(fixture(), torch.tensor([0, 2, 4]))
        mrope = torch.tensor([[[30, 31]], [[31, 32]], [[32, 33]]], dtype=torch.long)
        inputs = continuation_inputs(compact, mrope)
        self.assertEqual(inputs["cache_position"].tolist(), [3, 4])
        self.assertEqual(inputs["attention_mask"].tolist(), [[1] * 5])
        self.assertTrue(torch.equal(inputs["position_ids"], mrope))
        for rows in (
            [],
            [-1, 0],
            [2, 0],
            [2, 2],
            [0, 5],
            [False, 3],
            torch.tensor([0.0, 1.0]),
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                compact_cache(fixture(), rows)

    def test_independent_cpu_offload_and_disk_roundtrip(self):
        compact = compact_cache(fixture(), [0, 3])
        cpu_cache = copy_cache(compact, device="cpu")
        source_stats, host_stats = cache_stats(compact), cache_stats(cpu_cache)
        self.assertNotEqual(
            host_stats["layers"][0]["key"]["storage_ptr"],
            source_stats["layers"][0]["key"]["storage_ptr"],
        )
        meta = {
            "retained_rows": [0, 3],
            "logical_prefix_tokens": 5,
            "model_sha256": "fixture",
        }
        with tempfile.TemporaryDirectory() as temp:
            target = save_cache(Path(temp) / "terminal", cpu_cache, metadata=meta)
            restored, returned_meta = load_cache(
                target, device="cpu", expected_metadata=meta
            )
            self.assertEqual(returned_meta, meta)
            for expected, actual in zip(compact.layers, restored.layers):
                self.assertTrue(
                    torch.equal(
                        expected.keys.view(torch.uint8), actual.keys.view(torch.uint8)
                    )
                )
                self.assertTrue(
                    torch.equal(
                        expected.values.view(torch.uint8),
                        actual.values.view(torch.uint8),
                    )
                )
            with self.assertRaisesRegex(ValueError, "compatibility metadata mismatch"):
                load_cache(
                    target,
                    device="cpu",
                    expected_metadata={**meta, "retained_rows": [1, 3]},
                )
            with self.assertRaises(FileExistsError):
                save_cache(target, cpu_cache, metadata=meta)
            with (target / BANK).open("ab") as bank:
                bank.write(b"wrong")
            with self.assertRaisesRegex(ValueError, "payload hash or length mismatch"):
                load_cache(target, device="cpu", expected_metadata=meta)

    @torch.inference_mode()
    def test_tiny_qwen_continuation_matches_masked_dense_prefix(self):
        config = Qwen2_5_VLConfig(
            vision_config={
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 24,
                "num_heads": 2,
                "patch_size": 2,
                "out_hidden_size": 32,
                "window_size": 8,
                "fullatt_block_indexes": [0],
            },
            text_config={
                "vocab_size": 70,
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
            },
            vision_start_token_id=60,
            vision_end_token_id=62,
            image_token_id=61,
            video_token_id=63,
            attn_implementation="sdpa",
        )
        torch.manual_seed(12)
        model = Qwen2_5_VLForConditionalGeneration(config).eval()
        prefix = torch.tensor([[1, 2, 60, 61, 61, 61, 61, 62, 3, 4]])
        grid = torch.tensor([[1, 4, 4]])
        pixel_values = torch.randn(16, 3 * 2 * 2 * 2)
        prefill = model(
            input_ids=prefix,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            attention_mask=torch.ones_like(prefix),
            use_cache=True,
        )
        full = prefill.past_key_values
        indices = [0, 1, 2, 4, 6, 7, 8, 9]
        subset = compact_cache(full, indices)
        self.assertEqual(full.get_seq_length(), 10)
        self.assertEqual(subset.get_seq_length(), len(indices))
        suffix = torch.tensor([[5, 6, 7]])
        start = prefix.shape[-1] + prefill.rope_deltas.item()
        pos = (
            torch.arange(start, start + suffix.shape[-1])
            .view(1, 1, -1)
            .expand(3, 1, -1)
        )

        reference_args = continuation_inputs(full, pos)
        reference_args["attention_mask"][0, [3, 5]] = 0
        reference = model(
            input_ids=suffix, past_key_values=full, use_cache=True, **reference_args
        )
        result = model(
            input_ids=suffix,
            past_key_values=subset,
            use_cache=True,
            **continuation_inputs(subset, pos),
        )
        torch.testing.assert_close(
            result.logits, reference.logits, atol=1e-6, rtol=1e-6
        )
        self.assertEqual(
            result.past_key_values.get_seq_length(), len(indices) + suffix.shape[-1]
        )
