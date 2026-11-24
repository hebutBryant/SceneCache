"""CPU diagnostic with the local Mini-o3 processor and tiny random Qwen weights."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from sglang.srt.mem_cache.scene_graph_memory.terminal_kv import load_cache, save_cache

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/scene_graph_memory"))
from terminal_runtime import MiniO3Runtime, PrefixMismatch

MODEL = Path("/home/hdd3/yaoyf/Mini-o3/Mini-o3-7B-v1")
PIXELS = {"min_pixels": 3136, "max_pixels": 12544}
MARKER = "<|vision_start|><|image_pad|><|vision_end|>"


@unittest.skipUnless(
    (MODEL / "tokenizer.json").is_file(), "Mini-o3 tokenizer not installed locally"
)
class TerminalRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
        config = Qwen2_5_VLConfig(
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 4,
                "out_hidden_size": 64,
                "fullatt_block_indexes": [0],
            },
            text_config={
                "vocab_size": 152064,
                "hidden_size": 64,
                "intermediate_size": 96,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]},
            },
            pad_token_id=151643,
            eos_token_id=151645,
            bos_token_id=151643,
            attn_implementation="sdpa",
        )
        torch.manual_seed(2026)
        cls.model = Qwen2_5_VLForConditionalGeneration(config).eval()

    def setUp(self):
        self.runtime = MiniO3Runtime(self.model, self.processor, device="cpu", **PIXELS)
        self.image = Image.new("RGB", (113, 85), "blue")
        self.crop = Image.new("RGB", (82, 71), "red")
        self.messages = [
            {"role": "system", "content": "Locate visual evidence."},
            {"role": "user", "content": MARKER + "\nWhat?"},
        ]
        self.first = self.runtime.prepare(self.messages, [self.image])
        self.action = '<grounding>{"bbox_2d":[0.1,0.2,0.8,0.9],"source":"original_image"}</grounding>'
        self.action_ids = self.processor.tokenizer.encode(
            self.action, add_special_tokens=False
        )
        self.observed = self.messages + [
            {"role": "assistant", "content": self.action},
            {"role": "user", "content": "Observation follows:\n" + MARKER},
        ]

    @torch.inference_mode()
    def test_processor_and_native_greedy_and_fixed_sampling(self):
        normal = self.processor(
            text=[
                self.processor.apply_chat_template(
                    self.messages, tokenize=False, add_generation_prompt=True
                )
            ],
            images=[self.image],
            return_tensors="pt",
            images_kwargs=PIXELS,
        )
        torch.testing.assert_close(
            normal.input_ids, self.first.input_ids, rtol=0, atol=0
        )
        torch.testing.assert_close(
            normal.image_grid_thw, self.first.image_grid_thw, rtol=0, atol=0
        )
        torch.testing.assert_close(
            normal.pixel_values, self.first.image_patches[0], rtol=0, atol=0
        )
        for temperature in (0.0, 1.0):
            torch.manual_seed(117)
            native = self.model.generate(
                **normal,
                max_new_tokens=3,
                do_sample=temperature > 0,
                use_cache=True,
                stop_strings=["</grounding>", "</answer>"],
                tokenizer=self.processor.tokenizer,
            )
            result = self.runtime.generate(
                self.first,
                max_new_tokens=3,
                temperature=temperature,
                generator=torch.Generator().manual_seed(117),
            )
            self.assertEqual(
                tuple(native[0, normal.input_ids.shape[-1] :].tolist()),
                result.generated_ids,
            )

    def test_grounding_boundary_and_exact_append(self):
        first = self.runtime.teacher_forced(self.first, targets=self.action_ids)
        second = self.runtime.prepare(
            self.observed,
            [self.image, self.crop],
            assistant_token_ids={2: first.generated_ids},
        )
        self.assertEqual(
            self.runtime.verify_prefix(second, first.state), first.state.logical_length
        )
        self.assertEqual(
            second.input_ids[
                0, self.first.logical_length : first.state.logical_length + 1
            ].tolist(),
            self.action_ids,
        )
        cold = self.runtime.teacher_forced(second, targets=[42, 43, 44])
        warm = self.runtime.teacher_forced(
            second, state=first.state, targets=[42, 43, 44]
        )
        self.assertEqual(warm.metrics["vision_input_count"], 1)
        self.assertEqual(cold.metrics["vision_input_count"], 2)
        self.assertEqual(warm.metrics["processor_cache_hits"], 1)
        torch.testing.assert_close(
            torch.stack(cold.logits), torch.stack(warm.logits), atol=1e-6, rtol=1e-5
        )
        self.assertEqual(warm.state.logical_length, warm.state.cache.get_seq_length())
        self.assertTrue(first.state.consumed)

    def test_incompatible_image_and_positions_rejected(self):
        state = self.runtime.teacher_forced(self.first, targets=self.action_ids).state
        # Identical image dimensions and grid do not make old pixels reusable.
        different = Image.new("RGB", self.image.size, "green")
        prepared = self.runtime.prepare(
            self.observed,
            [different, self.crop],
            assistant_token_ids={2: self.action_ids},
        )
        with self.assertRaisesRegex(PrefixMismatch, "image identity"):
            self.runtime.verify_prefix(prepared, state)
        self.assertFalse(state.consumed)
        prepared = self.runtime.prepare(
            self.observed,
            [self.image, self.crop],
            assistant_token_ids={2: self.action_ids},
        )
        with torch.inference_mode():
            prepared.position_ids[0, 0, 0] += 1
        with self.assertRaisesRegex(PrefixMismatch, "position_ids"):
            self.runtime.verify_prefix(prepared, state)
        self.assertFalse(state.consumed)

    def test_restore_compaction_and_truncation(self):
        initial = self.runtime.prefill(self.first)
        # A closed prefix followed by a question remains append-only.
        prefix = self.runtime.prepare(
            self.messages, [self.image], add_generation_prompt=False
        )
        trimmed = self.runtime.truncate(initial, prefix.logical_length)
        self.assertEqual(
            self.runtime.verify_prefix(self.first, trimmed), prefix.logical_length
        )
        self.assertEqual(
            prefix.fingerprint,
            self.runtime.prepare(
                self.messages, [self.image], add_generation_prompt=False
            ).fingerprint,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bank"
            save_cache(path, trimmed.cache, metadata=prefix.metadata)
            cache, _ = load_cache(
                path, device="cpu", expected_metadata=prefix.metadata
            )
            restored = self.runtime.restore(prefix, cache)
        cold = self.runtime.teacher_forced(self.first, targets=[42])
        restored_output = self.runtime.teacher_forced(
            self.first, state=restored, targets=[42]
        )
        torch.testing.assert_close(
            restored_output.logits[0], cold.logits[0], atol=1e-6, rtol=1e-5
        )
        compact = self.runtime.compact(trimmed, range(0, trimmed.logical_length, 2))
        compact_output = self.runtime.teacher_forced(
            self.first, state=compact, targets=[42, 43]
        )
        metrics = compact_output.metrics
        self.assertEqual(metrics["prefix_logical_tokens"], prefix.logical_length)
        self.assertEqual(
            metrics["prefix_physical_tokens"], len(range(0, prefix.logical_length, 2))
        )
        self.assertEqual(metrics["vision_input_count"], 0)
        self.assertLess(
            metrics["physical_cache_tokens"], metrics["logical_cache_tokens"]
        )


if __name__ == "__main__":
    unittest.main()
