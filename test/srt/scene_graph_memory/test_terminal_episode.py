"""Exercise the complete runner on a locked fixture with deterministic tool replay.

Use actual Mini-o3 tokenizer/tool code and native random-weight Qwen forwards.
Forced actions keep this test independent of Mini-o3 search quality.
"""

from argparse import Namespace
from io import BytesIO
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image
import torch

import test_terminal_runtime as native
import run_terminal_state_reuse as episode
from terminal_dataset import load_image, load_locked_manifest, lock_gqa_manifest, lock_manifest
from terminal_evidence import CropTools
from terminal_runtime import MiniO3Runtime


@unittest.skipUnless(native.MODEL.is_dir(), "Mini-o3 processor and tools are not installed locally")
class TerminalEpisodeTest(unittest.TestCase):
    def test_gqa_direct_answer_reuses_whole_frame_for_three_questions(self):
        # A synthetic local GQA split exercises the real loader and native KV paths.
        questions = {
            "17": {"imageId": "1", "question": "What color is the scene?", "answer": "blue"},
            "18": {"imageId": "1", "question": "How many outlined boxes are visible?", "answer": "2"},
            "19": {"imageId": "1", "question": "Is the background uniform?", "answer": "yes"},
        }
        with TemporaryDirectory() as work:
            root = Path(work)
            with ZipFile(root / "questions1.2.zip", "w") as archive:
                archive.writestr("val_balanced_questions.json", json.dumps(questions))
            image_bytes = BytesIO()
            Image.new("RGB", (84, 84), "blue").save(image_bytes, format="JPEG")
            with ZipFile(root / "images.zip", "w") as archive:
                archive.writestr("images/1.jpg", image_bytes.getvalue())
            manifest_path = root / "selected_manifest.json"
            lock_gqa_manifest(root, manifest_path, image_count=1, questions_per_image=3)
            locked = load_locked_manifest(manifest_path)
            sample = locked["samples"][0]
            native.TerminalRuntimeTest.setUpClass()
            runtime = MiniO3Runtime(
                native.TerminalRuntimeTest.model,
                native.TerminalRuntimeTest.processor,
                device="cpu", **native.PIXELS,
            )
            tools = CropTools(native.MODEL.parent / "code")
            generate = runtime.generate
            tokenizer = runtime.processor.tokenizer
            search_ids = tokenizer.encode("<answer>ACQUISITION_MARKER</answer>", add_special_tokens=False)
            answer_ids = tokenizer.encode("blue</answer>", add_special_tokens=False)

            def scripted_search(prepared, **kwargs):
                return generate(prepared, state=kwargs["state"], force_tokens=search_ids, max_new_tokens=len(search_ids))

            directory = root / "run"
            with patch.object(runtime, "generate", new=scripted_search), torch.inference_mode():
                source, trace = episode.acquire(
                    runtime, tools, sample, load_image(locked, sample), directory, 42,
                    generation=episode.GQA_GENERATION,
                )
            self.assertEqual(trace["status"], "model_stopped_with_answer")
            self.assertEqual(trace["terminal_status"], "whole_frame_original_evidence")
            self.assertEqual(trace["terminal_boundary"]["observation_id"], 0)
            self.assertEqual(trace["acquisition_new_visual_encodes"], 1)
            self.assertEqual(len(trace["observations"]), 1)

            def scripted_answer(prepared, **kwargs):
                prompt = tokenizer.decode(prepared.input_ids[0].tolist())
                self.assertTrue(prompt.endswith("assistant\n<answer>"))
                self.assertNotIn("ACQUISITION_MARKER", prompt)
                return generate(prepared, state=kwargs["state"], force_tokens=answer_ids, max_new_tokens=len(answer_ids))

            args = Namespace(modes=episode.MODES, seed=42, generation=episode.GQA_GENERATION)
            model = {"fingerprint": "random-test-fixture-only"}
            with patch.object(runtime, "generate", new=scripted_answer), torch.inference_mode():
                rows, metrics = episode.path_modes(
                    runtime, source, trace, sample, directory, args, model, root / "banks", parity=False,
                )
            self.assertEqual(len(rows), 12)
            self.assertEqual(metrics["evidence_kind"], "original_image_no_crop")
            self.assertEqual(metrics["cost_D_minus_C"]["requests_compared"], 3)
            sliced = metrics["conversion"]["terminal_kv_slice"]
            self.assertEqual(sliced["retained_source_indices"], list(range(metrics["full_history_tokens"])))
            self.assertEqual(metrics["old_storage_release"]["live_old_tensor_references"], 0)
            self.assertEqual(metrics["conversion"]["canonical_state_kv"]["canonical_vision_encodes"], 1)
            for mode in episode.MODES:
                mode_rows = [row for row in rows if row["mode"] == mode]
                self.assertEqual({row["request"] for row in mode_rows}, {"original", "followup_1", "followup_2"})
                self.assertEqual({row["question_id"] for row in mode_rows}, set(questions))
                for row in mode_rows:
                    with self.subTest(mode=mode, question_id=row["question_id"]):
                        ref = questions[row["question_id"]]["answer"]
                        self.assertEqual(row["reference"], ref)
                        self.assertEqual(row["answer"], "blue")
                        self.assertEqual(row["matches_reference"], ref == "blue")
                        self.assertEqual(row["qa_assistant_prefix"], "<answer>")
                        self.assertFalse(row["approximate"])
                        stats = row["generation"]
                        self.assertEqual(stats["vision_input_count"], int(mode == "tool_result_cache"))
                        if mode in {"full_history", "terminal_kv_slice"}:
                            self.assertEqual(stats["prefix_physical_tokens"], metrics["full_history_tokens"])
                        elif mode == "canonical_state_kv":
                            self.assertEqual(stats["prefix_physical_tokens"], metrics["canonical_prefix_tokens"])
                        else:
                            self.assertEqual(stats["prefix_physical_tokens"], 0)

    def test_four_modes_and_pixel_matched_replay(self):
        device = os.environ.get("SCENECACHE_TEST_TERMINAL_DEVICE", "cpu")
        sample_manifest = "selected_manifest.json"
        with TemporaryDirectory() as work:
            root = Path(work)
            image = Image.new("RGB", (160, 120), "grey")
            image.paste("red", (40, 30, 112, 84))
            image.save(root / "source.png")
            split = root / "labels.json"
            split.write_text(json.dumps([{
                "id": "fixture", "difficulty": "Fixture", "image": "source.png",
                "question": "What color is the rectangular object?", "answer": "GOLD_MARKER",
            }]))
            lock_manifest(root, split, root / sample_manifest, experiment={"test_fixture": True})
            locked = load_locked_manifest(root / sample_manifest)
            sample = locked["samples"][0]

            # Model initialization and any forward occur only after locking this fixture.
            native.TerminalRuntimeTest.setUpClass()
            dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
            core = MiniO3Runtime(
                native.TerminalRuntimeTest.model.to(device, dtype=dtype),
                native.TerminalRuntimeTest.processor,
                device=device, **native.PIXELS,
            )
            tools = CropTools(native.MODEL.parent / "code")
            actual_generate = core.generate
            acquisition_markers = iter([
                '<grounding>{"bbox_2d":[0.25,0.25,0.7,0.7],"source":"original_image"}</grounding>',
                "<answer>ACQUISITION_ANSWER_MARKER</answer>",
            ])

            def scripted_search(prepared, **kwargs):
                ids = core.processor.tokenizer.encode(next(acquisition_markers), add_special_tokens=False)
                return actual_generate(
                    prepared, state=kwargs["state"], force_tokens=ids,
                    max_new_tokens=len(ids), capture_logits=True,
                )

            with patch.object(core, "generate", new=scripted_search), torch.inference_mode():
                source, trace = episode.acquire(core, tools, sample, load_image(locked, sample), root / "real", 42)
            self.assertEqual(trace["status"], "model_stopped_with_answer")
            self.assertIn("ACQUISITION_ANSWER_MARKER", trace["turns"][-1]["generated_text"])
            self.assertEqual(trace["terminal_boundary"]["observation_id"], 1)
            prefix_text = core.processor.tokenizer.decode(source[1].input_ids[0].tolist())
            self.assertNotIn("ACQUISITION_ANSWER_MARKER", prefix_text)

            def bounded_generate(prepared, **kwargs):
                if not kwargs.get("force_tokens"):
                    kwargs["max_new_tokens"] = min(kwargs["max_new_tokens"], 3)
                return actual_generate(prepared, **kwargs)

            args = Namespace(modes=episode.MODES, seed=42)
            model_identity = {"fingerprint": "random-test-fixture-only"}
            with patch.object(core, "generate", new=bounded_generate), torch.inference_mode():
                rows, metrics = episode.path_modes(
                    core, source, trace, sample, root / "real", args, model_identity, root / "banks",
                    parity=True,
                )
                replay_source, replay_trace = episode.replay(core, tools, source, trace, root / "alt")
                replay_rows, replay_metrics = episode.path_modes(
                    core, replay_source, replay_trace, sample, root / "alt", args, model_identity,
                    root / "banks", parity=False,
                )
            self.assertTrue(metrics["canonical_parity"]["passed"])
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(replay_rows), 8)
            self.assertTrue(replay_trace["terminal_equivalence"]["bytewise_equal"])
            self.assertEqual(metrics["state_hash"], replay_metrics["state_hash"])
            self.assertTrue(replay_metrics["conversion"]["canonical_state_kv"]["path_independent_bank_hit"])
            by_mode = {row["mode"]: row for row in rows if row["request"] == "original"}
            self.assertLess(
                by_mode["terminal_kv_slice"]["generation"]["prefix_physical_tokens"],
                by_mode["full_history"]["generation"]["prefix_physical_tokens"],
            )
            self.assertEqual(by_mode["tool_result_cache"]["generation"]["vision_input_count"], 1)
            self.assertEqual(by_mode["canonical_state_kv"]["generation"]["vision_input_count"], 0)
            self.assertEqual(
                by_mode["tool_result_cache"]["prompt_fingerprint"],
                by_mode["canonical_state_kv"]["prompt_fingerprint"],
            )
            release = metrics["old_storage_release"]
            self.assertEqual(release["live_old_tensor_references"], 0)
            self.assertEqual(release["live_old_view_base_references"], 0)
            if torch.device(device).type == "cuda":
                self.assertGreater(release["gpu_allocated_before_bytes"], release["gpu_allocated_after_bytes"])
            state_json = (root / "real" / "state.json").read_text()
            self.assertNotIn("ACQUISITION_ANSWER_MARKER", state_json)
            self.assertNotIn("GOLD_MARKER", state_json)
            self.assertTrue((root / "banks" / metrics["state_hash"] / "terminal_kv.safetensors").is_file())

            if output := os.environ.get("SCENECACHE_TERMINAL_DIAGNOSTIC_DIR"):
                output = Path(output)
                episode.write_json(output / "mechanism.json", {
                    "kind": "random_weight_qwen_tool_replay_test",
                    "visualprobe_samples": 0, "mini_o3_checkpoint_weights_loaded": False,
                    "device": device, "dtype": str(dtype), "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
                    "old_storage_release": release,
                    "state_hash_matches_across_paths": metrics["state_hash"] == replay_metrics["state_hash"],
                    "pixel_match": replay_trace["terminal_equivalence"],
                    "canonical_parity": metrics["canonical_parity"],
                    "terminal_slice_approximation": metrics["terminal_slice_approximation"],
                    "modes": [
                        {"mode": name, "reuse_rows": row["generation"]["prefix_physical_tokens"],
                         "visual_encodes": row["generation"]["vision_input_count"],
                         "retained_kv_bytes": row["prefix_allocated_kv_bytes"]}
                        for name, row in by_mode.items()
                    ],
                })
                for name in ("real", "alt", "banks"):
                    shutil.copytree(root / name, output / name, dirs_exist_ok=True)
                episode.dump_results(output, rows + replay_rows, [
                    {"status": "paired", "sample_key": sample["sample_key"]}
                ], "diagnostic_only")
