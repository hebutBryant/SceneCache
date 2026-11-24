import importlib.util
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from examples.scene_graph_memory.terminal_dataset import (
    GQA_IMAGES_ARCHIVE,
    GQA_QUESTIONS_ARCHIVE,
    GQA_SPLIT,
    VISUALPROBE_SPLITS,
    load_image,
    load_locked_manifest,
    lock_gqa_manifest,
    lock_manifest,
)


class TerminalDatasetLockTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "dataset"
        self.root.mkdir()
        self.output = Path(temporary.name) / "lock.json"
        self.image = self.root / "image.png"
        Image.new("RGB", (5, 3), "orange").save(self.image)

    def write_rows(self, path, rows):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        text = (
            "\n".join(map(json.dumps, rows))
            if path.suffix == ".jsonl"
            else json.dumps(rows)
        )
        path.write_text(text, encoding="utf-8")
        return path

    def test_official_visualprobe_replay_and_exact_references(self):
        for difficulty, relative in VISUALPROBE_SPLITS.items():
            self.write_rows(
                relative,
                [
                    {
                        "doc_id": index,
                        "images": ["image.png"],
                        "problem": f"<image>\nWhat is visible in item {index}?",
                        "solution": " C.\n",
                        "data_source": f"visual_probe_{difficulty}",
                    }
                    for index in range(7)
                ],
            )
        manifest = lock_manifest(self.root, output=self.output)
        self.assertEqual(
            Counter(row["difficulty"] for row in manifest["samples"]),
            {"easy": 5, "medium": 5, "hard": 5},
        )
        self.assertEqual(len({row["sample_key"] for row in manifest["samples"]}), 15)
        sample = manifest["samples"][0]
        self.assertEqual(sample["answer"], " C.\n")
        self.assertEqual(sample["source_question"], "<image>\n" + sample["question"])
        self.assertEqual(sample["difficulty_source"], "official_validation_split")
        self.assertEqual(sample["followup"]["scoring"], "rubric_only")
        self.assertIsNone(sample["followup"]["answer"])
        self.assertEqual(load_image(manifest, sample).size, (5, 3))
        self.assertEqual(load_locked_manifest(self.output), manifest)
        self.assertEqual(lock_manifest(self.root, output=self.output), manifest)
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            lock_manifest(self.root, output=self.output, seed=43)

    def test_explicit_visualprobe_splits_preserve_official_labels(self):
        for difficulty in ("easy", "hard"):
            self.write_rows(
                VISUALPROBE_SPLITS[difficulty],
                [
                    {
                        "doc_id": f"{difficulty}-{index}",
                        "images": [self.image.name],
                        "problem": "<image>\nWhat is visible?",
                        "solution": "orange",
                        "data_source": f"visual_probe_{difficulty}",
                    }
                    for index in range(3)
                ],
            )
        manifest = lock_manifest(
            self.root, output=self.output, per_difficulty=2,
            visualprobe_splits=("easy", "hard"),
        )
        self.assertEqual(
            Counter(row["difficulty"] for row in manifest["samples"]),
            {"easy": 2, "hard": 2},
        )
        self.assertEqual(load_locked_manifest(self.output), manifest)
        with self.assertRaisesRegex(ValueError, "unique selection"):
            lock_manifest(self.root, output=self.output, visualprobe_splits=("easy", "easy"))

    def test_generic_jsonl_stratification_and_explicit_followup(self):
        rows = [
            {
                "key": f"s{index}",
                "tier": 1 + index % 2,
                "media": {"path": self.image.name},
                "text": {"ask": "Describe the object.", "more": "What color is it?"},
                "label": {"initial": ["1", 2], "follow": " Orange "},
            }
            for index in range(12)
        ]
        first = self.write_rows("annotations.jsonl", rows)
        second = self.write_rows("shuffled.jsonl", list(reversed(rows)))
        fields = {
            "id": "key",
            "difficulty": "tier",
            "image": "media.path",
            "question": "text.ask",
            "answer": "label.initial",
            "followup_question": "text.more",
            "followup_answer": "label.follow",
        }
        first_lock = lock_manifest(self.root, first, self.output, fields=fields)
        second_lock = lock_manifest(
            self.root, second, self.output.with_name("another.json"), fields=fields
        )
        self.assertEqual(
            [row["sample_key"] for row in first_lock["samples"]],
            [row["sample_key"] for row in second_lock["samples"]],
        )
        sample = first_lock["samples"][0]
        self.assertEqual(sample["answer"], ["1", 2])
        self.assertEqual(
            sample["followup"],
            {
                "question": "What color is it?",
                "answer": " Orange ",
                "scoring": "reference",
                "source": "dataset",
            },
        )
        Image.new("RGB", (5, 3), "pink").save(self.image)
        with self.assertRaisesRegex(ValueError, "Locked image bytes changed"):
            load_image(first_lock, sample)
        first.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Locked annotation source changed"):
            load_locked_manifest(self.output, verify_images=False)

    def test_missing_split_does_not_publish_and_real_shortfall_is_frozen(self):
        with self.assertRaisesRegex(FileNotFoundError, "VisualProbe_Hard/val.json"):
            lock_manifest(self.root, output=self.output)
        self.assertFalse(self.output.exists())
        empty = self.write_rows("empty.json", [])
        with self.assertRaisesRegex(ValueError, "No annotation rows"):
            lock_manifest(self.root, empty, self.output)
        self.assertFalse(self.output.exists())
        first = self.write_rows(
            "rows.json",
            [
                {
                    "id": 6,
                    "difficulty": "level 6",
                    "image": "image.png",
                    "question": "What is visible?",
                    "answer": "A",
                },
            ],
        )
        manifest = lock_manifest(self.root, first, self.output)
        self.assertEqual(
            manifest["dataset"]["difficulty_counts"],
            [
                {"difficulty": "level 6", "count": 1, "selected": 1, "shortfall": 4},
            ],
        )
        self.assertEqual(load_locked_manifest(self.output), manifest)

    def test_local_gqa_zips_lock_distinct_questions_per_image(self):
        rows = {}
        for image in ("100", "200", "300"):
            for index in range(3):
                rows[f"{image}{index}"] = {
                    "imageId": image,
                    "question": f"What is object {index} in image {image}?",
                    "answer": f"gold{index}",
                    "types": {"structural": "query"},
                    "annotations": {"answer": {"0": "10001"}},
                    "semantic": [{"operation": "select", "argument": "ball", "dependencies": []}],
                }
        rows["1009"] = {**rows["1000"], "question": rows["1000"]["question"].upper()}
        questions_path = self.root / GQA_QUESTIONS_ARCHIVE
        image_zip = self.root / GQA_IMAGES_ARCHIVE
        with ZipFile(questions_path, "w") as archive:
            archive.writestr(GQA_SPLIT, json.dumps(rows))
        image_payload = self.image.read_bytes()
        with ZipFile(image_zip, "w") as archive:
            for image in ("100", "200", "300"):
                archive.writestr(f"images/{image}.jpg", image_payload)
        manifest = lock_gqa_manifest(
            self.root, self.output, image_count=2, experiment={"generation": "same"}
        )
        self.assertEqual([case["image_id"] for case in manifest["samples"]], ["300", "100"])
        self.assertEqual(manifest["dataset"]["eligible_image_count"], 3)
        self.assertEqual(manifest, load_locked_manifest(self.output))
        for case in manifest["samples"]:
            requests = case["requests"]
            self.assertEqual([req["name"] for req in requests], ["original", "followup_1", "followup_2"])
            self.assertEqual(case["question"], requests[0]["question"])
            self.assertEqual(case["answer"], requests[0]["answer"])
            for request in requests:
                self.assertEqual(rows[request["question_id"]]["imageId"], case["image_id"])
                self.assertEqual(rows[request["question_id"]]["answer"], request["answer"])
            self.assertEqual(load_image(manifest, case).size, (5, 3))
        self.assertFalse((self.root / "images").exists())
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            load_locked_manifest(self.output, experiment={"generation": "different"})
        with ZipFile(image_zip, "w") as archive:
            archive.writestr("images/300.jpg", image_payload + b"different")
        with self.assertRaisesRegex(ValueError, "Locked image bytes changed"):
            load_locked_manifest(self.output)
        with ZipFile(questions_path, "w") as archive:
            archive.writestr(GQA_SPLIT, "{}")
        with self.assertRaisesRegex(ValueError, "Locked GQA validation annotations changed"):
            load_locked_manifest(self.output, verify_images=False)

    def test_experiment_settings_are_fingerprinted(self):
        first = self.write_rows(
            "rows.json",
            [
                {
                    "id": 1,
                    "difficulty": "easy",
                    "image": "image.png",
                    "question": "What is visible?",
                    "answer": "A",
                },
            ],
        )
        settings = {"fingerprints": {"model": "m"}, "generation": {"temperature": 0}}
        manifest = lock_manifest(self.root, first, self.output, experiment=settings)
        self.assertEqual(
            load_locked_manifest(self.output, experiment=settings), manifest
        )
        settings["generation"]["temperature"] = 0.2
        for action in (
            lambda: load_locked_manifest(self.output, experiment=settings),
            lambda: lock_manifest(self.root, first, self.output, experiment=settings),
        ):
            with self.assertRaisesRegex(ValueError, "experiment metadata differs"):
                action()

    def test_fingerprint_and_root_boundary(self):
        first = self.write_rows(
            "rows.json",
            [
                {
                    "id": 1,
                    "difficulty": "easy",
                    "image": "image.png",
                    "question": "What is visible?",
                    "answer": "A",
                },
            ],
        )
        manifest = lock_manifest(self.root, first, self.output, per_difficulty=1)
        manifest["samples"][0]["answer"] = "B"
        self.output.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            load_locked_manifest(self.output)

        escaped = self.write_rows(
            "escape.json",
            [
                {
                    "id": 1,
                    "difficulty": "easy",
                    "image": "../foreign.png",
                    "question": "What is visible?",
                    "answer": "A",
                },
            ],
        )
        with self.assertRaisesRegex(ValueError, "escapes the dataset root"):
            lock_manifest(
                self.root,
                escaped,
                self.output.with_name("escape_lock.json"),
                per_difficulty=1,
            )

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow"), "optional pyarrow is unavailable"
    )
    def test_parquet_embedded_image_row_groups(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = []
        for index in range(6):
            data = io.BytesIO()
            Image.new("RGB", (index + 2, 3), (index, 120, 240)).save(data, format="PNG")
            rows.append(
                {
                    "id": index,
                    "difficulty": "medium",
                    "question": "What is visible?",
                    "answer": "010",
                    "images": [
                        {"path": f"archive/{index}.png", "bytes": data.getvalue()}
                    ],
                }
            )
        parquet = self.root / "rows.parquet"
        pq.write_table(pa.Table.from_pylist(rows), parquet, row_group_size=2)
        manifest = lock_manifest(
            self.root, parquet, self.output, fields={"image": "images"}
        )
        self.assertEqual(len(manifest["samples"]), 5)
        for sample in load_locked_manifest(self.output)["samples"]:
            self.assertEqual(sample["image_storage"], "parquet_embedded")
            self.assertEqual(sample["image"], f"archive/{sample['id']}.png")
            self.assertEqual(sample["answer"], "010")
            self.assertEqual(load_image(manifest, sample).size, (sample["id"] + 2, 3))


if __name__ == "__main__":
    unittest.main()
