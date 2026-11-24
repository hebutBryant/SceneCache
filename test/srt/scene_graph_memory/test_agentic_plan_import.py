"""CPU checks for importing a saved first SAM3 planning run."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from sglang.srt.mem_cache.scene_graph_memory.agentic_plan_import import (
    import_scene2graph_plan,
)


class Scene2GraphPlanImportTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = Image.new("RGB", (32, 16), (40, 120, 180))
        archived_input = "/archived/source.jpg"
        text = [" menu board ", "coffee bag", "menu board"]
        self.intent = {
            "id": "P1",
            "when": "now",
            "question": "Locate signage.",
            "suggested_tools": ["sam3"],
        }
        self.decision = {
            "action": "call_tools",
            "tool_calls": [
                {"call_id": "T1", "tool_name": "sam3", "arguments": {"text": text}}
            ],
        }
        self.write("plan.json", {"intents": [self.intent]})
        self.write("decision.json", self.decision)
        self.write(
            "first_sam3/request.json",
            {"tool": "sam", "image": archived_input, "params": {"text": text}},
        )
        thumbnail = self.image.copy()
        thumbnail.thumbnail((16, 16))
        for step, stage in enumerate(("planning", "decision"), start=1):
            call = f"thinking/call_{step:05d}_online_{stage}"
            self.write(
                f"{call}/request.json",
                {"stage": f"online_{stage}", "images": [archived_input]},
            )
            thumbnail.save(self.root / call / "image_0.png")

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_uses_first_actual_sam3_call(self):
        result = import_scene2graph_plan(
            self.root, image=self.image, question=" What is visible? "
        )
        self.assertEqual(result["question"], "What is visible?")
        self.assertEqual(result["concepts"], ["menu board", "coffee bag"])
        self.assertEqual(result["relation_intents"], [])
        self.assertEqual(result["source"]["intents"], [self.intent])
        self.assertEqual(result["source"]["sam3_call_id"], "T1")
        self.assertEqual(len(result["source"]["image_pixel_sha256"]), 64)

    def test_rejects_different_pixels(self):
        different = Image.new("RGB", self.image.size)
        with self.assertRaisesRegex(ValueError, "differs from the archived"):
            import_scene2graph_plan(self.root, image=different, question="What?")

    def test_rejects_mismatched_execution(self):
        self.decision["tool_calls"][0]["arguments"] = {"text": "person"}
        self.write("decision.json", self.decision)
        with self.assertRaisesRegex(ValueError, "disagree on concepts"):
            import_scene2graph_plan(self.root, image=self.image, question="What?")
