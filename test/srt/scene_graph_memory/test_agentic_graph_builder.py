import json
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from sglang.srt.mem_cache.scene_graph_memory.agentic_graph import (
    GraphBudget,
    build_agentic_graph,
    normalize_plan,
    plan_scene_graph,
)
from sglang.srt.mem_cache.scene_graph_memory.schema import (
    MaskInput,
    content_fingerprint,
)


class AgenticGraphTest(unittest.TestCase):
    question = "What color is the mask worn by the woman behind the scarf?"

    def setUp(self):
        self.image = Image.new("RGB", (8, 6), "navy")

    @staticmethod
    def mask(label, bounds):
        pixels = np.zeros((6, 8), dtype=np.bool_)
        x0, y0, x1, y1 = bounds
        pixels[y0:y1, x0:x1] = True
        return MaskInput(pixels, 0.9, label, "test_candidate")

    def test_native_plan_is_bounded_and_retains_generated_operations(self):
        messages = []

        def reason(prompt):
            messages.extend(prompt)
            raw = {
                "concepts": ["face mask", "Face Mask", "person", "fabric?", "scarf"],
                "relation_intents": [
                    {
                        "subject": "person",
                        "predicate": "Wearing",
                        "object": "Face Mask",
                    },
                    {"subject": "scarf", "predicate": "behind", "object": "person"},
                ],
                "operations": [
                    {
                        "id": "answer",
                        "kind": "answer",
                        "question": self.question,
                        "concepts": ["face mask"],
                    },
                    {
                        "id": "locate",
                        "kind": "perception",
                        "question": "Where is the mask?",
                        "concepts": ["Face Mask", "person"],
                    },
                    {
                        "id": "locate",
                        "kind": "perception",
                        "question": "Duplicate operation",
                        "concepts": ["person"],
                    },
                ],
            }
            return f"```json\n{json.dumps(raw)}\n```"

        config = GraphBudget(max_concepts=2)
        plan = plan_scene_graph(
            self.image, question=self.question, reason=reason, budget=config
        )
        self.assertEqual(plan["concepts"], ["face mask", "person"])
        self.assertEqual(
            plan["relation_intents"],
            [{"subject": "person", "predicate": "wearing", "object": "face mask"}],
        )
        self.assertEqual(
            [row["id"] for row in plan["operations"]], ["locate", "answer"]
        )
        self.assertEqual(plan["source"]["kind"], "vlm_generated")
        self.assertEqual(
            normalize_plan(plan, question=self.question, budget=config), plan
        )
        self.assertEqual(
            [entry["type"] for entry in messages[0]["content"]], ["image_url", "text"]
        )
        self.assertIn(
            "data:image/png;base64,", messages[0]["content"][0]["image_url"]["url"]
        )
        with self.assertRaisesRegex(ValueError, "different question"):
            normalize_plan(plan, question="Another question", budget=config)

    def test_segment_expand_and_bind_scoped_evidence(self):
        calls = []

        def segment(concepts):
            calls.append(tuple(concepts))
            if len(calls) == 1:
                return [
                    self.mask("person", (0, 0, 4, 6)),
                    self.mask("scarf", (1, 1, 4, 4)),
                    self.mask("alias", (0, 0, 4, 6)),
                    self.mask("empty", (0, 0, 0, 0)),
                ], {"backend": "injected"}
            return [self.mask("face mask", (2, 3, 4, 4))], {"backend": "injected"}

        calls_vlm = []

        def reason(messages):
            calls_vlm.append(messages)
            verified = {
                "subject_id": "r0001",
                "predicate": "wearing",
                "object_id": "r0002",
                "state": "supported",
                "evidence_ids": ["ev0001", "ev0002"],
                "visual_evidence": "The scarf follows the person's neck.",
            }
            if len(calls_vlm) == 1:
                return {
                    "relations": [
                        verified,
                        {**verified, "subject_id": "r9001"},
                        {
                            **verified,
                            "predicate": "holding",
                            "evidence_ids": ["ev0001"],
                        },
                    ],
                    "expand_concepts": ["Scarf", "face mask", "vegetable"],
                    "stop": False,
                }
            return {
                "relations": [
                    {**verified, "state": "contradicted"},
                    {
                        **verified,
                        "object_id": "r0003",
                        "evidence_ids": ["ev0001", "ev0003"],
                    },
                ],
                "expand_concepts": [],
                "stop": True,
            }

        graph = build_agentic_graph(
            self.image,
            question=self.question,
            reason=reason,
            segment=segment,
            existing_plan={"concepts": ["person", "scarf"]},
            budget=GraphBudget(max_concepts=3, max_vlm_calls=2, max_sam_calls=2),
        )
        self.assertEqual(calls, [("person", "scarf"), ("face mask",)])
        self.assertEqual((graph.trace["sam_calls"], graph.trace["vlm_calls"]), (2, 2))
        self.assertEqual(graph.masks[0].metadata["prompt_aliases"], ["alias"])
        self.assertEqual(
            [edge.state for edge in graph.semantic_edges], ["uncertain", "supported"]
        )
        self.assertEqual(len(graph.trace["rounds"][0]["rejected_relations"]), 2)
        self.assertEqual(
            set(graph.evidence_images), {"verification_01", "verification_02"}
        )
        for index, evidence in enumerate(graph.evidence):
            self.assertEqual(
                evidence["mask_sha256"], graph.regions[index]["mask_sha256"]
            )
        self.assertIn("ev0003", graph.trace["rounds"][1]["board"]["evidence_ids"])

        memory = SimpleNamespace(
            image_hash=graph.image_hash,
            fingerprint="memory-sha",
            fingerprints={"model": "model-sha"},
            regions=[
                SimpleNamespace(
                    mask_sha256=row["mask_sha256"], region_id=f"r{10-index:04d}"
                )
                for index, row in enumerate(graph.regions)
            ],
        )
        manifest = graph.manifest(memory)
        self.assertEqual(manifest["semantic_edges"][1]["subject_id"], "r0010")
        self.assertEqual(manifest["semantic_edges"][1]["object_id"], "r0008")
        self.assertEqual(manifest["evidence"][2]["region_id"], "r0008")
        self.assertEqual(manifest["memory_fingerprint"], "memory-sha")
        self.assertEqual(manifest["source_fingerprints"], {"model": "model-sha"})
        fingerprint = manifest.pop("fingerprint")
        self.assertEqual(fingerprint, content_fingerprint(manifest))

    def test_vlm_budget_prevents_unverified_edges(self):
        def reason(_):
            return {
                "concepts": ["face mask"],
                "operations": [
                    {
                        "id": "answer",
                        "kind": "answer",
                        "question": self.question,
                        "concepts": ["face mask"],
                    },
                ],
            }

        graph = build_agentic_graph(
            self.image,
            question=self.question,
            reason=reason,
            segment=lambda _: (
                [self.mask("face mask", (1, 2, 2, 6))],
                {"backend": "injected"},
            ),
            budget=GraphBudget(max_vlm_calls=1),
        )
        self.assertEqual(graph.trace["stop_reason"], "vlm_budget")
        self.assertEqual(graph.trace["vlm_calls"], 1)
        self.assertEqual(graph.semantic_edges, [])

    def test_targeted_alternative_and_attribute_use_actual_crop_evidence(self):
        concept_batches = []

        def segment(concepts):
            concept_batches.append(list(concepts))
            if len(concept_batches) == 1:
                return [self.mask("person", (0, 0, 4, 6))], {"backend": "injected"}
            return [self.mask("surgical mask", (2, 3, 4, 4))], {"backend": "injected"}

        def reason(_):
            if len(concept_batches) == 1:
                return {
                    "relations": [],
                    "stop": True,
                    "expand_concepts": [
                        {"concept": "surgical mask", "for_concept": "face mask"},
                        {"concept": "scarf", "for_concept": "unseen"},
                    ],
                }
            relations = [
                {
                    "subject_id": "r0001",
                    "object_id": "r0002",
                    "predicate": "is wearing",
                    "state": "supported",
                    "evidence_ids": ["ev0001", "ev0002"],
                    "confidence": 0.8,
                    "visual_evidence": "The straps connect to the sides of the person's face.",
                }
            ]
            observation = {
                "region_id": "r0002",
                "attribute": "color",
                "value": "blue",
                "evidence_id": "ev0002",
                "visual_evidence": "The retained fabric pixels are blue.",
            }
            return {
                "relations": [
                    *relations,
                    {**relations[0], "predicate": "holding", "confidence": 2},
                ],
                "attribute_observations": [
                    observation,
                    {**observation, "evidence_id": "ev0001"},
                ],
                "stop": True,
            }

        graph = build_agentic_graph(
            self.image,
            question=self.question,
            reason=reason,
            segment=segment,
            existing_plan={
                "concepts": ["person", "face mask"],
                "relation_intents": [
                    {"subject": "person", "predicate": "wearing", "object": "face mask"}
                ],
                "attribute_intents": [{"concept": "face mask", "attribute": "Color"}],
            },
        )
        self.assertEqual(concept_batches, [["person", "face mask"], ["surgical mask"]])
        self.assertEqual(
            graph.trace["rounds"][0]["unresolved_intents"],
            ["relation_01", "attribute_01"],
        )
        self.assertEqual(graph.trace["rounds"][1]["unresolved_intents"], [])
        self.assertEqual(graph.trace["stop_reason"], "verifier_stop")
        self.assertEqual(graph.regions[1]["prompt_aliases"], ["face mask"])
        self.assertEqual(len(graph.semantic_edges), 1)
        self.assertEqual(graph.semantic_edges[0].confidence, 0.8)
        self.assertEqual(graph.semantic_edges[0].source, "vlm_visual_verifier")
        self.assertEqual(len(graph.attribute_observations), 1)
        self.assertEqual(graph.attribute_observations[0]["value"], "blue")
        self.assertIsNone(graph.attribute_observations[0]["confidence"])

    def test_validation_repair_and_segmentation_failure_are_counted(self):
        prompts = []

        def reason(messages):
            prompts.append(messages[0]["content"][-1]["text"])
            if len(prompts) == 1:
                return "unparseable planner response"
            return {
                "concepts": ["face mask"],
                "operations": [
                    {
                        "id": "answer",
                        "kind": "answer",
                        "question": self.question,
                        "concepts": ["face mask"],
                    },
                ],
            }

        def fail(_):
            raise RuntimeError("SAM unavailable")

        graph = build_agentic_graph(
            self.image, question=self.question, reason=reason, segment=fail
        )
        self.assertIn("repair_previous_response", prompts[1])
        self.assertEqual(graph.trace["stop_reason"], "sam_error")
        self.assertEqual((graph.trace["vlm_calls"], graph.trace["sam_calls"]), (2, 1))
        self.assertEqual(
            graph.trace["rounds"][0]["segmentation_error"]["message"], "SAM unavailable"
        )
        self.assertEqual(graph.masks, [])

    def test_disjoint_wearable_endpoint_is_not_admitted_as_supported(self):
        def reason(_):
            return {
                "relations": [
                    {
                        "subject_id": "r0001",
                        "predicate": "wearing",
                        "object_id": "r0002",
                        "state": "supported",
                        "evidence_ids": ["ev0001", "ev0002"],
                        "visual_evidence": "The garment belongs to the person.",
                        "confidence": 0.95,
                    }
                ],
                "stop": True,
            }

        graph = build_agentic_graph(
            self.image,
            question=self.question,
            reason=reason,
            existing_plan={
                "concepts": ["person", "mask"],
                "relation_intents": [
                    {"subject": "person", "predicate": "wearing", "object": "mask"},
                ],
            },
            segment=lambda _: (
                [self.mask("person", (0, 0, 2, 6)), self.mask("mask", (5, 2, 7, 4))],
                {"backend": "injected"},
            ),
        )
        edge = graph.semantic_edges[0]
        self.assertEqual(edge.state, "uncertain")
        self.assertIsNone(edge.confidence)
        self.assertEqual(edge.geometry_review["bbox_gap_xy"], [3, 0])
        self.assertEqual(edge.geometry_review["original_confidence"], 0.95)
        self.assertEqual(graph.trace["stop_reason"], "unresolved_intents")
        self.assertEqual(
            graph.trace["rounds"][0]["geometry_reviews"][0]["original_state"],
            "supported",
        )


    def test_near_duplicate_merges_alias_without_collapsing_distinct_instances(self):
        image = Image.new("RGB", (48, 32))
        first = np.zeros((32, 48), dtype=bool)
        first[2:26, 2:23] = True
        near = first.copy()
        near[8, 8] = False
        second = np.zeros_like(first)
        second[3:27, 25:46] = True
        graph = build_agentic_graph(
            image, question=self.question,
            existing_plan={"concepts": ["person", "woman"]},
            segment=lambda _: (
                [MaskInput(first, label="person"), MaskInput(near, label="woman"),
                 MaskInput(second, label="person")], {"backend": "fixture"},
            ),
            reason=lambda _: {"relations": [], "expand_concepts": [], "stop": True},
        )
        self.assertEqual([row["region_id"] for row in graph.regions], ["r0001", "r0002"])
        self.assertIn("woman", graph.regions[0]["prompt_aliases"])
        duplicate, = graph.trace["rounds"][0]["filtered_masks"]
        self.assertEqual(duplicate["reason"], "near_duplicate_mask")
        self.assertGreater(duplicate["mask_iou"], 0.99)


if __name__ == "__main__":
    unittest.main()
