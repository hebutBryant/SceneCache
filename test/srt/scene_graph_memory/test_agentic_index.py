"""Physical provenance and retrieval decisions over a small CPU source bank."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from sglang.srt.mem_cache.scene_graph_memory.agentic_index import (
    AgenticEpisode,
    AgenticKVIndex,
)
from sglang.srt.mem_cache.scene_graph_memory.schema import content_fingerprint
from sglang.srt.mem_cache.scene_graph_memory.store import save_memory
from test_core import source_fixture


def signed(graph):
    graph.pop("fingerprint", None)
    graph["fingerprint"] = content_fingerprint(graph)
    return graph


def graph_fixture(memory):
    names = (
        ("mask", "face mask", ["surgical mask"], {"color": "light blue"}),
        ("wearer", "woman", ["shopper"], {"clothing": "polka dot"}),
        ("scarf", "headscarf", [], {"color": "white"}),
        ("other", "woman", ["customer"], {"clothing": "plaid"}),
    )
    regions, evidence, masks = [], [], {}
    for source, (name, label, aliases, attributes) in zip(memory.regions, names):
        evidence_id = f"crop-{name}"
        regions.append(
            {
                "region_id": name,
                "label": label,
                "aliases": aliases,
                "attributes": attributes,
                "evidence_id": evidence_id,
                "mask_sha256": source.mask_sha256,
                "bbox_xyxy": source.bbox_xyxy,
                "score": 0.9 if name == "other" else 0.5,
            }
        )
        evidence.append(
            {
                "evidence_id": evidence_id,
                "kind": "sam3_mask_crop",
                "region_id": name,
                "mask_sha256": source.mask_sha256,
                "bbox_xyxy": source.bbox_xyxy,
            }
        )
        masks[name] = memory.masks[source.region_id]
    graph = {
        "schema_version": 1,
        "image_hash": memory.image_hash,
        "memory_fingerprint": memory.fingerprint,
        "source_fingerprints": {"model": memory.fingerprints["model"]},
        "regions": regions,
        "evidence": evidence,
        "semantic_edges": [
            {
                "edge_id": "wears-mask",
                "subject_id": "wearer",
                "object_id": "mask",
                "predicate": "wearing",
                "state": "supported",
                "evidence_ids": ["crop-wearer", "crop-mask", "crop-scarf"],
            },
            {
                "edge_id": "wears-scarf",
                "subject_id": "wearer",
                "object_id": "scarf",
                "predicate": "wearing",
                "state": "supported",
                "evidence_ids": ["crop-wearer", "crop-scarf"],
            },
            {
                "edge_id": "false-wearer",
                "subject_id": "other",
                "object_id": "mask",
                "predicate": "wearing",
                "state": "uncertain",
                "evidence_ids": ["crop-other", "crop-mask"],
            },
            {
                "edge_id": "next-to-scarf",
                "subject_id": "scarf",
                "object_id": "other",
                "predicate": "next to",
                "state": "supported",
                "evidence_ids": ["crop-scarf", "crop-other"],
            },
        ],
        "plan": {
            "concepts": ["face mask", "woman", "headscarf"],
            "relation_intents": [
                {"subject": "woman", "predicate": "wears", "object": "口罩"}
            ],
            "operations": [
                {
                    "id": "color",
                    "question": "What color is the face mask?",
                    "concepts": ["face mask"],
                },
                {
                    "id": "wearer",
                    "question": "Which woman wears the mask?",
                    "concepts": ["face mask", "woman"],
                },
            ],
        },
    }
    return signed(graph), masks


class AgenticKVIndexTest(unittest.TestCase):
    def setUp(self):
        self.memory = source_fixture()
        self.graph, self.masks = graph_fixture(self.memory)

    def test_image_and_mask_graph_fingerprints(self):
        self.graph["regions"][0]["label"] = "other mask"
        with self.assertRaisesRegex(ValueError, "graph schema or fingerprint"):
            AgenticKVIndex(self.memory, self.graph)
        self.graph["image_hash"] = "wrong"
        with self.assertRaisesRegex(ValueError, "images differ"):
            AgenticKVIndex(self.memory, signed(self.graph))
        self.graph["image_hash"] = self.memory.image_hash
        self.graph["source_fingerprints"]["model"] = "changed-model"
        with self.assertRaisesRegex(ValueError, "prefix or capture fingerprint"):
            AgenticKVIndex(self.memory, signed(self.graph))
        self.graph["source_fingerprints"]["model"] = self.memory.fingerprints["model"]
        self.graph["regions"][0]["bbox_xyxy"] = [0, 0, 1, 1]
        with self.assertRaisesRegex(ValueError, "mismatched mask bbox"):
            AgenticKVIndex(self.memory, signed(self.graph))

    def test_mask_payload_and_evidence_checks(self):
        with self.assertRaisesRegex(ValueError, "mask payload differs"):
            AgenticKVIndex(
                self.memory, self.graph, dict(self.masks, mask=self.masks["wearer"])
            )
        self.graph["semantic_edges"][0]["evidence_ids"] = ["crop-mask"]
        with self.assertRaisesRegex(ValueError, "lacks evidence for both endpoints"):
            AgenticKVIndex(self.memory, signed(self.graph))
        self.graph["semantic_edges"][0]["evidence_ids"] += [
            "crop-wearer",
            "unregistered",
        ]
        with self.assertRaisesRegex(ValueError, "Invalid evidence"):
            AgenticKVIndex(self.memory, signed(self.graph))
        self.graph["semantic_edges"][0]["evidence_ids"].remove("unregistered")
        self.graph["semantic_edges"][0]["round_index"] = 1
        self.graph["trace"] = {
            "rounds": [{"round_index": 1, "board": {"evidence_ids": ["crop-mask"]}}]
        }
        with self.assertRaisesRegex(ValueError, "undisplayed round evidence"):
            AgenticKVIndex(self.memory, signed(self.graph))

    def test_verified_relation_evidence_and_disk_tensor_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = save_memory(self.memory, Path(temp) / "source")
            index = AgenticKVIndex(directory, self.graph, self.masks)
            result = index.retrieve(
                "Which woman wears the mask?", operation="wearer", token_budget=3
            )
            self.assertIsNone(result.fallback)
            self.assertEqual(
                [row["edge_id"] for row in result.selected_edges], ["wears-mask"]
            )
            self.assertEqual(
                {row["region_id"] for row in result.selected_nodes},
                {"wearer", "mask", "scarf"},
            )
            self.assertEqual(result.source_visual_indices, (0, 2, 5))
            self.assertEqual(result.region_ids, ("r0001", "r0002", "r0003"))
            self.assertEqual(result.kv_reference.source_sequence_indices, (2, 4, 7))
            self.assertEqual(
                result.kv_reference.mrope_positions, ((2, 2, 2), (2, 2, 4), (2, 3, 4))
            )
            self.assertEqual(
                result.kv_reference.layers[2]["key"]["tensor"], "layer.2.k"
            )
            self.assertEqual(
                result.kv_reference.bank_file, str(directory / "source_kv.safetensors")
            )
            self.assertEqual(len(result.kv_reference.bank_sha256), 64)
            self.assertEqual(
                result.selection_dict()["source_visual_indices"], [0, 2, 5]
            )
            self.assertEqual(
                result.selection_dict()["memory_fingerprint"], self.memory.fingerprint
            )
            self.assertEqual(
                json.loads(json.dumps(result.to_dict()))["operation_id"], "wearer"
            )

            exact = index.read_kv(result)
            for layer, pair in exact.layers.items():
                for actual, bank in zip(pair, self.memory.source_kv[layer]):
                    self.assertTrue(torch.equal(actual, bank[[0, 2, 5]]))
            with self.assertRaisesRegex(ValueError, "reference differs"):
                index.read_kv(
                    replace(result.kv_reference, source_sequence_indices=(2, 3, 7))
                )

    def test_budget_preserves_graph_requirements_or_reports_full_fallback(self):
        index = AgenticKVIndex(self.memory, self.graph)
        result = index.retrieve(
            "Who wears the face mask?", operation="wearer", token_budget=2
        )
        self.assertEqual(result.fallback["mode"], "full_visual")
        self.assertTrue(result.fallback["exceeds_token_budget"])
        self.assertEqual(result.source_visual_indices, tuple(range(6)))
        self.assertEqual(result.missing_evidence[0]["reason"], "no_supported_edge_fits")
        self.assertFalse(result.selected_edges)
        self.assertTrue(any(row["outcome"] == "token_budget" for row in result.reasons))

    def test_low_confidence_relation_logs_rejection_and_preserves_fallback(self):
        edge = self.graph["semantic_edges"][0]
        edge["confidence"] = 0.49
        index = AgenticKVIndex(self.memory, signed(self.graph))
        result = index.retrieve("unused", operation="wearer")
        self.assertEqual(result.fallback["mode"], "full_visual")
        self.assertEqual(result.missing_evidence[0]["reason"], "no_supported_edge")
        self.assertIn("wears-mask", result.missing_evidence[0]["candidate_edge_ids"])
        rejection = next(
            row for row in result.reasons if row["outcome"] == "low_confidence"
        )
        self.assertEqual(rejection["edge_id"], "wears-mask")
        self.assertEqual(rejection["confidence"], 0.49)
        self.assertEqual(rejection["min_edge_confidence"], 0.5)
        self.assertEqual(index.to_dict()["min_edge_confidence"], 0.5)
        expanded = index.retrieve("unused", operation="color", max_hops=2)
        self.assertEqual(expanded.region_ids, ("r0001",))
        self.assertFalse(expanded.selected_edges)
        self.assertTrue(
            any(row["outcome"] == "low_confidence" for row in expanded.reasons)
        )
        for confidence in (None, 0.5):
            edge["confidence"] = confidence
            eligible = AgenticKVIndex(self.memory, signed(self.graph))
            retrieval = eligible.retrieve("unused", operation="wearer")
            self.assertIsNone(retrieval.fallback)
            self.assertEqual(retrieval.selected_edges[0]["edge_id"], "wears-mask")

    def test_operation_question_expands_named_relation_endpoints(self):
        self.graph["regions"][0].update(label="shirt", aliases=[])
        self.graph["regions"][1].update(label="goalkeeper", aliases=["football keeper"])
        self.graph["plan"] = {
            "concepts": ["goalkeeper", "shirt"],
            "relation_intents": [
                {"subject": "goalkeeper", "predicate": "is wearing", "object": "shirt"}
            ],
        }
        index = AgenticKVIndex(self.memory, signed(self.graph))
        for noun in ("goalkeeper", "goalie", "football keeper", "守门员"):
            question = f"What color shirt is the {noun} wearing?"
            result = index.retrieve(
                question, operation={"kind": "answer", "concepts": ["shirt"]}
            )
            self.assertIsNone(result.fallback)
            self.assertEqual(
                [row["edge_id"] for row in result.selected_edges], ["wears-mask"]
            )
        no_mention = index.retrieve(
            "What color is the shirt?", operation={"concepts": ["shirt"]}
        )
        self.assertFalse(no_mention.selected_edges)
        self.assertEqual(no_mention.region_ids, ("r0001",))

    def test_uncertain_geometry_and_wrong_direction_stay_missing(self):
        index = AgenticKVIndex(self.memory, self.graph)
        for relation in (
            {
                "subject": {"concept": "woman", "attributes": {"clothing": "plaid"}},
                "predicate": "wearing",
                "object": "face mask",
            },
            {"subject": "face mask", "predicate": "wearing", "object": "woman"},
            {"subject": "face mask", "predicate": "touch", "object": "woman"},
        ):
            operation = {"concepts": [], "relation_intents": [relation]}
            result = index.retrieve("Where is the face mask?", operation=operation)
            self.assertEqual(result.missing_evidence[0]["reason"], "no_supported_edge")
            self.assertEqual(result.fallback["mode"], "full_visual")
            self.assertFalse(result.selected_edges)

    def test_operation_limits_terms_and_background_does_not_create_graph_nodes(self):
        index = AgenticKVIndex(self.memory, self.graph)
        result = index.retrieve(
            "unused", operation="color", token_budget=3, background_tokens=2
        )
        self.assertEqual(result.source_visual_indices, (0, 1, 2))
        self.assertEqual(result.background_visual_indices, (1,))
        self.assertEqual(result.region_ids, ("r0001",))
        self.assertFalse(result.selected_edges)
        self.assertIsNone(result.fallback)
        result = index.retrieve(
            "What color is the mask worn by the polka dot shopper?",
            operation={
                "concepts": [
                    {"concept": "face mask", "aliases": ["mask"]},
                    {"concept": "shopper", "attributes": {"clothing": "polka dot"}},
                ],
                "relation_intents": [],
            },
            token_budget=3,
        )
        self.assertEqual(result.region_ids, ("r0001", "r0002"))
        self.assertEqual(result.source_visual_indices, (0, 2, 5))
        self.assertEqual(result.reasons[-1]["matched_attributes"], ["polka dot"])
        inferred = index.retrieve("What color is the shopper's clothing?")
        self.assertEqual(inferred.region_ids, ("r0002",))

    def test_semantic_only_hop_limits_and_episode_state(self):
        index = AgenticKVIndex(self.memory, self.graph)
        episode = AgenticEpisode(index)
        retrievals = [
            episode.retrieve("unused", operation="color", max_hops=depth)
            for depth in range(3)
        ]
        self.assertEqual(
            [result.region_ids for result in retrievals],
            [
                ("r0001",),
                ("r0001", "r0002", "r0003"),
                ("r0001", "r0002", "r0003", "r0004"),
            ],
        )
        fresh = episode.retrieve("unused", operation="color")
        self.assertEqual(fresh.region_ids, ("r0001",))
        episode.bind(AgenticKVIndex(self.memory, self.graph))
        self.assertEqual(len(episode.history), 4)
        self.graph["plan"]["concepts"].append("updated")
        episode.bind(AgenticKVIndex(self.memory, signed(self.graph)))
        self.assertEqual(episode.history, [])
        self.assertEqual(episode.image_hash, self.memory.image_hash)
