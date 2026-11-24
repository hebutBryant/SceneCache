"""Bounded question-guided SAM3 localization and evidence-backed graph proposals."""

from __future__ import annotations

import base64
import io
import json
import re
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .graph import mask_bbox
from .sam3_adapter import create_masks
from .schema import (
    ImageMemory,
    MaskInput,
    content_fingerprint,
    image_sha256,
    json_bytes,
    mask_sha256,
)

Reason = Callable[[list[dict]], str | dict]
Segment = Callable[[list[str]], tuple[list[MaskInput], dict]]
_STATES = {"supported", "uncertain", "contradicted"}
_INSTANCE_MASK_IOU = 0.98
_PLAN_DESIGN = """Plan visible evidence for a single image question. Return only one JSON object:
{"concepts":["short English object or visible part name"],
 "relation_intents":[{"subject":"one concept","predicate":"short relation","object":"another concept","reason":"need for this check"}],
 "attribute_intents":[{"concept":"one concept","attribute":"color or other visible attribute","reason":"need for this check"}],
 "operations":[{"id":"inspect_1","kind":"perception","question":"specific follow-up visual question","concepts":["one concept"]},
               {"id":"answer","kind":"answer","question":"the exact original question","concepts":["one concept"]}]}
Order concepts by relevance for text-prompt segmentation. Keep only the visible objects and parts required for the question. Use short concrete noun phrases such as face mask, dotted headscarf, football and goalkeeper. Put spatial or action phrases such as behind another person or holding the ball in relation_intents, not SAM concept names. Each operation and intent references declared concepts. Use up to three short perception operations about distinct evidence gaps and one final answer operation that repeats the original question. Relations and attributes are investigative intents, not assertions. Empty intent lists are valid if the question needs no such check. Do not answer the question yet."""

_VERIFY_DESIGN = """Inspect the image and candidate region crops. The top panel is the original scene with labeled bounding boxes. The smaller panels show retained SAM3 foreground pixels against gray. Treat candidate labels and plan relations as hypotheses. Determine whether the exact region pairs visually support the relation, or whether the image contradicts it or leaves it uncertain. Spatial contact and overlapping masks alone do not support semantic properties such as holding, wearing or part of.
Return only one JSON object:
{"relations":[{"subject_id":"r0001","predicate":"short relation","object_id":"r0002",
               "state":"supported","evidence_ids":["ev0001","ev0002"],
               "confidence":null,
               "visual_evidence":"specific observation of the visible pixels"}],
 "attribute_observations":[{"region_id":"r0002","attribute":"color","value":"visible attribute or null","evidence_id":"ev0002","visual_evidence":"observable clue"}],
 "expand_concepts":[{"concept":"alternative visible noun phrase","for_concept":"declared concept being investigated"}],"stop":false}
Evaluate each relation intent that lists distinct candidate endpoints. Inspect the named isolated crops themselves and report a verdict even if it is uncertain. If a crop shows clothing on a different person, do not give it the target person's appearance or color from the overview. Mark the binding uncertain or contradicted and request a more specific SAM concept. Only use displayed region/evidence IDs and cite both endpoints for a supported or contradicted relation. For each visible attribute intent, include an attribute_observations item with region_id, attribute, value (null if unclear), evidence_id, visual_evidence, and optional confidence. Confidence can be a self-assessment from 0 to 1, or null. If a question-relevant intent lacks reliable candidates, request a few new short SAM concepts for those targets, and set stop to false. Do not claim the evidence is sufficient while relevant intents remain unresolved. Do not answer the original question here."""


@dataclass(frozen=True)
class GraphBudget:
    max_vlm_calls: int = 3
    max_sam_calls: int = 2
    max_concepts: int = 12
    max_regions: int = 20
    max_relations: int = 32
    max_attribute_intents: int = 16
    max_evidence_regions: int = 12
    max_operations: int = 6

    def __post_init__(self):
        for field, limit in asdict(self).items():
            if type(limit) is not int or limit < 1:
                raise ValueError(f"{field} must be a positive integer")


@dataclass(frozen=True)
class SemanticEdge:
    edge_id: str
    subject_id: str
    predicate: str
    object_id: str
    state: str
    evidence_ids: tuple[str, ...]
    visual_evidence: str
    round_index: int
    source: str = "vlm_visual_verifier"
    confidence: float | None = None
    confidence_source: str | None = None
    geometry_review: dict | None = None


@dataclass
class AgenticGraph:
    image_hash: str
    plan: dict
    masks: list[MaskInput]
    regions: list[dict]
    evidence: list[dict]
    semantic_edges: list[SemanticEdge]
    trace: dict
    evidence_images: dict[str, Image.Image]
    attribute_observations: list[dict] = field(default_factory=list)

    def manifest(self, memory: ImageMemory | None = None) -> dict:
        """Serialize observations and optionally bind them to persisted memory IDs."""
        regions = self.regions
        evidence = self.evidence
        edges = [asdict(edge) for edge in self.semantic_edges]
        attributes = self.attribute_observations
        result = {
            "schema_version": 1,
            "image_hash": self.image_hash,
            "plan": self.plan,
            "trace": self.trace,
            "region_id_basis": "agentic_graph",
        }
        if memory is not None:
            if memory.image_hash != self.image_hash:
                raise ValueError("Graph and KV memory refer to different source images")
            by_mask = {row.mask_sha256: row.region_id for row in memory.regions}
            if len(by_mask) != len(memory.regions):
                raise ValueError(
                    "Mask SHA-256 does not uniquely identify regions in this memory"
                )
            ids = {}
            for row in regions:
                current = by_mask.get(row["mask_sha256"])
                if current is None:
                    raise ValueError(
                        f"Graph mask {row['region_id']} is missing in KV memory"
                    )
                ids[row["region_id"]] = current
            regions = [
                {
                    **row,
                    "region_id": ids[row["region_id"]],
                    "agent_region_id": row["region_id"],
                }
                for row in regions
            ]
            evidence = [{**row, "region_id": ids[row["region_id"]]} for row in evidence]
            edges = [
                {
                    **row,
                    "subject_id": ids[row["subject_id"]],
                    "object_id": ids[row["object_id"]],
                }
                for row in edges
            ]
            attributes = [
                {**row, "region_id": ids[row["region_id"]]} for row in attributes
            ]
            result.update(
                region_id_basis="kv_memory",
                memory_fingerprint=memory.fingerprint,
                source_fingerprints={
                    key: memory.fingerprints[key]
                    for key in ("model", "processor", "source_prefix", "capture")
                    if key in memory.fingerprints
                },
            )
        result.update(
            regions=regions,
            evidence=evidence,
            semantic_edges=edges,
            attribute_observations=attributes,
        )
        # Return JSON-compatible values directly for callers that consume the dict in memory.
        result = json.loads(json_bytes(result))
        return {**result, "fingerprint": content_fingerprint(result)}


def _read_object(value, context: str) -> dict:
    if isinstance(value, str):
        if len(value.encode("utf-8")) > 128 * 1024:
            raise ValueError(f"{context} exceeded the JSON response limit")
        text = value.strip()
        fenced = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
        )
        if fenced:
            text = fenced[1]
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context} did not return one JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} did not return one JSON object")
    encoded = json_bytes(value)
    if len(encoded) > 128 * 1024:
        raise ValueError(f"{context} exceeded the JSON response limit")
    return json.loads(encoded)


def _text(value, field: str, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise ValueError(
            f"{field} must be nonempty text of at most {max_length} characters"
        )
    return " ".join(value.split())


def _concept(value) -> str:
    text = _text(value, "SAM3 concept", 96)
    if any(character in value for character in "?!;\n\r") or len(text.split()) > 12:
        raise ValueError("SAM3 concepts must be short noun phrases")
    return text


def _bounded_concepts(rows, *, limit: int) -> tuple[list[str], list[dict]]:
    if not isinstance(rows, list):
        raise ValueError("concepts must be an array of SAM3 noun phrases")
    concepts, rejected = [], []
    seen = set()
    for index, value in enumerate(rows):
        try:
            name = _concept(value)
        except ValueError as exc:
            rejected.append({"index": index, "reason": str(exc)})
            continue
        key = name.casefold()
        if key in seen:
            rejected.append(
                {"index": index, "concept": name, "reason": "duplicate_concept"}
            )
        elif len(concepts) >= limit:
            rejected.append(
                {"index": index, "concept": name, "reason": "concept_budget"}
            )
        else:
            concepts.append(name)
            seen.add(key)
    return concepts, rejected


def normalize_plan(
    plan: dict | str, *, question: str, budget: GraphBudget = GraphBudget()
) -> dict:
    """Resolve concept references and admit only bounded question operations."""
    data = _read_object(plan, "Graph plan")
    wanted = _text(question, "question", 1024)
    if "question" in data and _text(data["question"], "plan question", 1024) != wanted:
        raise ValueError("Graph plan belongs to a different question")
    concepts, rejected = _bounded_concepts(
        data.get("concepts"), limit=budget.max_concepts
    )
    if not concepts:
        raise ValueError("Graph plan has no valid SAM3 concepts")
    by_name = {name.casefold(): name for name in concepts}
    intentions, attributes, operations = [], [], []
    seen_intentions, seen_operations = set(), set()
    for key, target, limit in (
        ("relation_intents", intentions, budget.max_relations),
        ("attribute_intents", attributes, budget.max_attribute_intents),
        ("operations", operations, budget.max_operations),
    ):
        rows = data.get(key, [])
        if not isinstance(rows, list):
            raise ValueError(f"plan {key} must be an array")
        for index, row in enumerate(rows):
            try:
                if len(target) >= limit:
                    raise ValueError("budget_exceeded")
                if not isinstance(row, dict):
                    raise ValueError("expected an object")
                if key == "relation_intents":
                    subject = by_name[_concept(row.get("subject")).casefold()]
                    object_ = by_name[_concept(row.get("object")).casefold()]
                    predicate = _text(row.get("predicate"), "predicate", 64).casefold()
                    signature = (subject, predicate, object_)
                    if signature in seen_intentions:
                        raise ValueError("duplicate_intent")
                    intent = {
                        "subject": subject,
                        "predicate": predicate,
                        "object": object_,
                    }
                    if row.get("reason") is not None:
                        intent["reason"] = _text(row["reason"], "intent reason", 256)
                    target.append(intent)
                    seen_intentions.add(signature)
                elif key == "attribute_intents":
                    concept = by_name[_concept(row.get("concept")).casefold()]
                    attribute = _text(row.get("attribute"), "attribute", 64).casefold()
                    signature = (concept, attribute)
                    if signature in seen_intentions:
                        raise ValueError("duplicate_intent")
                    intent = {"concept": concept, "attribute": attribute}
                    if row.get("reason") is not None:
                        intent["reason"] = _text(row["reason"], "intent reason", 256)
                    target.append(intent)
                    seen_intentions.add(signature)
                else:
                    operation_id = _text(row.get("id"), "operation ID", 48)
                    if (
                        not re.fullmatch(r"[A-Za-z0-9_-]+", operation_id)
                        or operation_id in seen_operations
                    ):
                        raise ValueError("invalid or duplicate operation ID")
                    prompt = _text(row.get("question"), "operation question", 1024)
                    kind = row.get("kind", "perception")
                    if kind not in ("perception", "answer") or (
                        kind == "answer" and prompt != wanted
                    ):
                        raise ValueError(
                            "invalid operation kind or mismatched final question"
                        )
                    if kind == "answer" and any(
                        item["kind"] == "answer" for item in target
                    ):
                        raise ValueError("duplicate final answer operation")
                    names = row.get("concepts")
                    if not isinstance(names, list) or not names:
                        raise ValueError("operation must reference declared concepts")
                    names = list(
                        dict.fromkeys(
                            by_name[_concept(name).casefold()] for name in names
                        )
                    )
                    target.append(
                        {
                            "id": operation_id,
                            "question": prompt,
                            "kind": kind,
                            "concepts": names,
                        }
                    )
                    seen_operations.add(operation_id)
            except (KeyError, ValueError) as exc:
                rejected.append({"field": key, "index": index, "reason": str(exc)})
    source = data.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("plan source must be an object")
    prior_rejected = data.get("filtered", [])
    if not isinstance(prior_rejected, list):
        raise ValueError("plan filtered records must be an array")
    operations.sort(key=lambda item: item["kind"] == "answer")
    return {
        "question": wanted,
        "concepts": concepts,
        "relation_intents": intentions,
        "attribute_intents": attributes,
        "operations": operations,
        "source": source,
        "filtered": [*prior_rejected, *rejected],
    }


def _messages(image: Image.Image, prompt: str) -> list[dict]:
    packed = io.BytesIO()
    image.save(packed, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(packed.getvalue()).decode(
        "ascii"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def plan_scene_graph(
    image: Image.Image,
    *,
    question: str,
    reason: Reason,
    observations=None,
    budget: GraphBudget = GraphBudget(),
    validation_feedback: str | None = None,
) -> dict:
    """Ask the supplied VLM for the acquisition plan once."""
    question = _text(question, "question", 1024)
    preview = ImageOps.exif_transpose(image).convert("RGB")
    preview.thumbnail((840, 640), Image.Resampling.LANCZOS)
    initial_concepts = budget.max_concepts
    if budget.max_sam_calls > 1:
        initial_concepts -= min(4, max(0, initial_concepts - 4))
    context = {
        "question": question,
        "budget": {
            "concepts": initial_concepts,
            "operations": budget.max_operations,
            "relation_intents": budget.max_relations,
            "attribute_intents": budget.max_attribute_intents,
        },
    }
    if observations is not None:
        encoded = json_bytes(observations)
        if len(encoded) > 4096:
            raise ValueError("Planner observations exceed 4096 JSON bytes")
        context["previous_observations"] = observations
    if validation_feedback is not None:
        context["repair_previous_response"] = validation_feedback
    output = _read_object(
        reason(
            _messages(
                preview, _PLAN_DESIGN + "\n" + json.dumps(context, ensure_ascii=False)
            )
        ),
        "Planner",
    )
    plan = normalize_plan(
        output, question=question, budget=replace(budget, max_concepts=initial_concepts)
    )
    if not plan["operations"]:
        raise ValueError("The planner supplied no valid visual operations")
    plan["source"] = {
        "kind": "vlm_generated",
        "model_output_sha256": content_fingerprint(output),
    }
    return plan


def make_sam3_segmenter(image: Image.Image, **kwargs) -> Segment:
    """Use the existing local SAM3 adapter with a bounded batch of text prompts."""
    if "masks_file" in kwargs or "sam_prompts_file" in kwargs:
        raise ValueError("The agentic planner supplies the SAM3 concepts")
    kwargs.setdefault("sam3_max_candidates", 3)

    def segment(concepts):
        with tempfile.TemporaryDirectory(prefix="vlcache-agentic-sam3-") as tmp:
            prompts = Path(tmp) / "prompts.json"
            prompts.write_bytes(json_bytes(concepts))
            return create_masks(image, sam_prompts_file=prompts, **kwargs)

    return segment


def _interleave_masks(masks):
    per_label = {}
    for index, item in enumerate(masks):
        if not isinstance(item, MaskInput):
            raise ValueError(f"SAM3 candidate {index} must be MaskInput")
        per_label.setdefault((item.label or "").casefold(), []).append((index, item))
    # Keep SAM ranking within each concept while giving each concept first access to the budget.
    for rank in range(max((len(rows) for rows in per_label.values()), default=0)):
        for rows in per_label.values():
            if rank < len(rows):
                yield rows[rank]


def _near_duplicate(mask, bbox, area, masks, regions):
    best_index, best_iou = None, _INSTANCE_MASK_IOU
    for index, region in enumerate(regions):
        prior_area = region["area"]
        if min(area, prior_area) / max(area, prior_area) < best_iou:
            continue
        x0, y0, x1, y1 = region["bbox_xyxy"]
        left, top = max(x0, bbox[0]), max(y0, bbox[1])
        right, bottom = min(x1, bbox[2]), min(y1, bbox[3])
        if left >= right or top >= bottom:
            continue
        window = np.s_[top:bottom, left:right]
        overlap = int(np.count_nonzero(mask[window] & masks[index].mask[window]))
        iou = overlap / (area + prior_area - overlap)
        if iou >= best_iou and (best_index is None or iou > best_iou):
            best_index, best_iou = index, iou
    return best_index, best_iou


def _add_masks(image, candidates, masks, regions, evidence, known_masks, max_regions):
    rejected = []
    ordered = list(_interleave_masks(candidates))
    for index, item in ordered:
        data = item.mask
        if (
            not isinstance(data, np.ndarray)
            or data.shape != (image.height, image.width)
            or data.dtype != np.bool_
        ):
            raise ValueError(
                f"SAM3 candidate {index} must be an oriented full-image boolean array"
            )
    for index, item in ordered:
        data = item.mask
        bbox = mask_bbox(data)
        if bbox is None:
            rejected.append({"index": index, "reason": "empty_mask"})
            continue
        digest = mask_sha256(data)
        area = int(np.count_nonzero(data))
        previous, overlap = (
            (known_masks[digest], 1.0)
            if digest in known_masks
            else _near_duplicate(data, bbox, area, masks, regions)
        )
        if previous is not None:
            rejected.append(
                {
                    "index": index,
                    "reason": "duplicate_mask" if overlap == 1 else "near_duplicate_mask",
                    "duplicate_of": regions[previous]["region_id"],
                    "mask_iou": overlap,
                }
            )
            metadata = masks[previous].metadata
            names = [
                *metadata.get("prompt_aliases", []),
                *item.metadata.get("prompt_aliases", []),
                item.label,
            ]
            aliases = list(
                dict.fromkeys(
                    name for name in names if name and name != masks[previous].label
                )
            )
            if aliases:
                masks[previous] = replace(
                    masks[previous], metadata={**metadata, "prompt_aliases": aliases}
                )
                regions[previous]["prompt_aliases"] = aliases
            continue
        if len(masks) >= max_regions:
            rejected.append({"index": index, "reason": "region_budget"})
            continue
        known_masks[digest] = len(masks)
        ordinal = len(masks) + 1
        region_id, evidence_id = f"r{ordinal:04d}", f"ev{ordinal:04d}"
        masks.append(replace(item, mask=data.copy(order="C")))
        region = {
            "region_id": region_id,
            "mask_sha256": digest,
            "bbox_xyxy": bbox,
            "area": area,
            "label": item.label,
            "label_source": item.label_source,
            "score": item.score,
            "score_source": item.metadata.get("score_type")
            if item.score is not None
            else None,
            "evidence_id": evidence_id,
        }
        if aliases := item.metadata.get("prompt_aliases"):
            region["prompt_aliases"] = list(aliases)
        regions.append(region)
        evidence.append(
            {
                "evidence_id": evidence_id,
                "kind": "sam3_mask_crop",
                "region_id": region_id,
                "mask_sha256": digest,
                "bbox_xyxy": bbox,
            }
        )
    return rejected


def _evidence_board(image, masks, regions, indices):
    scene = image.copy()
    scene.thumbnail((768, 320), Image.Resampling.LANCZOS)
    scale_x, scale_y = scene.width / image.width, scene.height / image.height
    context = ImageDraw.Draw(scene)
    rows = (len(indices) + 3) // 4
    board = Image.new("RGB", (768, scene.height + 120 * rows), "#151515")
    tags = ImageDraw.Draw(board)
    for tile, index in enumerate(indices):
        row, item = regions[index], masks[index]
        x0, y0, x1, y1 = row["bbox_xyxy"]
        xyxy = (
            round(x0 * scale_x),
            round(y0 * scale_y),
            round(x1 * scale_x),
            round(y1 * scale_y),
        )
        context.rectangle(xyxy, outline="#14edf8", width=1)
        context.text(
            (xyxy[0], max(0, xyxy[1] - 10)),
            row["region_id"],
            fill="#14edf8",
            stroke_width=1,
            stroke_fill="#141414",
        )
        foreground = image.crop((x0, y0, x1, y1))
        mask = Image.fromarray(item.mask[y0:y1, x0:x1])
        crop = Image.new("RGB", foreground.size, "#555555")
        crop.paste(foreground, mask=mask)
        scale = min(184 / crop.width, 100 / crop.height, 4)
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
            Image.Resampling.NEAREST if scale >= 1 else Image.Resampling.LANCZOS,
        )
        left, top = 192 * (tile % 4), scene.height + 120 * (tile // 4)
        board.paste(crop, (left + (192 - crop.width) // 2, top + 16))
        tags.text(
            (left + 4, top + 2),
            f"{row['region_id']}  {row['evidence_id']}",
            fill="white",
        )
    board.paste(scene, (0, 0))
    return board


def _region_labels(region):
    return {
        (region["label"] or "").casefold(),
        *(alias.casefold() for alias in region.get("prompt_aliases", [])),
    }


def _displayed_indices(regions, first_new, plan, limit):
    relevant = {
        concept.casefold()
        for row in plan["relation_intents"]
        for concept in (row["subject"], row["object"])
    }
    relevant.update(row["concept"].casefold() for row in plan["attribute_intents"])
    order = [*range(first_new, len(regions)), *range(first_new)]
    order.sort(key=lambda index: not bool(_region_labels(regions[index]) & relevant))
    return order[:limit]


def _intent_targets(plan, displayed):
    def candidates(concept):
        label = concept.casefold()
        return [row["region_id"] for row in displayed if label in _region_labels(row)]

    relations = [
        {
            "intent_id": f"relation_{i:02d}",
            **intent,
            "subject_candidates": candidates(intent["subject"]),
            "object_candidates": candidates(intent["object"]),
        }
        for i, intent in enumerate(plan["relation_intents"], 1)
    ]
    attributes = [
        {
            "intent_id": f"attribute_{i:02d}",
            **intent,
            "candidates": candidates(intent["concept"]),
        }
        for i, intent in enumerate(plan["attribute_intents"], 1)
    ]
    return relations, attributes


def _confidence(value):
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
    ):
        raise ValueError("semantic confidence must be a number in [0,1] or null")
    return float(value)


def _predicate(value):
    text = _text(value, "predicate", 64).casefold()
    return re.sub(r"^(?:is|are|was|were)\s+", "", text)


def _targeted_expansion(rows, declared, limit):
    concepts, targets, rejected = [], {}, []
    seen = set()
    for index, row in enumerate(rows):
        try:
            concept = _concept(row.get("concept") if isinstance(row, dict) else row)
            key = concept.casefold()
            if key in seen or len(concepts) >= limit:
                raise ValueError(
                    "duplicate_concept" if key in seen else "concept_budget"
                )
            if isinstance(row, dict):
                target = _concept(row.get("for_concept")).casefold()
                if target not in declared:
                    raise ValueError("Expansion target must be a declared plan concept")
                targets[key] = declared[target]
            concepts.append(concept)
            seen.add(key)
        except ValueError as exc:
            rejected.append({"index": index, "reason": str(exc)})
    return concepts, targets, rejected


def _mark_expansion_targets(candidates, targets):
    result = []
    for item in candidates:
        if isinstance(item, MaskInput) and (
            target := targets.get((item.label or "").casefold())
        ):
            metadata = item.metadata
            aliases = list(dict.fromkeys([*metadata.get("prompt_aliases", []), target]))
            item = replace(
                item,
                metadata={
                    **metadata,
                    "prompt_aliases": aliases,
                    "targeted_expansion_for": target,
                },
            )
        result.append(item)
    return result


def _attribute_observations(rows, evidence_by_region, targets, limit):
    if not isinstance(rows, list):
        raise ValueError("attribute_observations must be an array")
    allowed = {
        (region, target["attribute"])
        for target in targets
        for region in target["candidates"]
    }
    results, rejected = [], []
    for index, row in enumerate(rows):
        try:
            if len(results) >= limit:
                raise ValueError("attribute_budget")
            if not isinstance(row, dict):
                raise ValueError("expected an attribute object")
            region_id = _text(row.get("region_id"), "region ID", 32)
            attribute = _text(row.get("attribute"), "attribute", 64).casefold()
            if (region_id, attribute) not in allowed:
                raise ValueError(
                    "attribute must match an intent and a displayed region"
                )
            evidence_id = row.get("evidence_id")
            if evidence_id != evidence_by_region[region_id]:
                raise ValueError(
                    "attribute evidence must identify the displayed region crop"
                )
            value = row.get("value")
            confidence = _confidence(row.get("confidence"))
            results.append(
                {
                    "region_id": region_id,
                    "attribute": attribute,
                    "value": _text(value, "value", 128) if value is not None else None,
                    "evidence_id": evidence_id,
                    "visual_evidence": _text(
                        row.get("visual_evidence"), "visual evidence", 512
                    ),
                    "source": "vlm_visual_verifier",
                    "confidence": confidence,
                    "confidence_source": "vlm_self_reported"
                    if confidence is not None
                    else None,
                }
            )
        except ValueError as exc:
            rejected.append({"index": index, "reason": str(exc)})
    return results, rejected


def _unresolved_intents(relations, attributes, assessments, observations):
    unresolved = [
        row["intent_id"]
        for row in relations
        if not any(
            state != "uncertain"
            and subject in row["subject_candidates"]
            and object_ in row["object_candidates"]
            and predicate == _predicate(row["predicate"])
            for subject, predicate, object_, state, *_ in assessments
        )
    ]
    unresolved.extend(
        row["intent_id"]
        for row in attributes
        if not any(
            item["value"] is not None
            and item["attribute"] == row["attribute"]
            and item["region_id"] in row["candidates"]
            for item in observations
        )
    )
    return unresolved


def _verification(value, displayed, attribute_targets, declared, budget):
    response = _read_object(value, "Semantic verifier")
    rows, expand, stop = (
        response.get("relations", []),
        response.get("expand_concepts", []),
        response.get("stop", False),
    )
    if (
        not isinstance(rows, list)
        or not isinstance(expand, list)
        or type(stop) is not bool
    ):
        raise ValueError(
            "Semantic verifier requires relations and expand_concepts arrays and a boolean stop"
        )
    evidence_by_region = {row["region_id"]: row["evidence_id"] for row in displayed}
    assessments, rejected = [], []
    for index, row in enumerate(rows):
        try:
            if len(assessments) >= budget.max_relations:
                raise ValueError("relation_budget")
            if not isinstance(row, dict):
                raise ValueError("expected a relation object")
            subject = _text(row.get("subject_id"), "subject ID", 32)
            object_ = _text(row.get("object_id"), "object ID", 32)
            if (
                subject == object_
                or subject not in evidence_by_region
                or object_ not in evidence_by_region
            ):
                raise ValueError(
                    "relation endpoints must be distinct displayed region IDs"
                )
            state = row.get("state")
            if not isinstance(state, str) or state not in _STATES:
                raise ValueError("invalid semantic verification state")
            predicate = _predicate(row.get("predicate"))
            observation = _text(row.get("visual_evidence"), "visual evidence", 512)
            refs = row.get("evidence_ids", [])
            expected = {evidence_by_region[subject], evidence_by_region[object_]}
            if not isinstance(refs, list) or any(
                not isinstance(ref, str) or ref not in expected for ref in refs
            ):
                raise ValueError(
                    "evidence IDs must identify displayed crops for the two endpoints"
                )
            if state != "uncertain" and set(refs) != expected:
                raise ValueError(
                    "decisive relation requires evidence for both endpoints"
                )
            confidence = _confidence(row.get("confidence"))
            assessments.append(
                (
                    subject,
                    predicate,
                    object_,
                    state,
                    tuple(sorted(set(refs))),
                    observation,
                    confidence,
                )
            )
        except ValueError as exc:
            rejected.append({"index": index, "reason": str(exc)})
    expansions, expansion_targets, filtered = _targeted_expansion(
        expand, declared, budget.max_concepts
    )
    attributes, rejected_attributes = _attribute_observations(
        response.get("attribute_observations", []),
        evidence_by_region,
        attribute_targets,
        budget.max_attribute_intents,
    )
    return (
        assessments,
        expansions,
        expansion_targets,
        stop,
        rejected,
        filtered,
        attributes,
        rejected_attributes,
        content_fingerprint(response),
    )


def _attachment_reviews(assessments, displayed):
    """Flag disjoint attachment proposals without inferring any semantic edges."""
    boxes = {row["region_id"]: row["bbox_xyxy"] for row in displayed}
    reviews = {}
    for subject, predicate, object_, state, _, evidence, confidence in assessments:
        if state != "supported" or predicate.replace("_", " ") not in (
            "wearing",
            "wears",
            "part of",
        ):
            continue
        a, b = boxes[subject], boxes[object_]
        gap_x = max(0, max(a[0], b[0]) - min(a[2], b[2]))
        gap_y = max(0, max(a[1], b[1]) - min(a[3], b[3]))
        if gap_x or gap_y:
            reviews[subject, predicate, object_] = {
                "rule": "disjoint_attachment_bboxes",
                "action": "downgraded_to_uncertain",
                "bbox_gap_xy": [gap_x, gap_y],
                "original_state": state,
                "original_confidence": confidence,
                "original_visual_evidence": evidence,
            }
    return reviews


def _update_edges(edges, assessments, limit, round_index, geometry_reviews):
    rejected = []
    for (
        subject,
        predicate,
        object_,
        state,
        refs,
        observation,
        confidence,
    ) in assessments:
        key = (subject, predicate, object_)
        previous = edges.get(key)
        review = geometry_reviews.get(key) if state == "supported" else None
        if review is not None:
            state = "uncertain"
            confidence = None
            observation = (
                "The visible attachment candidates have disjoint mask bounding boxes."
            )
        if previous is None and len(edges) >= limit:
            rejected.append(
                {
                    "endpoints": [subject, object_],
                    "predicate": predicate,
                    "reason": "relation_budget",
                }
            )
            continue
        edge_id = previous.edge_id if previous else f"s{len(edges) + 1:04d}"
        if previous is not None and previous.state != state:
            # Keep both round results in the trace and preserve their disagreement in the graph.
            state = "uncertain"
            refs = tuple(sorted(set(refs) | set(previous.evidence_ids)))
            observation = "Verification rounds gave different assessments. See round observations."
            confidence = None
        edges[key] = SemanticEdge(
            edge_id,
            subject,
            predicate,
            object_,
            state,
            refs,
            observation,
            round_index,
            confidence=confidence,
            confidence_source="vlm_self_reported" if confidence is not None else None,
            geometry_review=review,
        )
    return rejected


def build_agentic_graph(
    image: Image.Image,
    *,
    question: str,
    reason: Reason,
    segment: Segment,
    existing_plan: dict | str | None = None,
    observations=None,
    budget: GraphBudget = GraphBudget(),
) -> AgenticGraph:
    """Acquire masks, verify directed relations with visual evidence and expand within budget."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    question = _text(question, "question", 1024)
    vlm_calls = 0
    planner_attempts = []

    def counted_reason(messages):
        nonlocal vlm_calls
        vlm_calls += 1
        return reason(messages)

    if existing_plan is None:
        plan = {
            "question": question,
            "concepts": [],
            "relation_intents": [],
            "attribute_intents": [],
            "operations": [],
            "source": {"kind": "planner_unavailable"},
            "filtered": [],
        }
        feedback = None
        for _ in range(min(2, budget.max_vlm_calls)):
            calls_before = vlm_calls
            try:
                plan = plan_scene_graph(
                    image,
                    question=question,
                    reason=counted_reason,
                    observations=observations,
                    budget=budget,
                    validation_feedback=feedback,
                )
            except (TypeError, ValueError, RuntimeError, OSError) as exc:
                if vlm_calls == calls_before:
                    raise
                feedback = str(exc)[:2048]
                planner_attempts.append(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "message": feedback,
                    }
                )
                if isinstance(exc, (RuntimeError, OSError)):
                    break
            else:
                planner_attempts.append({"status": "valid"})
                break
    else:
        plan = normalize_plan(existing_plan, question=question, budget=budget)
    masks, regions, evidence = [], [], []
    known_masks, edges, boards = {}, {}, {}
    attribute_observations = []
    pending = plan["concepts"].copy()
    declared = {concept.casefold(): concept for concept in pending}
    pending_targets = {}
    used_concepts, rounds = [], []
    stop_reason = "sam_budget" if pending else "planner_error"
    for round_index in range(1, budget.max_sam_calls + 1):
        if not pending:
            break
        used_concepts.extend(pending)
        first_new = len(regions)
        record = {
            "round_index": round_index,
            "concepts": pending,
        }
        rounds.append(record)
        cap = budget.max_regions
        if (
            round_index == 1
            and budget.max_sam_calls > 1
            and len(used_concepts) < budget.max_concepts
            and vlm_calls < budget.max_vlm_calls
        ):
            # Leave room for a small targeted acquisition after inspecting these proposals.
            cap -= min(3, cap // 4)
        try:
            candidates, segmentation = segment(pending)
            if not isinstance(candidates, list) or not isinstance(segmentation, dict):
                raise ValueError(
                    "Segmentation must return a list of masks and a provenance object"
                )
            candidates = _mark_expansion_targets(candidates, pending_targets)
            room = _add_masks(
                image, candidates, masks, regions, evidence, known_masks, cap
            )
        except (TypeError, ValueError, RuntimeError, OSError) as exc:
            record["segmentation_error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:2048],
            }
            stop_reason = "sam_error"
            break
        record.update(
            segmentation=segmentation,
            filtered_masks=room,
            new_region_ids=[row["region_id"] for row in regions[first_new:]],
        )
        if vlm_calls == budget.max_vlm_calls:
            stop_reason = "vlm_budget"
            break
        indices = _displayed_indices(
            regions, first_new, plan, budget.max_evidence_regions
        )
        displayed = [regions[index] for index in indices]
        relation_targets, attribute_targets = _intent_targets(plan, displayed)
        board = _evidence_board(image, masks, regions, indices)
        board_id = f"verification_{round_index:02d}"
        boards[board_id] = board
        record["board"] = {
            "id": board_id,
            "image_hash": image_sha256(board),
            "evidence_ids": [row["evidence_id"] for row in displayed],
        }
        context = {
            "question": plan["question"],
            "used_concepts": used_concepts,
            "relation_intents": relation_targets,
            "attribute_intents": attribute_targets,
            "perception_questions": [
                row["question"]
                for row in plan["operations"]
                if row["kind"] == "perception"
            ],
            "displayed_regions": [
                {key: row[key] for key in ("region_id", "evidence_id", "label")}
                for row in displayed
            ],
            "remaining_sam_calls": budget.max_sam_calls - round_index,
            "remaining_concepts": budget.max_concepts - len(used_concepts),
            "max_relations": budget.max_relations,
        }
        try:
            response = counted_reason(
                _messages(
                    board,
                    _VERIFY_DESIGN
                    + "\n"
                    + json.dumps(context, ensure_ascii=False, separators=(",", ":")),
                )
            )
        except (RuntimeError, OSError) as exc:
            record["verification_error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:2048],
            }
            stop_reason = "verification_error"
            break
        try:
            (
                assessments,
                expansion,
                expansion_targets,
                stop,
                rejected,
                filtered_expansion,
                attributes,
                rejected_attributes,
                digest,
            ) = _verification(response, displayed, attribute_targets, declared, budget)
        except (TypeError, ValueError) as exc:
            record["verification_error"] = str(exc)
            stop_reason = "invalid_verification"
            break
        for attribute in attributes:
            attribute["round_index"] = round_index
        attribute_observations.extend(attributes)
        geometry_reviews = _attachment_reviews(assessments, displayed)
        rejected.extend(
            _update_edges(
                edges, assessments, budget.max_relations, round_index, geometry_reviews
            )
        )
        current_edges = [
            (edge.subject_id, edge.predicate, edge.object_id, edge.state)
            for edge in edges.values()
        ]
        unresolved = _unresolved_intents(
            relation_targets, attribute_targets, current_edges, attribute_observations
        )
        record.update(
            response_sha256=digest,
            assessments=assessments,
            rejected_relations=rejected,
            filtered_expansion=filtered_expansion,
            requested_expansion=expansion,
            expansion_targets=expansion_targets,
            attribute_observations=attributes,
            rejected_attributes=rejected_attributes,
            unresolved_intents=unresolved,
            geometry_reviews=[
                {
                    "subject_id": key[0],
                    "predicate": key[1],
                    "object_id": key[2],
                    **review,
                }
                for key, review in geometry_reviews.items()
            ],
        )
        used_keys = {value.casefold() for value in used_concepts}
        remaining_concepts = budget.max_concepts - len(used_concepts)
        pending = []
        pending_targets = {}
        for value in expansion:
            if value.casefold() in used_keys:
                filtered_expansion.append(
                    {"concept": value, "reason": "already_segmented"}
                )
            elif len(pending) >= remaining_concepts:
                filtered_expansion.append(
                    {"concept": value, "reason": "concept_budget"}
                )
            else:
                pending.append(value)
                if value.casefold() in expansion_targets:
                    pending_targets[value.casefold()] = expansion_targets[
                        value.casefold()
                    ]
        if (stop and not unresolved) or not pending:
            if unresolved:
                stop_reason = "unresolved_intents"
            elif stop:
                stop_reason = "verifier_stop"
            else:
                stop_reason = (
                    "no_new_concepts" if remaining_concepts else "concept_budget"
                )
            break
        if len(masks) == budget.max_regions:
            stop_reason = "region_budget"
            break
    trace = {
        "budget": asdict(budget),
        "vlm_calls": vlm_calls,
        "planner_attempts": planner_attempts,
        "sam_calls": len(rounds),
        "used_concepts": used_concepts,
        "stop_reason": stop_reason,
        "rounds": rounds,
    }
    return AgenticGraph(
        image_sha256(image),
        plan,
        masks,
        regions,
        evidence,
        list(edges.values()),
        trace,
        boards,
        attribute_observations,
    )
