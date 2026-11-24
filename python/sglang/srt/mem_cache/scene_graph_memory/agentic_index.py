"""Bind verified graph evidence to original decoder KV rows and retrieve them."""

from __future__ import annotations

import json
import re
import unicodedata
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from .schema import BANK_REF, ImageMemory, KVSlice, content_fingerprint
from .store import BANK, MANIFEST, load_memory


_ALIASES = (
    ("face mask", "surgical mask", "medical mask", "口罩"),
    ("headscarf", "head scarf", "hijab", "头巾"),
    ("woman", "women", "女子", "女人", "女士"),
    ("man", "men", "男子", "男人"),
    ("plaid coat", "checkered coat", "格子外套", "格纹外套"),
    ("tomato", "tomatoes", "番茄", "西红柿"),
    ("paper money", "banknote", "bank note", "bill", "纸币", "钞票"),
    ("soccer ball", "football", "足球"),
    ("goalkeeper", "goalie", "守门员", "门将"),
    ("wearing", "wears", "wear", "wore", "戴着", "穿着"),
    ("holding", "holds", "hold", "held", "拿着", "握着"),
    ("next to", "beside", "adjacent to", "旁边"),
    ("behind", "in back of", "身后", "后面"),
    ("in front of", "前面"),
    ("handing over", "passing", "giving", "递给"),
    ("light blue", "pale blue", "浅蓝色", "淡蓝色"),
    ("beige", "米色"),
)
_ALIAS_BY_TERM = {term: frozenset(group) for group in _ALIASES for term in group}
_SOURCE_IDENTITY = ("model", "processor", "source_prefix", "capture")


def _text(value):
    if not isinstance(value, str):
        return ""
    return " ".join(
        re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", value).casefold())
    )


def _contains(text, term):
    if not term:
        return False
    if any(ord(char) > 127 for char in term):
        return term in text
    return f" {term} " in f" {text} "


def _terms(values):
    names = {_text(value) for value in values} - {""}
    return names | set().union(*(_ALIAS_BY_TERM.get(name, ()) for name in names))


def _predicate_terms(value):
    words = _text(value).split()
    if words and words[0] in {"is", "are", "was", "were", "be", "being"}:
        words = words[1:]
    return frozenset(_terms([" ".join(words)]))


def _items(value):
    if isinstance(value, str):
        return [value]
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("Expected an array of terms")
    return value


def _name(item):
    if isinstance(item, str):
        return item
    for key in ("concept", "text", "name", "label"):
        if isinstance(item.get(key), str) and item[key].strip():
            return item[key]
    raise ValueError("Every query concept must have a name")


def _attributes(value):
    if isinstance(value, dict):
        return {
            _text(key): frozenset(_terms(_items(item)))
            for key, item in value.items()
            if _text(key)
        }
    return {"": frozenset(_terms(_items(value)))} if value else {}


@dataclass(frozen=True)
class _Concept:
    name: str
    terms: frozenset[str]
    attributes: dict[str, frozenset[str]]
    required: bool = True

    @classmethod
    def parse(cls, item, required=True):
        if not isinstance(item, (str, dict)):
            raise ValueError("A concept must be a string or object")
        name = _name(item)
        options = item if isinstance(item, dict) else {}
        terms = frozenset(_terms([name, *_items(options.get("aliases"))]))
        if not terms:
            raise ValueError("Concept is empty")
        return cls(
            name,
            terms,
            _attributes(options.get("attributes")),
            _required(options, required),
        )


def _required(options, default=True):
    value = options.get("required", default)
    if not isinstance(value, bool):
        raise ValueError("Evidence required flag must be a boolean")
    return value


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _probability(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0 <= value <= 1
    )


@dataclass(frozen=True)
class _Relation:
    subject: _Concept | None
    predicate: _Concept
    object: _Concept | None
    required: bool

    @classmethod
    def parse(cls, row):
        if not isinstance(row, dict) or not row.get("predicate"):
            raise ValueError("A relation intent must provide a predicate")
        terms = frozenset().union(
            *(_predicate_terms(value) for value in [row["predicate"], *_items(row.get("aliases"))])
        )
        if not terms:
            raise ValueError("A relation intent must provide a predicate")
        predicate = _Concept(row["predicate"], terms, {})
        return cls(
            _Concept.parse(row["subject"]) if row.get("subject") else None,
            predicate,
            _Concept.parse(row["object"]) if row.get("object") else None,
            _required(row),
        )

    @property
    def name(self):
        return " ".join(
            x.name for x in (self.subject, self.predicate, self.object) if x
        )


@dataclass(frozen=True)
class RegionBinding:
    region_id: str
    physical_region_id: str
    label: str
    aliases: tuple[str, ...]
    attributes: dict[str, frozenset[str]]
    mask_sha256: str
    visual_indices: tuple[int, ...]
    coverage_weights: tuple[float, ...]
    score: float
    evidence_id: str


@dataclass(frozen=True)
class KVReference:
    """Immutable source coordinates and named safetensors pairs for every layer."""

    memory_fingerprint: str
    index_fingerprint: str
    bank_id: str
    bank_file: str | None
    bank_sha256: str | None
    physical_region_ids: tuple[str, ...]
    source_visual_indices: tuple[int, ...]
    source_sequence_indices: tuple[int, ...]
    mrope_positions: tuple[tuple[int, int, int], ...]
    key_representation_stage: str
    value_representation_stage: str
    layers: tuple[dict, ...]


@dataclass(frozen=True)
class RetrievalResult:
    question: str
    operation_id: str | None
    token_budget: int
    source_visual_tokens: int
    selected_nodes: tuple[dict, ...]
    selected_edges: tuple[dict, ...]
    reasons: tuple[dict, ...]
    missing_evidence: tuple[dict, ...]
    background_visual_indices: tuple[int, ...]
    fallback: dict | None
    kv_reference: KVReference

    @property
    def source_visual_indices(self):
        return self.kv_reference.source_visual_indices

    @property
    def region_ids(self):
        return self.kv_reference.physical_region_ids

    def to_dict(self):
        return asdict(self)

    def selection_dict(self):
        return {
            "memory_fingerprint": self.kv_reference.memory_fingerprint,
            "region_ids": list(self.region_ids),
            "source_visual_indices": list(self.source_visual_indices),
        }


class AgenticKVIndex:
    """An image-local index. Graph IDs and native physical region IDs may differ."""

    def __init__(
        self,
        memory: str | Path | ImageMemory,
        graph_manifest: dict,
        graph_masks: Mapping | Iterable | None = None,
        *,
        min_edge_confidence: float = 0.5,
    ):
        if not _probability(min_edge_confidence):
            raise ValueError("min_edge_confidence must be a number between 0 and 1")
        self.min_edge_confidence = float(min_edge_confidence)
        self.memory_dir = (
            None if isinstance(memory, ImageMemory) else Path(memory).resolve()
        )
        if self.memory_dir is not None:
            manifest = json.loads(
                (self.memory_dir / MANIFEST).read_text(encoding="utf-8")
            )
            memory = load_memory(
                self.memory_dir, expected_fingerprint=manifest["fingerprint"]
            )
            self.bank_sha256 = manifest["files"][BANK]["sha256"]
        else:
            memory.validate()
            self.bank_sha256 = None
        self.memory = memory
        self.memory_fingerprint = memory.fingerprint
        self.graph = json.loads(json.dumps(graph_manifest, allow_nan=False))
        self.graph_fingerprint = self.graph.get("fingerprint")
        if self.graph.get(
            "schema_version"
        ) != 1 or self.graph_fingerprint != content_fingerprint(
            {key: value for key, value in self.graph.items() if key != "fingerprint"}
        ):
            raise ValueError("Agentic graph schema or fingerprint mismatch")
        if self.graph.get("image_hash") != memory.image_hash:
            raise ValueError("Agentic graph and source memory images differ")
        if (
            self.graph.get("memory_fingerprint", self.memory_fingerprint)
            != self.memory_fingerprint
        ):
            raise ValueError(
                "Agentic graph references a different source memory fingerprint"
            )

        self.source_identity = {
            key: memory.fingerprints.get(key) for key in _SOURCE_IDENTITY
        }
        expected = self.graph.get("source_fingerprints", {})
        if not isinstance(expected, dict) or any(
            key not in self.source_identity or value != self.source_identity[key]
            for key, value in expected.items()
        ):
            raise ValueError(
                "Agentic graph source model, processor, prefix or capture fingerprint mismatch"
            )
        regions = self.graph["regions"]
        if graph_masks is not None and not isinstance(graph_masks, Mapping):
            graph_masks = tuple(graph_masks)
            if len(graph_masks) != len(regions):
                raise ValueError("Graph mask count differs from its region manifest")
            graph_masks = {
                row["region_id"]: getattr(mask, "mask", mask)
                for row, mask in zip(regions, graph_masks)
            }
        self.nodes = self._bind_regions(regions, graph_masks)
        self.evidence = self._bind_evidence(self.graph["evidence"])
        self.edges = self._bind_edges(self.graph["semantic_edges"])
        self.fingerprint = content_fingerprint(
            {
                "graph": self.graph_fingerprint,
                "memory": self.memory_fingerprint,
                "source_identity": self.source_identity,
                "source_bank_sha256": self.bank_sha256,
                "min_edge_confidence": self.min_edge_confidence,
                "physical_regions": {
                    key: value.physical_region_id
                    for key, value in sorted(self.nodes.items())
                },
            }
        )
        self.layer_refs = tuple(
            {
                "layer": int(layer),
                "key": {"tensor": f"layer.{layer}.k", **spec["key"]},
                "value": {"tensor": f"layer.{layer}.v", **spec["value"]},
            }
            for layer, spec in memory.layer_specs().items()
        )

    def _bind_regions(self, regions, graph_masks):
        physical = {region.region_id: region for region in self.memory.regions}
        by_mask = {}
        for region in physical.values():
            by_mask.setdefault(region.mask_sha256, []).append(region)
        result = {}
        for row in regions:
            node_id, digest = row["region_id"], row["mask_sha256"]
            if not isinstance(node_id, str) or not node_id or node_id in result:
                raise ValueError(f"Invalid or duplicate graph region {node_id!r}")
            candidates = by_mask.get(digest, [])
            same_id = physical.get(node_id)
            if same_id is not None and same_id.mask_sha256 == digest:
                target = same_id
            elif len(candidates) == 1:
                target = candidates[0]
            else:
                raise ValueError(
                    f"Graph region {node_id!r} has no unique physical mask binding"
                )
            if tuple(row["bbox_xyxy"]) != target.bbox_xyxy:
                raise ValueError(f"Graph region {node_id!r} has a mismatched mask bbox")
            if graph_masks is not None:
                import numpy as np

                mask_options = (
                    graph_masks.get(key)
                    for key in (node_id, row.get("agent_region_id"))
                )
                if not any(
                    isinstance(mask, np.ndarray)
                    and mask.dtype == np.bool_
                    and np.array_equal(mask, self.memory.masks[target.region_id])
                    for mask in mask_options
                ):
                    raise ValueError(
                        f"Graph region {node_id!r} mask payload differs from source"
                    )
            label = row.get("label") or target.label or ""
            aliases = [*_items(row.get("aliases")), *_items(row.get("prompt_aliases"))]
            result[node_id] = RegionBinding(
                node_id,
                target.region_id,
                label,
                tuple(sorted(_terms([label, *aliases]))),
                _attributes(row.get("attributes")),
                digest,
                target.token_indices,
                target.token_weights,
                float(row.get("score") or 0),
                row["evidence_id"],
            )
        return result

    def _bind_evidence(self, evidence):
        records = {}
        physical = {region.region_id: region for region in self.memory.regions}
        for row in evidence:
            evidence_id, region_id = row["evidence_id"], row["region_id"]
            node = self.nodes.get(region_id)
            if (
                not isinstance(evidence_id, str)
                or not evidence_id
                or evidence_id in records
                or node is None
            ):
                raise ValueError(f"Invalid region evidence {evidence_id!r}")
            target = physical[node.physical_region_id]
            if (
                row.get("kind") != "sam3_mask_crop"
                or row["mask_sha256"] != node.mask_sha256
                or tuple(row["bbox_xyxy"]) != target.bbox_xyxy
            ):
                raise ValueError(
                    f"Evidence {evidence_id!r} differs from its physical mask"
                )
            records[evidence_id] = row
        for node in self.nodes.values():
            if records.get(node.evidence_id, {}).get("region_id") != node.region_id:
                raise ValueError(
                    f"Region {node.region_id!r} is missing its primary evidence"
                )
        return records

    def _bind_edges(self, edges):
        result = {}
        rounds = self.graph.get("trace", {}).get("rounds", ())
        displayed = {
            step["round_index"]: set(step["board"]["evidence_ids"])
            for step in rounds
            if "board" in step
        }
        for row in edges:
            edge_id, a, b = row["edge_id"], row["subject_id"], row["object_id"]
            if (
                not isinstance(edge_id, str)
                or not edge_id
                or edge_id in result
                or a not in self.nodes
                or b not in self.nodes
                or a == b
                or row.get("state") not in {"supported", "uncertain", "contradicted"}
                or not _text(row.get("predicate"))
            ):
                raise ValueError(f"Invalid semantic edge {edge_id!r}")
            confidence = row.get("confidence")
            if confidence is not None and not _probability(confidence):
                raise ValueError(f"Invalid confidence on semantic edge {edge_id!r}")
            evidence_ids = row.get("evidence_ids", [])
            if not isinstance(evidence_ids, list) or any(
                value not in self.evidence for value in evidence_ids
            ):
                raise ValueError(f"Invalid evidence on semantic edge {edge_id!r}")
            evidence_nodes = {
                self.evidence[value]["region_id"] for value in evidence_ids
            }
            if row["state"] in {"supported", "contradicted"}:
                if not {a, b} <= evidence_nodes:
                    raise ValueError(
                        f"Semantic edge {edge_id!r} lacks evidence for both endpoints"
                    )
                if rounds and not set(evidence_ids) <= displayed.get(
                    row.get("round_index"), set()
                ):
                    raise ValueError(
                        f"Semantic edge {edge_id!r} cites undisplayed round evidence"
                    )
            result[edge_id] = row
        return result

    def _edge_rejection(self, edge):
        if edge["state"] != "supported":
            return {"outcome": edge["state"]}
        confidence = edge.get("confidence")
        if confidence is not None and confidence < self.min_edge_confidence:
            return {
                "outcome": "low_confidence",
                "confidence": confidence,
                "min_edge_confidence": self.min_edge_confidence,
            }
        return None

    def _reference(self, indices, region_ids):
        metadata = self.memory.kv_metadata
        tokens = tuple(self.memory.tokens[index] for index in indices)
        return KVReference(
            self.memory_fingerprint,
            self.fingerprint,
            BANK_REF,
            str(self.memory_dir / BANK) if self.memory_dir else None,
            self.bank_sha256,
            tuple(region_ids),
            tuple(indices),
            tuple(token.source_index for token in tokens),
            tuple(token.mrope_position for token in tokens),
            metadata.key_representation_stage,
            metadata.value_representation_stage,
            deepcopy(self.layer_refs),
        )

    def to_dict(self):
        """Serialize graph-to-source bindings and provenance without copying any K/V tensors."""
        return {
            "schema_version": 1,
            "fingerprint": self.fingerprint,
            "graph_fingerprint": self.graph_fingerprint,
            "image_hash": self.memory.image_hash,
            "source_fingerprints": dict(self.source_identity),
            "source_compatibility": {
                "model_files_sha256": deepcopy(self.memory.fingerprints.get("model_weights_sha256", {})),
                "adapter": "none",
                "layout": "decoder_visual_rows_flattened_kv_heads",
                "preprocessing": asdict(self.memory.geometry),
                "qk_norm": deepcopy(self.memory.kv_metadata.qk_norm),
                "rope": deepcopy(self.memory.kv_metadata.rope),
            },
            "min_edge_confidence": self.min_edge_confidence,
            "source": asdict(self._reference(range(len(self.memory.tokens)), ())),
            "unassigned_token_indices": self.memory.unassigned_token_indices,
            "regions": [
                self._node_record(node) for _, node in sorted(self.nodes.items())
            ],
            "semantic_edges": [
                deepcopy(edge) for _, edge in sorted(self.edges.items())
            ],
            "evidence": [deepcopy(row) for _, row in sorted(self.evidence.items())],
        }

    @staticmethod
    def _node_record(node):
        return {
            "region_id": node.region_id,
            "physical_region_id": node.physical_region_id,
            "label": node.label,
            "mask_sha256": node.mask_sha256,
            "evidence_id": node.evidence_id,
            "source_visual_indices": node.visual_indices,
            "coverage_weights": node.coverage_weights,
            "aliases": node.aliases,
            "attributes": {
                key: sorted(values) for key, values in node.attributes.items()
            },
        }

    def read_kv(
        self, result: RetrievalResult | KVReference, layers: Iterable[int] | None = None
    ):
        """Read only selected CPU rows from the verified source bank."""
        import torch

        reference = (
            result.kv_reference if isinstance(result, RetrievalResult) else result
        )
        indices = reference.source_visual_indices
        if (
            tuple(sorted(set(indices))) != indices
            or any(
                not _integer(index) or not 0 <= index < len(self.memory.tokens)
                for index in indices
            )
            or reference != self._reference(indices, reference.physical_region_ids)
        ):
            raise ValueError(
                "KV reference differs from the indexed source coordinates or fingerprint"
            )
        regions = {region.region_id for region in self.memory.regions}
        if tuple(
            sorted(set(reference.physical_region_ids))
        ) != reference.physical_region_ids or (
            set(reference.physical_region_ids) - regions
        ):
            raise ValueError("KV reference contains invalid physical region IDs")
        requested = sorted(self.memory.source_kv if layers is None else set(layers))
        index = torch.tensor(indices, dtype=torch.long)
        kv = {
            layer: tuple(
                tensor.index_select(0, index) for tensor in self.memory.source_kv[layer]
            )
            for layer in requested
        }
        return KVSlice(
            kv,
            reference.physical_region_ids,
            indices,
            reference.source_sequence_indices,
            reference.mrope_positions,
            reference.key_representation_stage,
            reference.value_representation_stage,
        )

    def _concept_match(self, node_id, concept, question):
        node = self.nodes[node_id]
        matched = concept.terms.intersection(node.aliases)
        if not matched:
            return None
        node_attributes = node.attributes
        all_values = set().union(*node_attributes.values())
        for key, values in concept.attributes.items():
            if not (
                node_attributes.get(key, all_values if not key else set()) & values
            ):
                return None
        attribute_terms = sorted(
            term for term in all_values if _contains(question, term)
        )
        exact = _text(concept.name) == _text(node.label)
        score = 20 * len(attribute_terms) + 10 * exact + node.score
        return score, sorted(matched), attribute_terms

    def _edge_match(self, row, intent, question):
        if not intent.predicate.terms & _predicate_terms(row["predicate"]):
            return None
        score = 0.0
        for concept, endpoint in (
            (intent.subject, row["subject_id"]),
            (intent.object, row["object_id"]),
        ):
            if concept:
                match = self._concept_match(endpoint, concept, question)
                if match is None:
                    return None
                score += match[0]
        return score

    def _edge_nodes(self, row):
        return {
            row["subject_id"],
            row["object_id"],
            *(self.evidence[key]["region_id"] for key in row["evidence_ids"]),
        }

    def _endpoint_mentioned(self, concept, question):
        if any(_contains(question, term) for term in concept.terms):
            return True
        return any(
            any(_contains(question, alias) for alias in node.aliases)
            and self._concept_match(node_id, concept, question) is not None
            for node_id, node in self.nodes.items()
        )

    def _query(self, question, plan, operation):
        embedded = plan is None
        plan = self.graph.get("plan", {}) if embedded else plan
        if isinstance(operation, str):
            operation_id = operation
            operation = next(
                (
                    row
                    for row in plan.get("operations", ())
                    if row["id"] == operation_id
                ),
                None,
            )
            if operation is None:
                raise ValueError(f"Unknown operation {operation_id!r}")
        operation = operation or {}
        question = operation.get("question") or question
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Retrieval requires a question")
        normalized = _text(question)
        raw_concepts = operation.get("concepts", plan.get("concepts", ()))
        concepts = [_Concept.parse(item) for item in _items(raw_concepts)]
        different_question = embedded and _text(plan.get("question")) != normalized
        if not operation and (not concepts or different_question):
            hits = {
                term
                for node in self.nodes.values()
                for term in node.aliases
                if _contains(normalized, term)
            }
            concepts = [_Concept.parse(term, required=False) for term in sorted(hits)]
        # The operation may carry its own narrower intents. Otherwise only keep
        # plan intents whose endpoints are listed or mentioned in the question.
        raw_relations = operation.get(
            "relation_intents", plan.get("relation_intents", ())
        )
        relations = [_Relation.parse(item) for item in _items(raw_relations)]
        if operation and "relation_intents" not in operation:
            operation_terms = set().union(*(concept.terms for concept in concepts))
            relations = [
                relation
                for relation in relations
                if all(
                    endpoint is None
                    or bool(endpoint.terms & operation_terms)
                    or self._endpoint_mentioned(endpoint, normalized)
                    for endpoint in (relation.subject, relation.object)
                )
            ]
        elif not operation and different_question:
            relations = [
                relation
                for relation in relations
                if any(_contains(normalized, term) for term in relation.predicate.terms)
                and all(
                    endpoint is None
                    or any(_contains(normalized, term) for term in endpoint.terms)
                    for endpoint in (relation.subject, relation.object)
                )
            ]
        return question, normalized, operation.get("id"), concepts, relations

    def retrieve(
        self,
        question: str,
        plan: dict | None = None,
        *,
        operation: str | dict | None = None,
        token_budget: int | None = None,
        anchor_region_ids: Iterable[str] = (),
        background_tokens: int = 0,
        max_hops: int = 0,
    ) -> RetrievalResult:
        question, normalized, operation_id, concepts, relations = self._query(
            question, plan, operation
        )
        total = len(self.memory.tokens)
        budget = total if token_budget is None else token_budget
        if not _integer(budget) or not 1 <= budget <= total:
            raise ValueError(
                "token_budget must be an integer between 1 and the full visual row count"
            )
        for name, amount in (
            ("background_tokens", background_tokens),
            ("max_hops", max_hops),
        ):
            if not _integer(amount) or amount < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if isinstance(anchor_region_ids, str):
            raise ValueError("anchor_region_ids must be a sequence of region IDs")

        selection = _Selection(self, budget)
        for node_id in sorted(set(anchor_region_ids)):
            if node_id not in self.nodes:
                raise ValueError(f"Unknown anchor region {node_id!r}")
            selection.choose({node_id}, "anchor", node_id, required=True)

        selected_edges = {}
        for intent in relations:
            matching = []
            eligible = []
            for edge_id, edge in self.edges.items():
                score = self._edge_match(edge, intent, normalized)
                if score is not None:
                    matching.append(edge_id)
                    rejection = self._edge_rejection(edge)
                    if rejection:
                        selection.record(
                            "relation", intent.name, edge_id=edge_id, **rejection
                        )
                    else:
                        eligible.append((score, edge_id, edge))
            chosen = False
            for _, edge_id, edge in sorted(
                eligible, key=lambda match: (-match[0], match[1])
            ):
                if selection.choose(
                    self._edge_nodes(edge), "relation", intent.name, edge_id=edge_id
                ):
                    selected_edges[edge_id] = edge
                    chosen = True
                    break
            if not chosen:
                selection.missing.append(
                    {
                        "kind": "relation",
                        "query": intent.name,
                        "required": intent.required,
                        "reason": "no_supported_edge_fits"
                        if eligible
                        else "no_supported_edge",
                        "candidate_edge_ids": sorted(matching),
                    }
                )

        seen = set()
        for concept in concepts:
            key = (_text(concept.name), tuple(sorted(concept.attributes.items())))
            if key in seen:
                continue
            seen.add(key)
            matching = []
            for node_id in self.nodes:
                match = self._concept_match(node_id, concept, normalized)
                if match is not None:
                    score, names, attributes = match
                    matching.append(
                        (node_id in selection.nodes, score, node_id, names, attributes)
                    )
            for _, score, node_id, names, attributes in sorted(
                matching, key=lambda match: (not match[0], -match[1], match[2])
            ):
                if selection.choose(
                    {node_id},
                    "concept",
                    concept.name,
                    matched_names=names,
                    matched_attributes=attributes,
                    score=score,
                ):
                    break
            else:
                selection.missing.append(
                    {
                        "kind": "concept",
                        "query": concept.name,
                        "required": concept.required,
                        "reason": "no_matching_region_fits"
                        if matching
                        else "no_matching_region",
                        "candidate_node_ids": sorted(match[2] for match in matching),
                    }
                )

        for hop in range(1, max_hops + 1):
            frontier = set(selection.nodes)
            for edge_id, edge in sorted(self.edges.items()):
                if edge_id in selected_edges:
                    continue
                if frontier.isdisjoint((edge["subject_id"], edge["object_id"])):
                    continue
                rejection = self._edge_rejection(edge)
                if rejection:
                    selection.record(
                        "expansion",
                        edge["predicate"],
                        edge_id=edge_id,
                        hop=hop,
                        **rejection,
                    )
                    continue
                if selection.choose(
                    self._edge_nodes(edge),
                    "expansion",
                    edge["predicate"],
                    edge_id=edge_id,
                    hop=hop,
                ):
                    selected_edges[edge_id] = edge
            if frontier == selection.nodes:
                break

        background = self._background(
            selection.rows, min(background_tokens, budget - len(selection.rows))
        )
        if background:
            selection.record(
                "background",
                "nearby unassigned source tokens",
                "selected",
                visual_indices=background,
            )
        indices = tuple(sorted(selection.rows.union(background)))
        fallback = None
        if not indices or any(row["required"] for row in selection.missing):
            fallback = {
                "mode": "full_visual",
                "reason": "required_evidence_missing"
                if indices
                else "no_retrievable_visual_rows",
                "exceeds_token_budget": total > budget,
                "retrieved_visual_indices_before_fallback": indices,
            }
            indices = tuple(range(total))
        physical_ids = tuple(
            sorted(
                {self.nodes[node_id].physical_region_id for node_id in selection.nodes}
            )
        )
        node_records = tuple(
            self._node_record(self.nodes[node_id])
            for node_id in sorted(selection.nodes)
        )
        return RetrievalResult(
            question,
            operation_id,
            budget,
            total,
            node_records,
            tuple(deepcopy(selected_edges[key]) for key in sorted(selected_edges)),
            tuple(selection.reasons),
            tuple(selection.missing),
            background,
            fallback,
            self._reference(indices, physical_ids),
        )

    def _background(self, rows, limit):
        if not rows or not limit:
            return ()
        tokens = self.memory.tokens
        anchors = [(tokens[index].grid_y, tokens[index].grid_x) for index in rows]

        def distance(index):
            y, x = tokens[index].grid_y, tokens[index].grid_x
            return min(abs(y - ay) + abs(x - ax) for ay, ax in anchors), index

        candidates = set(self.memory.unassigned_token_indices) - rows
        return tuple(sorted(sorted(candidates, key=distance)[:limit]))


@dataclass
class _Selection:
    index: AgenticKVIndex
    budget: int
    rows: set[int] = field(default_factory=set)
    nodes: set[str] = field(default_factory=set)
    reasons: list[dict] = field(default_factory=list)
    missing: list[dict] = field(default_factory=list)

    def record(self, kind, query, outcome, **details):
        self.reasons.append(
            {"kind": kind, "query": query, "outcome": outcome, **details}
        )

    def choose(self, nodes, kind, query, required=False, **details):
        by_node = {
            node_id: self.index.nodes[node_id].visual_indices for node_id in nodes
        }
        rows = set().union(*by_node.values())
        added = rows - self.rows
        if not all(by_node.values()):
            outcome = "no_visual_rows"
        elif len(self.rows) + len(added) > self.budget:
            outcome = "token_budget"
        else:
            outcome = "selected"
            self.rows.update(added)
            self.nodes.update(nodes)
        self.record(
            kind,
            query,
            outcome,
            node_ids=sorted(nodes),
            marginal_tokens=len(added),
            **details,
        )
        if outcome != "selected" and required:
            self.missing.append(
                {"kind": kind, "query": query, "reason": outcome, "required": True}
            )
        return outcome == "selected"


@dataclass
class AgenticEpisode:
    """A fixed image/source binding with independent per-question retrieval decisions."""

    index: AgenticKVIndex
    history: list[RetrievalResult] = field(default_factory=list, init=False)

    @property
    def image_hash(self):
        return self.index.memory.image_hash

    def bind(self, index: AgenticKVIndex):
        """Keep history only when the exact image, graph and source remain bound."""
        if index.fingerprint != self.index.fingerprint:
            self.reset()
        self.index = index

    def retrieve(self, question, plan=None, **options):
        result = self.index.retrieve(question, plan, **options)
        self.history.append(result)
        return result

    def reset(self):
        self.history.clear()
