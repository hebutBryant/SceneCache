"""Import text concepts from Scene2Graph's saved first-SAM3 planning run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

from .schema import image_sha256


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _text_concepts(arguments, description: str) -> list[str]:
    if not isinstance(arguments, dict) or set(arguments) != {"text"}:
        raise ValueError(f"{description} must contain only a SAM3 text argument.")
    items = arguments["text"]
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list) or not items or any(
        not isinstance(item, str) or not item.strip() for item in items
    ):
        raise ValueError(f"{description} must have nonempty SAM3 text concepts.")
    return [item.strip() for item in items]


def _verify_input(directory: Path, image: Image.Image, stage: str, request_image: str) -> dict:
    matches = sorted((directory / "thinking").glob(f"call_*_online_{stage}"))
    if len(matches) != 1:
        raise ValueError(f"Expected a single archived Scene2Graph {stage} call.")
    call = matches[0]
    request = _read_object(call / "request.json")
    if request.get("stage") != f"online_{stage}" or request.get("images") != [request_image]:
        raise ValueError(f"Scene2Graph {stage} and SAM3 requests must refer to the same image.")
    with Image.open(call / "image_0.png") as file:
        archived = file.convert("RGB")
    # Scene2Graph's QwenAdapter saves PIL thumbnail() before processor transforms.
    thumbnail = image.copy()
    side = max(archived.size)
    thumbnail.thumbnail((side, side))
    archived_hash = image_sha256(archived)
    if image_sha256(thumbnail) != archived_hash:
        raise ValueError(f"The supplied image differs from the archived Scene2Graph {stage} input.")
    return {
        "request": str((call / "request.json").relative_to(directory)),
        "thumbnail": str((call / "image_0.png").relative_to(directory)),
        "thumbnail_pixel_sha256": archived_hash,
    }


def import_scene2graph_plan(
    directory: str | Path, *, image: str | Path | Image.Image, question: str
) -> dict:
    """Normalize one Scene2Graph first-tool export into an existing plan.

    The expected export has plan.json, decision.json, first_sam3/request.json,
    and the controller's archived planning/decision image and request. Only
    the text from the first actual request is reusable as SAM3 concepts.
    The original intents are retained verbatim as provenance.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Import requires the current question.")
    root = Path(directory).resolve()
    artifacts = ("plan.json", "decision.json", "first_sam3/request.json")
    plan, decision, request = (_read_object(root / name) for name in artifacts)
    intents = plan.get("intents")
    if not isinstance(intents, list) or any(not isinstance(intent, dict) for intent in intents):
        raise ValueError("Scene2Graph plan.json must contain an intents list.")
    calls = decision.get("tool_calls")
    if decision.get("action") != "call_tools" or not isinstance(calls, list) or not calls:
        raise ValueError("Scene2Graph decision must record a first SAM3 tool call.")
    first = calls[0]
    if not isinstance(first, dict) or first.get("tool_name") not in {"sam3", "sam"}:
        raise ValueError("Scene2Graph first decision tool must be SAM3.")
    if request.get("tool") not in {"sam3", "sam"} or not isinstance(request.get("image"), str):
        raise ValueError("Scene2Graph first_sam3/request.json must identify its SAM3 input image.")
    concepts = _text_concepts(request.get("params"), "Scene2Graph SAM3 request")
    if concepts != _text_concepts(first.get("arguments"), "Scene2Graph first decision"):
        raise ValueError("Scene2Graph decision and actual SAM3 request disagree on concepts.")
    if isinstance(image, Image.Image):
        normalized = ImageOps.exif_transpose(image).convert("RGB")
    else:
        with Image.open(image) as source:
            normalized = ImageOps.exif_transpose(source).convert("RGB")
    controller = {
        stage: _verify_input(root, normalized, stage, request["image"])
        for stage in ("planning", "decision")
    }
    return {
        "question": question.strip(),
        "concepts": list(dict.fromkeys(concepts)),
        "relation_intents": [],
        "source": {
            "type": "scene2graph_first_sam3",
            "directory": str(root),
            "artifacts_sha256": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in artifacts
            },
            "archived_input_path": request["image"],
            "image_pixel_sha256": image_sha256(normalized),
            "controller": controller,
            "intents": intents,
            "sam3_call_id": first.get("call_id"),
        },
    }
