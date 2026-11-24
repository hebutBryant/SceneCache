"""Mini-o3 observation provenance and a canonical, evidence-only terminal state."""

from __future__ import annotations

import json
import math
import re
import runpy
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from sglang.srt.mem_cache.scene_graph_memory.schema import (
    content_fingerprint,
    image_sha256,
)

VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
GROUNDING = re.compile(r"<grounding>(.*?)</grounding>", re.DOTALL)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
SOURCE = re.compile(r"original_image|observation_[1-9][0-9]*")
STATE_PROMPT_VERSION = "terminal_perceptual_fields_v1"
STATE_SYSTEM = (
    "Answer the next question using the supplied observed evidence. "
    "Provide a concise answer inside <answer> and </answer>. "
    "If the observation does not support an answer, emit <need_evidence> "
    "describe what additional evidence is required, and close with </need_evidence>."
)
REQUEST_SUFFIX = (
    "\nAnswer in <answer> and </answer>. If the available evidence is "
    "insufficient, request additional visual evidence inside <need_evidence> and </need_evidence>."
)
STOP_STRINGS = ("</grounding>", "</answer>", "</need_evidence>")


@dataclass
class Observation:
    image: Image.Image
    index: int
    parent: int | None
    parent_box: tuple[int, int, int, int]
    original_box: tuple[int, int, int, int]
    image_hash: str
    message_index: int

    @classmethod
    def original(cls, image):
        box = (0, 0, image.width, image.height)
        return cls(image, 0, None, box, box, image_sha256(image), 1)

    def metadata(self):
        return {
            "observation_id": self.index,
            "parent_observation_id": self.parent,
            "message_index": self.message_index,
            "image_hash": self.image_hash,
            "size_wh": list(self.image.size),
            "bbox_parent_xyxy": list(self.parent_box),
            "bbox_original_xyxy": list(self.original_box),
            "coordinate_system": "display_rgb_integer_pixels_xyxy_exclusive",
            "transform": {"operation": "crop" if self.parent is not None else "source", "resize": 1},
            "file": f"observation_{self.index:02d}.png",
        }


class CropTools:
    def __init__(self, code_dir):
        code_dir = Path(code_dir)
        self.prompts = runpy.run_path(str(code_dir / "verl/trainer/constants.py"))
        methods = runpy.run_path(
            str(code_dir / "verl/workers/rollout/vllm_rollout/function_tools.py")
        )
        self.prepare_crop = methods["prepare_grounding_inputs_multi_turn"]
        self.crop_image = methods["crop_image"]

    def initial_messages(self, question):
        return [
            {"role": "system", "content": self.prompts["TOOL_CROP_SYSTEM_PROMPT"]},
            {"role": "user", "content": VISION_MARKER + "\n" + question},
        ]

    def observe(self, observations, action, message_index):
        images = [item.image for item in observations]
        _, (source_image, box) = self.prepare_crop(
            [action], images, [im.size for im in images], use_relative_coordinates=True
        )
        parent = 0 if action["source"] == "original_image" else int(action["source"].split("_")[-1])
        offset_x, offset_y = observations[parent].original_box[:2]
        left, top, right, bottom = box
        image = self.crop_image(source_image, box, [], resize=1)
        if image.size != (right - left, bottom - top):
            raise ValueError("Mini-o3 crop unexpectedly rescaled its output")
        return Observation(
            image, len(observations), parent, tuple(box),
            (left + offset_x, top + offset_y, right + offset_x, bottom + offset_y),
            image_sha256(image), message_index,
        )

    def response(self, turn, observation):
        return {
            "role": "user",
            "content": self.prompts["TOOL_CALL_CROP_MULTI_TRUN_PROMPT"].format(
                action_turn=turn - 1, observation_turn=observation.index
            ),
        }

    def failure(self, exc):
        return {
            "role": "user",
            "content": (
                f"ERROR occurs during grounding. Error Information: {exc}.\n"
                + self.prompts["ERROR_INFO_MULTI_TURN_PROMPT"]
            ),
        }


def parse_action(raw):
    action = json.loads(raw)
    if not isinstance(action, dict):
        raise ValueError("grounding payload must be a JSON object")
    bbox = action.get("bbox_2d")
    source = action.get("source")
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(
        type(value) in (int, float) and math.isfinite(value) for value in bbox
    ):
        raise ValueError("bbox_2d must contain four finite numbers")
    if not isinstance(source, str) or not SOURCE.fullmatch(source):
        raise ValueError("source must be original_image or observation_i")
    return action


def canonical_state(original, terminal):
    # Identity and pixels are shared across replay paths. Acquisition provenance
    # stays in the trace rather than making z depend on how the crop was found.
    evidence = {
        "kind": "terminal_crop" if terminal.parent is not None else "original_image_no_crop",
        "image_hash": terminal.image_hash,
        "size_wh": list(terminal.image.size),
        "bbox_original_xyxy": list(terminal.original_box),
        "coordinate_system": "display_rgb_integer_pixels_xyxy_exclusive",
    }
    state = {
        "schema_version": 1,
        "source_image_hash": original.image_hash,
        "entity_id": "roi:" + content_fingerprint([original.image_hash, list(terminal.original_box)]),
        "evidence": [evidence],
        "evidence_hash": content_fingerprint(evidence),
        "perception": {"attributes": [], "ocr": [], "relations": [], "verification": "unverified"},
        "unresolved": ["The crop may omit context or evidence required by a new question."],
        "text_evidence_directly_answers_question": False,
    }
    return {"state": state, "state_hash": content_fingerprint(state)}


def canonical_messages(state):
    # Bind hashes through the bank manifest. Keep the model input focused on
    # the perceptual fields and coordinates that can inform an answer.
    z = state["state"]
    evidence = z["evidence"][0]
    visible_state = {
        "evidence": evidence["kind"],
        "bbox_original_xyxy": evidence["bbox_original_xyxy"],
        "crop_size_wh": evidence["size_wh"],
        "perception": z["perception"],
        "unresolved": z["unresolved"],
    }
    payload = json.dumps(visible_state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return [
        {"role": "system", "content": STATE_SYSTEM},
        {"role": "user", "content": payload + "\n" + VISION_MARKER},
    ]


def integer_action(box, size, source):
    # Aim at the interior of int()'s pixel bin, not a rounded floating boundary.
    width, height = size
    return {
        "bbox_2d": [(value + 0.25) / limit for value, limit in zip(box, (width, height) * 2)],
        "source": source,
    }


def alternate_terminal(tools, observations, terminal, first_message_index):
    """Find a distinct, valid intermediate field of view and prove equal crop pixels."""
    original = observations[0]
    width, height = original.image.size
    left, top, right, bottom = terminal.original_box
    parent_box = observations[terminal.parent].original_box
    for pad in (8, 28, 56, 4, 2, 1):
        candidate = (
            max(0, left - pad), max(0, top - pad),
            min(width - 1, right + pad), min(height - 1, bottom + pad),
        )
        if candidate in (terminal.original_box, parent_box, original.original_box):
            continue
        x0, y0, x1, y1 = candidate
        # The official crop clamps the right/bottom edge to size-1.
        if not (x0 <= left and y0 <= top and right < x1 and bottom < y1):
            continue
        parent_action = integer_action(candidate, original.image.size, "original_image")
        try:
            parent = tools.observe(observations, parent_action, first_message_index + 1)
            child_box = (left - x0, top - y0, right - x0, bottom - y0)
            child_action = integer_action(child_box, parent.image.size, f"observation_{parent.index}")
            child = tools.observe([*observations, parent], child_action, first_message_index + 3)
        except (ValueError, AssertionError):
            continue
        if (
            parent.original_box != candidate
            or child.original_box != terminal.original_box
            or child.image.mode != terminal.image.mode
            or child.image.size != terminal.image.size
            or child.image_hash != terminal.image_hash
            or child.image.tobytes() != terminal.image.tobytes()
        ):
            continue
        return [(parent_action, parent), (child_action, child)]
    raise ValueError("no distinct upstream-compatible integer crop parent with identical final pixels")
