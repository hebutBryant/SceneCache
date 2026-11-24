"""Prepare EXIF-normalized region masks using a local SAM3 worker or saved masks.

The Scene2Graph SAM3 wrapper uses a newer Transformers than VLCache. Invoke it
through its existing interpreter and exchange lossless pixels and tensor data.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps

if TYPE_CHECKING:
    from .schema import MaskInput


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_hash(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _load_prompts(path: Path) -> list[str]:
    value = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(value)
        if isinstance(value, dict):
            value = value.get("prompts")
    else:
        value = [line.strip() for line in value.splitlines() if line.strip()]
    if not isinstance(value, list) or not value or any(
        not isinstance(prompt, str) or not prompt.strip() for prompt in value
    ):
        raise ValueError("SAM3 prompts must be nonempty concept strings, in JSON or one per line.")
    return [prompt.strip() for prompt in value]


def _binary_mask(value: np.ndarray, image: Image.Image) -> np.ndarray:
    if value.shape != (image.height, image.width):
        raise ValueError(
            f"Mask shape {value.shape} must match the oriented source image "
            f"(H, W) = {(image.height, image.width)}."
        )
    if value.dtype != np.bool_ and (
        value.dtype.kind not in "ui" or not np.isin(value, (0, 1, 255)).all()
    ):
        raise ValueError("Expected a boolean or binary integer mask (0/1 or 0/255).")
    return np.ascontiguousarray(value, dtype=np.bool_)


def _mask_rows(path: Path, image: Image.Image) -> tuple[list, dict]:
    """Read a mask stack or a PNG manifest from either supported pipeline."""
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            masks = arrays["masks"]
        if masks.ndim != 3 or masks.shape[1:] != (image.height, image.width):
            raise ValueError("NPZ 'masks' must have shape [N, H, W] in oriented image coordinates.")
        return [(mask, {}) for mask in masks], {}
    if path.suffix.lower() != ".json":
        raise ValueError("--masks-file requires .npz with 'masks' or a JSON manifest of PNG masks.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(manifest, list):
        manifest = {"regions": manifest}
    if not isinstance(manifest, dict):
        raise ValueError("Expected an object or region list in mask manifest.")
    if "source_bank" in manifest:
        return _memory_mask_rows(path.parent, manifest, image)
    if "image_size_wh" in manifest and manifest["image_size_wh"] != list(image.size):
        raise ValueError("The imported mask manifest is for a different image size.")
    if "pixel_sha256" in manifest and manifest["pixel_sha256"] != _pixel_hash(image):
        raise ValueError("The imported mask manifest is for different image pixels.")
    body = manifest.get("data", manifest)
    rows = body.get("regions") if isinstance(body, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Mask manifest must contain a 'regions' list.")
    pairs = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("mask"), str):
            raise ValueError("Each imported region must specify its PNG mask path in 'mask'.")
        mask_path = (path.parent / row["mask"]).resolve()
        with Image.open(mask_path) as source:
            if source.format != "PNG":
                raise ValueError(f"Expected a PNG mask, got {mask_path}.")
            values = np.array(source)
        info = {key: value for key, value in row.items() if key != "mask"}
        info.update(mask_path=str(mask_path), mask_sha256=_file_hash(mask_path))
        pairs.append((values, info))
    return pairs, {key: manifest[key] for key in ("tool", "model", "versions") if key in manifest}


def _memory_mask_rows(root: Path, manifest: dict, image: Image.Image) -> tuple[list, dict]:
    from .schema import BANK_REF, SCHEMA_VERSION, image_sha256, mask_sha256

    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest["source_bank"].get("id") != BANK_REF
    ):
        raise ValueError("Unsupported scene graph memory mask manifest.")
    if manifest["geometry"]["original_size"] != list(image.size):
        raise ValueError("The imported mask manifest is for a different image size.")
    if manifest["image_hash"] != image_sha256(image):
        raise ValueError("The imported mask manifest is for different image pixels.")

    pairs = []
    for index, row in enumerate(manifest["regions"], 1):
        region_id = f"r{index:04d}"
        reference = f"masks/{region_id}.png"
        if row["region_id"] != region_id or row["mask_ref"] != reference:
            raise ValueError("Invalid scene graph memory mask reference.")
        mask_path = (root / reference).resolve()
        if not mask_path.is_relative_to(root):
            raise ValueError(f"Scene graph memory mask resolves outside its directory: {reference}")
        file_sha256 = _file_hash(mask_path)
        if manifest["files"].get(reference) != {
            "bytes": mask_path.stat().st_size, "sha256": file_sha256,
        }:
            raise ValueError(f"Scene graph memory mask payload hash or size mismatch: {reference}")
        with Image.open(mask_path) as source:
            if source.format != "PNG" or source.mode != "1":
                raise ValueError(f"Expected a 1-bit PNG, got {mask_path}.")
            mask = _binary_mask(np.array(source), image)
        if mask_sha256(mask) != row["mask_sha256"]:
            raise ValueError(f"Scene graph memory mask pixel hash mismatch: {reference}")
        info = {
            **row.get("metadata", {}),
            "label": row.get("label"),
            "label_source": row.get("label_source"),
            "score": row.get("score"),
            "source_region_id": region_id,
            "mask_path": str(mask_path),
            "mask_file_sha256": file_sha256,
        }
        pairs.append((mask, info))
    return pairs, {
        "format": "scene_graph_memory_manifest",
        "fingerprint": manifest["fingerprint"],
        "segmentation": manifest["segmentation"],
    }


def _worker(job_path: Path) -> None:
    """Run the host's text/box detector in the SAM3 environment."""
    job = json.loads(job_path.read_text(encoding="utf-8"))
    sys.path.insert(0, job["source"])
    import torch
    import transformers
    from visual_tools.sam import SAMTool

    with Image.open(job_path.parent / "image.png") as source:
        image = source.convert("RGB")
    records = []
    masks = []
    filtered = []
    counts = []
    with SAMTool(model=job["checkpoint"], device=job["device"]) as tool:
        for query, prompt in enumerate(job["prompts"], 1):
            candidates = tool._detect(
                image, prompt, [], [], job["score_threshold"], job["mask_threshold"]
            )
            kept = 0
            for rank, (values, score) in enumerate(candidates, 1):
                mask = _binary_mask(values, image)
                if not math.isfinite(score):
                    raise RuntimeError("SAM3 returned a non-finite instance confidence.")
                metadata = {
                    "candidate_id": f"Q{query:03d}_C{rank:03d}",
                    "prompt_id": f"Q{query:03d}",
                    "score": score,
                    "score_type": "SAM3 instance confidence",
                    "label": prompt,
                    "label_source": "sam3_text_prompt",
                }
                area = int(np.count_nonzero(mask))
                if area == 0:
                    reason = "empty_mask"
                elif area < job["min_area"]:
                    reason = "below_min_area"
                elif job["max_candidates"] is not None and kept >= job["max_candidates"]:
                    reason = "over_prompt_candidate_limit"
                else:
                    masks.append(mask)
                    records.append(metadata)
                    kept += 1
                    continue
                filtered.append({**metadata, "area": area, "reason": reason})
            counts.append({
                "prompt": prompt,
                "candidates_after_detector_score_threshold": len(candidates),
                "retained": kept,
            })
        model_dtype = str(next(tool._model.parameters()).dtype)

    checkpoint = Path(job["checkpoint"])
    checkpoint_files = {
        path.name: _file_hash(path)
        for path in sorted(checkpoint.iterdir())
        if path.is_file() and path.suffix.lower() in {".json", ".txt", ".safetensors"}
    }
    wrapper = Path(job["source"]) / "visual_tools" / "sam.py"
    manifest = {
        "backend": "sam3",
        "input_mode": "explicit_text_prompts",
        "worker": {
            "implementation": "Scene2Graph visual_tools.sam.SAMTool._detect",
            "wrapper": str(wrapper),
            "wrapper_sha256": _file_hash(wrapper),
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": job["device"],
            "dtype": model_dtype,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_files_sha256": checkpoint_files,
        "score_threshold": job["score_threshold"],
        "mask_threshold": job["mask_threshold"],
        "min_area": job["min_area"],
        "max_candidates_per_prompt": job["max_candidates"],
        "image_size_wh": list(image.size),
        "pixel_sha256": _pixel_hash(image),
        "prompts": job["prompts"],
        "per_prompt": counts,
        "candidate_count_after_detector_score_threshold": len(records) + len(filtered),
        "retained_count": len(records),
        "filtered_count": len(filtered),
        "filtered": filtered,
        "records": records,
    }
    stack = np.stack(masks) if masks else np.empty((0, image.height, image.width), dtype=np.bool_)
    np.savez_compressed(job_path.parent / "masks.npz", masks=stack)
    (job_path.parent / "result.json").write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False))


def create_masks(
    image: Image.Image,
    *,
    masks_file: str | Path | None = None,
    sam3_checkpoint: str | Path | None = None,
    sam_prompts_file: str | Path | None = None,
    sam3_python: str | Path | None = None,
    sam3_source: str | Path | None = None,
    sam3_device: str = "cuda:0",
    sam3_score_threshold: float = 0.5,
    sam3_mask_threshold: float = 0.5,
    sam3_max_candidates: int | None = None,
    sam3_min_area: int = 1,
    sam3_timeout: float | None = None,
) -> tuple[list[MaskInput], dict]:
    """Return full-image boolean masks in EXIF-normalized coordinates and provenance."""
    from .schema import MaskInput

    normalized = ImageOps.exif_transpose(image).convert("RGB")
    if (masks_file is None) == (sam_prompts_file is None):
        raise ValueError("Specify exactly one of --masks-file and --sam-prompts-file.")
    if type(sam3_min_area) is not int or sam3_min_area < 1:
        raise ValueError("Minimum mask area must be a positive number of pixels.")
    if masks_file is not None:
        path = Path(masks_file).resolve()
        pairs, source = _mask_rows(path, normalized)
        kept = []
        filtered = []
        for index, (values, metadata) in enumerate(pairs):
            mask = _binary_mask(values, normalized)
            area = int(np.count_nonzero(mask))
            if area < sam3_min_area:
                filtered.append({
                    "index": index, "area": area,
                    "reason": "empty_mask" if area == 0 else "below_min_area",
                })
                continue
            kept.append((mask, {**metadata, "import_index": index}))
        provenance = {
            "backend": "imported_masks",
            "source": source,
            "path": str(path),
            "file_sha256": _file_hash(path),
            "image_size_wh": list(normalized.size),
            "pixel_sha256": _pixel_hash(normalized),
            "min_area": sam3_min_area,
            "candidate_count": len(pairs),
            "retained_count": len(kept),
            "filtered_count": len(filtered),
            "filtered": filtered,
        }
    else:
        if not all((sam3_checkpoint, sam3_source, sam3_python)):
            raise ValueError("SAM3 needs --sam3-checkpoint, --sam3-source, and --sam3-python.")
        for name, threshold in (("score", sam3_score_threshold), ("mask", sam3_mask_threshold)):
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                raise ValueError(f"SAM3 {name} threshold must be in [0, 1].")
        if sam3_max_candidates is not None and (
            type(sam3_max_candidates) is not int or sam3_max_candidates < 1
        ):
            raise ValueError("SAM3 per-prompt candidate limit must be a positive integer.")
        checkpoint = Path(sam3_checkpoint).resolve()
        source = Path(sam3_source).resolve()
        python = Path(sam3_python).absolute()
        for required in (
            checkpoint / "model.safetensors", checkpoint / "config.json",
            source / "visual_tools" / "sam.py", python,
        ):
            if not required.is_file():
                raise FileNotFoundError(f"Missing local SAM3 resource: {required}")
        prompts_file = Path(sam_prompts_file).resolve()
        job = {
            "checkpoint": str(checkpoint), "source": str(source), "device": sam3_device,
            "prompts": _load_prompts(prompts_file), "score_threshold": sam3_score_threshold,
            "mask_threshold": sam3_mask_threshold, "max_candidates": sam3_max_candidates,
            "min_area": sam3_min_area,
        }
        with tempfile.TemporaryDirectory(prefix="vlcache-sam3-") as temporary:
            directory = Path(temporary)
            normalized.save(directory / "image.png")
            job_path = directory / "job.json"
            job_path.write_text(json.dumps(job, ensure_ascii=False))
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            try:
                completed = subprocess.run(
                    [str(python), str(Path(__file__).resolve()), str(job_path)],
                    env=env, capture_output=True, text=True, timeout=sam3_timeout,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"SAM3 did not finish within {sam3_timeout:g} seconds") from error
            if completed.returncode:
                raise RuntimeError(f"SAM3 worker exited ({completed.returncode}):\n{completed.stderr.strip()}")
            provenance = json.loads((directory / "result.json").read_text())
            if provenance["pixel_sha256"] != _pixel_hash(normalized):
                raise RuntimeError("SAM3 received different source pixels.")
            rows, _ = _mask_rows(directory / "masks.npz", normalized)
            kept = [(mask, metadata) for (mask, _), metadata in zip(
                rows, provenance.pop("records"), strict=True
            )]

    regions = []
    for mask, metadata in kept:
        values = dict(metadata)
        score = values.pop("score", None)
        label = values.pop("label", None)
        label_source = values.pop("label_source", None)
        if score is not None and (not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise ValueError("Region score must be finite.")
        for name, value in (("label", label), ("label_source", label_source)):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"Region {name} must be a string.")
        regions.append(MaskInput(
            mask=mask, score=float(score) if score is not None else None,
            label=label, label_source=label_source or ("imported_metadata" if label else None),
            metadata=values,
        ))
    return regions, provenance


if __name__ == "__main__":
    _worker(Path(sys.argv[1]))
