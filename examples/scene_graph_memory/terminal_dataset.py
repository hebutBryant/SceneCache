"""Freeze local terminal questions, labels and image identities before inference."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from PIL import Image, ImageOps

from sglang.srt.mem_cache.scene_graph_memory.schema import (
    content_fingerprint,
    image_sha256,
    json_bytes,
)

SCHEMA_VERSION = 1
GQA_LAYOUT = "gqa_val_balanced_by_image"
GQA_SPLIT = "val_balanced_questions.json"
GQA_QUESTIONS_ARCHIVE = "questions1.2.zip"
GQA_IMAGES_ARCHIVE = "images.zip"
VISUALPROBE_SPLITS = {
    "easy": "VisualProbe_Easy/val.json",
    "medium": "VisualProbe_Medium/val.json",
    "hard": "VisualProbe_Hard/val.json",
}
VISUALPROBE_FIELDS = {
    "id": "doc_id",
    "question": "problem",
    "answer": "solution",
    "image": "images",
}
REQUIRED_FIELDS = frozenset(("id", "difficulty", "image", "question", "answer"))
OPTIONAL_FIELDS = frozenset(("followup_question", "followup_answer"))
FOLLOWUP_RUBRIC = [
    "Identifies the visual region or object relevant to the original question.",
    "Names one concrete cue a reviewer can inspect directly in the original image.",
    "Locates the cue using an observable position or relation in the image.",
]


def _inside(root: Path, location: str | Path, *, kind: str) -> Path:
    path = (root / Path(location)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{kind} escapes the dataset root: {location}")
    return path


def _relative(root: Path, location: Path) -> str:
    return location.relative_to(root).as_posix()


def _file_sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _field(row, key):
    if isinstance(row, dict) and key in row:
        return row[key]
    for part in key.split("."):
        if not isinstance(row, dict) or part not in row:
            raise ValueError(f"Missing dataset field {key!r}")
        row = row[part]
    return row


def _parquet(path: Path):
    try:
        from pyarrow.parquet import ParquetFile
    except ImportError as exc:
        raise RuntimeError("Reading a local Parquet dataset requires pyarrow") from exc
    return ParquetFile(path)


def _annotation_rows(path: Path, fields: dict, rows_key: str | None):
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        if rows_key is not None:
            raise ValueError(
                "A nested rows key is only applicable to a JSON annotation file"
            )
        # Collect identities and labels first. Read embedded image bytes only for the frozen selection.
        columns = sorted(
            {
                source.split(".", 1)[0]
                for key, source in fields.items()
                if key != "image"
            }
        )
        with _parquet(path) as source:
            missing = sorted(set(columns) - set(source.schema_arrow.names))
            if missing:
                raise ValueError(
                    f"Missing Parquet columns in {path}: {', '.join(missing)}"
                )
            for batch in source.iter_batches(batch_size=256, columns=columns):
                yield from batch.to_pylist()
    elif suffix == ".jsonl":
        if rows_key is not None:
            raise ValueError(
                "A nested rows key is only applicable to a JSON annotation file"
            )
        with path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSONL in {path} line {line_no}"
                        ) from exc
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if rows_key is not None:
            value = _field(value, rows_key)
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON list of annotation rows in {path}")
        yield from value
    else:
        raise ValueError(
            f"Supported local annotation files are JSON, JSONL and Parquet: {path}"
        )


def _sources(root, rows_file, fields, visualprobe_splits=None):
    if rows_file is None:
        if fields:
            raise ValueError("Custom fields require an explicit annotation file")
        selected = tuple(VISUALPROBE_SPLITS) if visualprobe_splits is None else tuple(visualprobe_splits)
        if not selected or len(set(selected)) != len(selected) or set(selected).difference(VISUALPROBE_SPLITS):
            raise ValueError("Expected a unique selection of VisualProbe validation splits")
        files = [
            (difficulty, _inside(root, VISUALPROBE_SPLITS[difficulty], kind="Annotation"))
            for difficulty in selected
        ]
        missing = [str(path) for _, path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing VisualProbe validation files: " + ", ".join(missing)
            )
        return files, VISUALPROBE_FIELDS.copy(), "mini_o3_visualprobe_validation"
    if visualprobe_splits is not None:
        raise ValueError("VisualProbe splits and explicit annotations are mutually exclusive")
    names = {key: key for key in REQUIRED_FIELDS}
    for key, value in (fields or {}).items():
        if (
            key not in REQUIRED_FIELDS | OPTIONAL_FIELDS
            or not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(f"Invalid dataset field mapping: {key!r}")
        names[key] = value
    if "followup_answer" in names and "followup_question" not in names:
        raise ValueError(
            "A follow-up answer mapping requires a follow-up question mapping"
        )
    path = _inside(root, rows_file, kind="Annotation")
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset annotation file: {path}")
    return [(None, path)], names, "explicit_annotations"


def _identity(value, field):
    if type(value) not in (int, str) or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{field} must be an integer or nonempty string")
    return value


def _question(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    normalized = re.sub(r"^\s*<image>(?:\r?\n)?", "", value, count=1)
    if "<image>" in normalized or not normalized.strip():
        raise ValueError(f"{field} must describe a single image")
    return normalized


def _label(value, field):
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{field} must contain a reference answer")
    json_bytes(value)
    return value


def _followup(row, fields, question):
    if "followup_question" in fields:
        followup = _question(
            _field(row, fields["followup_question"]), "followup_question"
        )
        answer = (
            _field(row, fields["followup_answer"])
            if "followup_answer" in fields
            else None
        )
        if answer is not None:
            return {
                "question": followup,
                "answer": _label(answer, "followup_answer"),
                "scoring": "reference",
                "source": "dataset",
            }
        source = "dataset"
    else:
        followup = (
            "Which part of this image would you inspect to verify the original question? "
            "Describe one concrete visible cue and locate it in the image.\n\n"
            f"Original question:\n{question}"
        )
        source = "locked_visual_evidence_prompt_v1"
    return {
        "question": followup,
        "answer": None,
        "scoring": "rubric_only",
        "rubric": FOLLOWUP_RUBRIC.copy(),
        "source": source,
    }


def _candidate(row, fields, relative, index, split, *, parquet):
    if not isinstance(row, dict):
        raise ValueError("Expected an annotation object")
    case_id = _identity(_field(row, fields["id"]), "id")
    if split is None:
        difficulty = _identity(_field(row, fields["difficulty"]), "difficulty")
        source = fields["difficulty"]
    else:
        difficulty, source = split, "official_validation_split"
        if row.get("data_source") not in (None, f"visual_probe_{split}"):
            raise ValueError(
                f"VisualProbe data_source does not match the official {split} split"
            )
    raw_question = _field(row, fields["question"])
    question = _question(raw_question, "question")
    answer = _label(_field(row, fields["answer"]), "answer")
    sample_key = content_fingerprint([difficulty, case_id])[:20]
    sample = {
        "id": case_id,
        "sample_key": sample_key,
        "difficulty": difficulty,
        "difficulty_source": source,
        "question": question,
        "source_question": raw_question,
        "answer": answer,
        "source_row": {"file": relative, "index": index},
        "followup": _followup(row, fields, question),
    }
    if not parquet:
        sample["image_value"] = _field(row, fields["image"])
    return sample


def _parquet_images(path: Path, indexes, field: str):
    """Yield only requested image entries, scanning each relevant row group once."""
    pending = set(indexes)
    column = field.split(".", 1)[0]
    start = 0
    with _parquet(path) as source:
        if column not in source.schema_arrow.names:
            raise ValueError(f"Missing Parquet image column {field!r} in {path}")
        for group in range(source.num_row_groups):
            end = start + source.metadata.row_group(group).num_rows
            requested = {i for i in pending if start <= i < end}
            if requested:
                cursor = start
                for batch in source.iter_batches(
                    batch_size=8, row_groups=[group], columns=[column]
                ):
                    batch_end = cursor + batch.num_rows
                    for index in sorted(
                        i for i in requested if cursor <= i < batch_end
                    ):
                        yield index, _field(
                            batch.slice(index - cursor, 1).to_pylist()[0], field
                        )
                        requested.remove(index)
                        pending.remove(index)
                    if not requested:
                        break
                    cursor = batch_end
            if not pending:
                return
            start = end
    if pending:
        raise ValueError(f"Missing Parquet source rows in {path}: {sorted(pending)}")


def _image_bytes(root: Path, value):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Terminal samples must contain exactly one image")
        value = value[0]
    if isinstance(value, dict):
        payload, path = value.get("bytes"), value.get("path")
        if payload is not None:
            if not isinstance(payload, bytes) or not payload:
                raise ValueError("Embedded dataset images must contain bytes")
            return payload, path, "parquet_embedded"
        value = path
    if isinstance(value, bytes) and value:
        return value, None, "parquet_embedded"
    if not isinstance(value, str) or not value.strip():
        raise ValueError("An image must have a local path or embedded Parquet bytes")
    path = _inside(root, value, kind="Image")
    return path.read_bytes(), _relative(root, path), "dataset_file"


def _rgb(payload):
    with Image.open(io.BytesIO(payload)) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def _freeze_image(sample, root, value):
    data, name, storage = _image_bytes(root, value)
    image = _rgb(data)
    sample.update(
        image=name,
        image_storage=storage,
        image_file_sha256=hashlib.sha256(data).hexdigest(),
        image_sha256=image_sha256(image),
        image_size_wh=list(image.size),
    )


def _publish(path: Path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as out:
        temporary = Path(out.name)
        try:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = json.loads(path.read_bytes())
                if existing != manifest:
                    if existing.get("experiment") != manifest.get("experiment"):
                        raise ValueError(
                            "Requested experiment metadata differs from the frozen manifest"
                        ) from None
                    raise FileExistsError(
                        f"Refusing to overwrite a different manifest lock at {path}"
                    ) from None
        finally:
            temporary.unlink(missing_ok=True)


def lock_manifest(
    dataset_root,
    rows_file=None,
    output=None,
    *,
    seed=42,
    per_difficulty=5,
    fields=None,
    rows_key=None,
    experiment=None,
    visualprobe_splits=None,
) -> dict:
    """Freeze the sample set and follow-up rubric from local annotations only."""
    if output is None:
        raise ValueError("An output path is required for the manifest lock")
    if type(seed) is not int or type(per_difficulty) is not int or per_difficulty < 1:
        raise ValueError(
            "Seed must be an integer and sample count must be a positive integer"
        )
    if experiment is not None and not isinstance(experiment, dict):
        raise ValueError("Experiment metadata must be a JSON object")
    root = Path(dataset_root).resolve(strict=True)
    output = Path(output).absolute()
    sources, fields, layout = _sources(root, rows_file, fields, visualprobe_splits)
    source_files, groups, seen = [], defaultdict(list), set()
    for split, path in sources:
        relative = _relative(root, path)
        source_files.append(
            {"path": relative, "sha256": _file_sha(path), "split_difficulty": split}
        )
        parquet = path.suffix.lower() == ".parquet"
        count = 0
        for index, row in enumerate(_annotation_rows(path, fields, rows_key)):
            try:
                sample = _candidate(
                    row, fields, relative, index, split, parquet=parquet
                )
                identity = (sample["difficulty"], sample["id"])
                if identity in seen:
                    raise ValueError(
                        f"Duplicate sample ID within difficulty: {identity!r}"
                    )
            except ValueError as exc:
                raise ValueError(f"{relative} row {index}: {exc}") from exc
            groups[identity[0]].append(sample)
            seen.add(identity)
            count += 1
        if not count:
            raise ValueError(f"No annotation rows in {relative}")
        source_files[-1]["rows"] = count
    rng = random.Random(seed)
    difficulties = sorted(groups, key=json_bytes)
    samples = [
        sample
        for tier in difficulties
        for sample in rng.sample(
            sorted(groups[tier], key=lambda row: json_bytes(row["id"])),
            min(len(groups[tier]), per_difficulty),
        )
    ]
    parquet_samples = defaultdict(dict)
    for sample in samples:
        if "image_value" in sample:
            _freeze_image(sample, root, sample.pop("image_value"))
        else:
            source = sample["source_row"]
            parquet_samples[source["file"]][source["index"]] = sample
    for file, lookup in parquet_samples.items():
        for index, value in _parquet_images(
            _inside(root, file, kind="Annotation"), lookup, fields["image"]
        ):
            _freeze_image(lookup[index], root, value)
    for source in source_files:
        if (
            _file_sha(_inside(root, source["path"], kind="Annotation"))
            != source["sha256"]
        ):
            raise RuntimeError(
                f"Annotations changed while building the lock: {source['path']}"
            )
    dataset = {
        "root": str(root),
        "layout": layout,
        "rows_files": source_files,
        "fields": fields,
        "rows_key": rows_key,
        "difficulty_counts": [
            {
                "difficulty": key,
                "count": len(groups[key]),
                "selected": min(len(groups[key]), per_difficulty),
                "shortfall": max(0, per_difficulty - len(groups[key])),
            }
            for key in difficulties
        ],
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "sampling": {
            "seed": seed,
            "per_difficulty": per_difficulty,
            "algorithm": "python_random_sample_sorted_exact_ids_v1",
        },
        "samples": samples,
    }
    if experiment is not None:
        manifest["experiment"] = json.loads(json_bytes(experiment))
    manifest["fingerprint"] = content_fingerprint(manifest)
    _publish(output, manifest)
    return manifest


def _gqa_image_member(image_id):
    if not isinstance(image_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", image_id):
        raise ValueError(f"Invalid GQA imageId {image_id!r}")
    return f"images/{image_id}.jpg"


def _gqa_question_payload(root):
    path = _inside(root, GQA_QUESTIONS_ARCHIVE, kind="GQA questions archive")
    try:
        with ZipFile(path) as archive:
            return archive.read(GQA_SPLIT)
    except (BadZipFile, KeyError) as exc:
        raise ValueError(f"Cannot read {GQA_SPLIT} from {path}") from exc


def _gqa_image_archive(root):
    path = _inside(root, GQA_IMAGES_ARCHIVE, kind="GQA images archive")
    try:
        return ZipFile(path)
    except BadZipFile as exc:
        raise ValueError(f"Cannot read GQA image archive {path}") from exc


def lock_gqa_manifest(
    dataset_root,
    output,
    *,
    seed=42,
    image_count=10,
    questions_per_image=3,
    experiment=None,
) -> dict:
    """Select image episodes from the local GQA balanced validation ZIPs."""
    if type(seed) is not int:
        raise ValueError("GQA sampling requires an integer seed")
    if type(image_count) is not int or image_count < 1:
        raise ValueError("GQA sampling requires a positive image count")
    if type(questions_per_image) is not int or questions_per_image < 2:
        raise ValueError("GQA episodes need an anchor and at least one held-out question")
    if experiment is not None and not isinstance(experiment, dict):
        raise ValueError("Experiment metadata must be a JSON object")

    root = Path(dataset_root).resolve(strict=True)
    payload = _gqa_question_payload(root)
    rows = json.loads(payload)
    split_sha256 = hashlib.sha256(payload).hexdigest()
    del payload
    if not isinstance(rows, dict) or not rows:
        raise ValueError("GQA balanced validation annotations must map IDs to questions")

    # Group by original image before sampling. Identical question strings in
    # annotation variants count once, with the lowest question ID as source.
    by_image = defaultdict(dict)
    for question_id, row in rows.items():
        if not isinstance(row, dict):
            raise ValueError(f"Invalid GQA question {question_id!r}")
        image_id = _identity(_field(row, "imageId"), "imageId")
        question = _question(_field(row, "question"), "question")
        answer = _label(_field(row, "answer"), "answer")
        if not isinstance(answer, str):
            raise ValueError(f"GQA question {question_id!r} needs a text reference")
        question_key = " ".join(question.casefold().split())
        old_question_id = by_image[image_id].get(question_key)
        if old_question_id is None or question_id < old_question_id:
            by_image[image_id][question_key] = question_id

    rng = random.Random(seed)
    samples = []
    with _gqa_image_archive(root) as images:
        members = set(images.namelist())
        eligible = sorted(
            image_id
            for image_id, questions in by_image.items()
            if len(questions) >= questions_per_image
            and _gqa_image_member(image_id) in members
        )
        if len(eligible) < image_count:
            raise ValueError(
                f"Expected {image_count} GQA images with {questions_per_image} "
                f"distinct questions and local pixels, found {len(eligible)}"
            )
        for image_id in rng.sample(eligible, image_count):
            requests = []
            question_ids = rng.sample(
                sorted(by_image[image_id].values()), questions_per_image
            )
            for index, question_id in enumerate(question_ids):
                row = rows[question_id]
                requests.append({
                    "name": "original" if index == 0 else f"followup_{index}",
                    "question_id": question_id,
                    "question": _question(row["question"], "question"),
                    "answer": row["answer"],
                    "types": row.get("types"),
                    "annotations": row.get("annotations"),
                    "semantic": row.get("semantic"),
                })
            member = _gqa_image_member(image_id)
            data = images.read(member)
            image = _rgb(data)
            anchor = requests[0]
            followup = requests[1]
            samples.append({
                "id": image_id,
                "image_id": image_id,
                "sample_key": content_fingerprint([GQA_LAYOUT, image_id])[:20],
                "difficulty": "gqa_val_balanced",
                "difficulty_source": "split_only_not_difficulty",
                "image": member,
                "image_storage": "gqa_zip_member",
                "image_file_sha256": hashlib.sha256(data).hexdigest(),
                "image_sha256": image_sha256(image),
                "image_size_wh": list(image.size),
                "question": anchor["question"],
                "source_question": rows[anchor["question_id"]]["question"],
                "answer": anchor["answer"],
                "source_row": {
                    "file": GQA_QUESTIONS_ARCHIVE,
                    "member": GQA_SPLIT,
                    "question_id": anchor["question_id"],
                },
                "followup": {
                    "question": followup["question"],
                    "answer": followup["answer"],
                    "scoring": "reference",
                    "source": "dataset",
                },
                "requests": requests,
            })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "root": str(root),
            "layout": GQA_LAYOUT,
            "split": GQA_SPLIT,
            "question_archive": GQA_QUESTIONS_ARCHIVE,
            "question_member_sha256": split_sha256,
            "image_archive": GQA_IMAGES_ARCHIVE,
            "annotation_count": len(rows),
            "eligible_image_count": len(eligible),
        },
        "sampling": {
            "seed": seed,
            "image_count": image_count,
            "questions_per_image": questions_per_image,
            "algorithm": "python_random_sorted_image_then_distinct_text_v1",
        },
        "samples": samples,
    }
    if experiment is not None:
        manifest["experiment"] = json.loads(json_bytes(experiment))
    manifest["fingerprint"] = content_fingerprint(manifest)
    _publish(Path(output).absolute(), manifest)
    return manifest


def _verify_image(data, sample):
    if hashlib.sha256(data).hexdigest() != sample["image_file_sha256"]:
        raise ValueError(f"Locked image bytes changed for sample {sample['id']!r}")
    image = _rgb(data)
    if (
        image_sha256(image) != sample["image_sha256"]
        or list(image.size) != sample["image_size_wh"]
    ):
        raise ValueError(
            f"Locked oriented image pixels changed for sample {sample['id']!r}"
        )
    return image


def _gqa_load_image(archive, sample):
    member = _gqa_image_member(sample["image_id"])
    if sample["image"] != member or sample["image_storage"] != "gqa_zip_member":
        raise ValueError(f"GQA sample {sample['id']!r} has an invalid image reference")
    try:
        return _verify_image(archive.read(member), sample)
    except (BadZipFile, KeyError) as exc:
        raise ValueError(f"Cannot read locked GQA image member {member}") from exc


def _verify_gqa_manifest(manifest, *, verify_images, verify_annotations):
    samples = manifest["samples"]
    sampling = manifest["sampling"]
    image_ids = {sample["image_id"] for sample in samples}
    sample_keys = {sample["sample_key"] for sample in samples}
    if (
        not samples
        or len(samples) != sampling["image_count"]
        or len(image_ids) != len(samples)
        or len(sample_keys) != len(samples)
    ):
        raise ValueError("GQA manifest must contain the locked number of unique image episodes")
    for sample in samples:
        requests = sample["requests"]
        names = ["original", *(f"followup_{index}" for index in range(1, len(requests)))]
        qids = {request["question_id"] for request in requests}
        if (
            len(requests) != sampling["questions_per_image"]
            or len(qids) != len(requests)
            or [request["name"] for request in requests] != names
            or len({" ".join(request["question"].casefold().split()) for request in requests}) != len(requests)
            or sample["id"] != sample["image_id"]
            or sample["question"] != requests[0]["question"]
            or sample["answer"] != requests[0]["answer"]
        ):
            raise ValueError(f"Inconsistent GQA questions for image {sample['id']!r}")
    root = Path(manifest["dataset"]["root"]).resolve(strict=True)
    if verify_annotations:
        current = hashlib.sha256(_gqa_question_payload(root)).hexdigest()
        if current != manifest["dataset"]["question_member_sha256"]:
            raise ValueError("Locked GQA validation annotations changed")
    if verify_images:
        with _gqa_image_archive(root) as archive:
            for sample in samples:
                _gqa_load_image(archive, sample)


def load_image(manifest: dict, sample: dict) -> Image.Image:
    """Read one locked original image and check both file bytes and oriented pixels."""
    dataset = manifest["dataset"]
    root = Path(dataset["root"]).resolve(strict=True)
    if dataset["layout"] == GQA_LAYOUT:
        with _gqa_image_archive(root) as archive:
            return _gqa_load_image(archive, sample)
    if sample["image_storage"] == "parquet_embedded":
        source = sample["source_row"]
        file = _inside(root, source["file"], kind="Annotation")
        ((_, value),) = _parquet_images(
            file, [source["index"]], dataset["fields"]["image"]
        )
    elif sample["image_storage"] == "dataset_file":
        value = sample["image"]
    else:
        raise ValueError(f"Unknown locked image storage {sample['image_storage']!r}")
    data, _, storage = _image_bytes(root, value)
    if storage != sample["image_storage"]:
        raise ValueError(f"Locked image storage changed for sample {sample['id']!r}")
    return _verify_image(data, sample)


def load_locked_manifest(
    path, *, verify_images=True, verify_annotations=True, experiment=None
) -> dict:
    manifest = json.loads(Path(path).read_bytes())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("Unsupported terminal manifest schema")
    if manifest.get("fingerprint") != content_fingerprint(
        {key: value for key, value in manifest.items() if key != "fingerprint"}
    ):
        raise ValueError("Terminal manifest fingerprint mismatch")
    if experiment is not None and (
        not isinstance(experiment, dict)
        or json_bytes(experiment) != json_bytes(manifest.get("experiment"))
    ):
        raise ValueError(
            "Requested experiment metadata differs from the frozen manifest"
        )
    if manifest["dataset"]["layout"] == GQA_LAYOUT:
        _verify_gqa_manifest(
            manifest, verify_images=verify_images, verify_annotations=verify_annotations
        )
        return manifest
    samples = manifest["samples"]
    count = manifest["sampling"]["per_difficulty"]
    groups = manifest["dataset"]["difficulty_counts"]
    expected = {row["difficulty"]: min(row["count"], count) for row in groups}
    if (
        not samples
        or Counter(sample["difficulty"] for sample in samples) != expected
        or len({row["sample_key"] for row in samples}) != len(samples)
    ):
        raise ValueError("Terminal manifest has an inconsistent stratified sample set")
    if any(
        row["selected"] != expected[row["difficulty"]]
        or row["shortfall"] != count - row["selected"]
        for row in groups
    ):
        raise ValueError("Terminal manifest has inconsistent difficulty counts")
    if verify_annotations:
        root = Path(manifest["dataset"]["root"]).resolve(strict=True)
        for source in manifest["dataset"]["rows_files"]:
            if (
                _file_sha(_inside(root, source["path"], kind="Annotation"))
                != source["sha256"]
            ):
                raise ValueError(f"Locked annotation source changed: {source['path']}")
    if verify_images:
        for sample in samples:
            load_image(manifest, sample)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("lock", help="Create a manifest without running models")
    lock.add_argument("--dataset-root", required=True, type=Path)
    lock.add_argument(
        "--rows-file",
        type=Path,
        help="Default uses the official VisualProbe validation split layout",
    )
    lock.add_argument(
        "--rows-key",
        help="Optional dotted key for JSON annotation lists stored inside an object",
    )
    lock.add_argument(
        "--field", action="append", default=[], metavar="CANONICAL=SOURCE"
    )
    lock.add_argument("--output", required=True, type=Path)
    lock.add_argument(
        "--experiment",
        type=Path,
        help="Optional JSON run settings to freeze with the sample set",
    )
    lock.add_argument("--seed", default=42, type=int)
    lock.add_argument("--per-difficulty", default=5, type=int)
    lock_gqa = commands.add_parser("lock-gqa", help="Select image episodes from GQA ZIPs")
    lock_gqa.add_argument("--dataset-root", required=True, type=Path)
    lock_gqa.add_argument("--output", required=True, type=Path)
    lock_gqa.add_argument("--experiment", type=Path)
    lock_gqa.add_argument("--seed", default=42, type=int)
    lock_gqa.add_argument("--images", default=10, type=int)
    lock_gqa.add_argument("--questions-per-image", default=3, type=int)
    verify = commands.add_parser(
        "verify", help="Check the frozen manifest and local payloads"
    )
    verify.add_argument("manifest", type=Path)
    verify.add_argument(
        "--experiment",
        type=Path,
        help="Optional JSON run settings to compare with the lock",
    )
    args = parser.parse_args(argv)
    experiment = json.loads(args.experiment.read_bytes()) if args.experiment else None
    if args.command == "lock":
        fields = dict(value.split("=", 1) for value in args.field if "=" in value)
        if len(fields) != len(args.field):
            parser.error("Each --field requires a unique CANONICAL=SOURCE mapping")
        result = lock_manifest(
            args.dataset_root,
            args.rows_file,
            args.output,
            seed=args.seed,
            per_difficulty=args.per_difficulty,
            fields=fields,
            rows_key=args.rows_key,
            experiment=experiment,
        )
    elif args.command == "lock-gqa":
        result = lock_gqa_manifest(
            args.dataset_root,
            args.output,
            seed=args.seed,
            image_count=args.images,
            questions_per_image=args.questions_per_image,
            experiment=experiment,
        )
    else:
        result = load_locked_manifest(args.manifest, experiment=experiment)
    counts = (
        {"eligible_images": result["dataset"]["eligible_image_count"],
         "questions_per_image": result["sampling"]["questions_per_image"]}
        if result["dataset"]["layout"] == GQA_LAYOUT
        else {"difficulty_counts": result["dataset"]["difficulty_counts"]}
    )
    print(
        json.dumps(
            {
                "fingerprint": result["fingerprint"],
                "samples": len(result["samples"]),
                **counts,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
