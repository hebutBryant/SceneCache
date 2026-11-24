"""Public initialization API for a one-image, one-GPU memory."""

import hashlib
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
from time import perf_counter

from PIL import Image, ImageOps
from safetensors.torch import load_file

from .capture import CAPTURE_ENV, KEY_STAGE, VALUE_STAGE
from .mapping import assemble_memory
from .sam3_adapter import create_masks
from .schema import ImageGeometry, ImageMemory, MaskInput, SourceKVMetadata, content_fingerprint


@dataclass(frozen=True)
class BuildConfig:
    model_path: Path
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    sam3_checkpoint: Path | None = None
    sam_prompts_file: Path | None = None
    masks_file: Path | None = None
    sam3_python: Path | None = None
    sam3_source: Path | None = None
    sam3_device: str | None = None
    sam3_score_threshold: float = 0.5
    sam3_mask_threshold: float = 0.5
    sam3_max_candidates: int | None = None
    sam3_min_area: int = 1
    mask_token_overlap_threshold: float = 0.0
    contact_mode: str = "pixel_cells"
    max_image_tokens: int = 2048
    max_image_edge: int | None = None
    context_length: int = 4096
    mem_fraction_static: float | None = None
    verify_capture: bool = False


def _cuda_selection(device: str) -> str:
    match = re.fullmatch(r"cuda(?::(\d+))?", device)
    if not match:
        raise ValueError("Qwen3-VL capture requires a single CUDA device, e.g. cuda:0")
    ordinal = int(match.group(1) or 0)
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current is None:
        return str(ordinal)
    visible = [name.strip() for name in current.split(",") if name.strip()]
    if ordinal >= len(visible) or visible[ordinal] == "-1":
        raise ValueError(f"{device} is outside CUDA_VISIBLE_DEVICES={current}")
    return visible[ordinal]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _model_files(model_path: Path) -> tuple[list[Path], list[Path]]:
    index = model_path / "model.safetensors.index.json"
    if index.is_file():
        weight_names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        weight_names = ["model.safetensors"]
    if any(Path(name).name != name for name in weight_names):
        raise ValueError("Expected model shards in the local model directory")
    weights = [model_path / name for name in weight_names]
    processor = sorted(path for path in model_path.iterdir() if path.is_file() and (
        path.suffix in {".json", ".jinja", ".model", ".tiktoken"} or path.name == "merges.txt"
    ))
    return weights, processor


def _fingerprint_files(paths):
    hashes = {}
    for path in paths:
        before = path.stat()
        hashes[path.name] = _hash_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise RuntimeError(f"Local model asset changed while being hashed: {path}")
    return hashes


def _run_worker(directory: Path, config: BuildConfig, model_path: Path, visible_gpu: str):
    job = {
        "model_path": str(model_path), "dtype": config.dtype,
        "mem_fraction_static": config.mem_fraction_static,
        "context_length": config.context_length, "max_image_tokens": config.max_image_tokens,
        "verify_capture": config.verify_capture,
    }
    (directory / "worker.json").write_text(json.dumps(job))
    env = os.environ.copy()
    for name in (
        CAPTURE_ENV, "SGLANG_VISUAL_KV_MODE", "SGLANG_VISUAL_KV_DISABLE_DECODER_CACHE",
        "SGLANG_VISUAL_KV_DISABLE_VISION_CACHE",
    ):
        env.pop(name, None)
    env.update(
        CUDA_VISIBLE_DEVICES=visible_gpu,
        CUDA_DEVICE_ORDER=env.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "4"),
        SGLANG_CPU_WORKERS=env.get("SGLANG_CPU_WORKERS", "2"),
        MAX_JOBS=env.get("MAX_JOBS", "4"),
    )
    command = [sys.executable, "-m", f"{__package__}.capture_worker", str(directory)]
    with (directory / "qwen.log").open("w+") as log:
        process = subprocess.Popen(
            command, env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            status = process.wait(timeout=900)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if status:
            log.seek(0)
            details = "\n".join(log.read().splitlines()[-60:])
            raise RuntimeError(f"Qwen capture worker failed ({status}):\n{details}")
    metadata = json.loads((directory / "capture.json").read_text())
    verification = json.loads((directory / "verification.json").read_text())
    rendered = json.loads((directory / "worker.json").read_text())
    tensors = load_file(str(directory / "capture.safetensors"), device="cpu")
    source_kv = {
        layer: (tensors.pop(f"{layer}.k"), tensors.pop(f"{layer}.v"))
        for layer in metadata["expected_layers"]
    }
    if set(tensors) != {"logits"}:
        raise RuntimeError("Unexpected tensors in the capture output")
    return metadata, verification, rendered, source_kv


def build_image_memory(
    image: str | Path | Image.Image,
    config: BuildConfig,
    *,
    prepared_masks: tuple[list[MaskInput], dict] | None = None,
    timings: dict[str, float] | None = None,
) -> ImageMemory:
    """Build from a fresh single-image prefill without connecting to an existing server."""
    model_path = Path(config.model_path).absolute()
    model = json.loads((model_path / "config.json").read_text())
    if model.get("architectures") != ["Qwen3VLForConditionalGeneration"] or model.get("quantization_config"):
        raise ValueError("Supported architecture is local, unquantized dense Qwen3VLForConditionalGeneration")
    vision, text = model["vision_config"], model["text_config"]
    if config.dtype not in {"bfloat16", "float16"}:
        raise ValueError("Expected bfloat16 or float16")
    if not 0 < config.max_image_tokens < config.context_length - 1:
        raise ValueError("Set a positive image token budget smaller than the full context length")
    if config.max_image_edge is not None and config.max_image_edge < 1:
        raise ValueError("Maximum image edge must be positive")
    if not 0 <= config.mask_token_overlap_threshold <= 1 or config.contact_mode not in {"pixel_cells", "overlap_only"}:
        raise ValueError("Invalid region overlap threshold or contact mode")
    visible_gpu = _cuda_selection(config.device)
    mark = perf_counter()
    weights, processor_files = _model_files(model_path)
    weight_hashes = _fingerprint_files(weights)
    processor_hashes = _fingerprint_files(processor_files)
    if timings is not None:
        timings["asset_fingerprint_s"] = perf_counter() - mark

    if isinstance(image, Image.Image):
        original = ImageOps.exif_transpose(image).convert("RGB")
    else:
        with Image.open(image) as source:
            original = ImageOps.exif_transpose(source).convert("RGB")
    if prepared_masks is None:
        masks, segmentation = create_masks(
            original, masks_file=config.masks_file,
            sam3_checkpoint=config.sam3_checkpoint, sam_prompts_file=config.sam_prompts_file,
            sam3_python=config.sam3_python, sam3_source=config.sam3_source,
            sam3_device=config.sam3_device or config.device,
            sam3_score_threshold=config.sam3_score_threshold, sam3_mask_threshold=config.sam3_mask_threshold,
            sam3_max_candidates=config.sam3_max_candidates, sam3_min_area=config.sam3_min_area,
        )
    else:
        if config.masks_file is not None or config.sam_prompts_file is not None:
            raise ValueError("Prepared masks cannot be combined with a configured mask source")
        masks, segmentation = prepared_masks
    print(f"Segmentation retained {len(masks)} masks ({segmentation['backend']}). Starting Qwen prefill.", file=sys.stderr)
    submitted = original.copy()
    if config.max_image_edge is not None:
        submitted.thumbnail((config.max_image_edge, config.max_image_edge), Image.Resampling.LANCZOS)

    mark = perf_counter()
    with tempfile.TemporaryDirectory(prefix="vlcache-scene-graph-") as temporary:
        directory = Path(temporary)
        submitted.save(directory / "submitted.png")
        metadata, verification, rendered, source_kv = _run_worker(directory, config, model_path, visible_gpu)
    if timings is not None:
        timings["capture_worker_s"] = perf_counter() - mark
    mark = perf_counter()

    observed = metadata["geometry"]
    if tuple(observed["submitted_size"]) != submitted.size:
        raise RuntimeError("Qwen received a different image extent")
    patch, merge = metadata["patch_size"], metadata["spatial_merge_size"]
    if not (patch == observed["processor_patch_size"] == vision["patch_size"]
            and merge == observed["processor_merge_size"] == vision["spatial_merge_size"]):
        raise RuntimeError("The processor and loaded Qwen patch/merger geometry disagree")
    temporal, patch_rows, patch_cols = metadata["grid_thw"]
    processed_size = (patch_cols * patch, patch_rows * patch)
    geometry = ImageGeometry(
        original_size=original.size, processed_size=processed_size,
        grid_thw=(temporal, patch_rows, patch_cols), patch_size=patch, spatial_merge_size=merge,
        steps=[
            {"operation": "exif_transpose_and_rgb", "output_size": list(original.size)},
            {"operation": "resize", "implementation": "Pillow.thumbnail LANCZOS", "output_size": list(submitted.size)},
            {"operation": "resize", "implementation": "SGLang qwen_vl.resize_image", "output_size": observed["fork_resize_size"]},
            {"operation": "resize", "implementation": observed["processor_class"], "output_size": list(processed_size)},
        ],
    )
    geometry.validate()
    rows, cols = geometry.token_grid
    count = rows * cols
    source_indices = metadata["source_indices"]
    mrope_positions = [metadata["source_mrope_positions"][i] for i in source_indices]
    if len(source_indices) != count or metadata["embedding_shape"][0] != count:
        raise RuntimeError("Visual span, encoder rows and merger grid disagree")
    first = mrope_positions[0]
    expected_grid = [
        [first[0], first[1] + row, first[2] + col]
        for row in range(rows) for col in range(cols)
    ]
    if mrope_positions != expected_grid:
        raise RuntimeError("Captured MRoPE positions do not follow Qwen3-VL's merged row-major order")
    width = text["head_dim"] * text["num_key_value_heads"]
    if sorted(source_kv) != list(range(text["num_hidden_layers"])) or any(
        tuple(tensor.shape) != (count, width)
        for pair in source_kv.values() for tensor in pair
    ):
        raise RuntimeError("The captured decoder K/V bank does not match the loaded model's dimensions")
    source_ids = metadata["source_token_ids"]
    captured_stage = {"key": KEY_STAGE, "value": VALUE_STAGE}
    metadata_fields = {
        "rendered_prompt": rendered["rendered_prompt"],
        "scheduler_input_ids": metadata["scheduler_input_ids"],
        "source_linear_positions": metadata["source_linear_positions"],
        "source_mrope_positions": metadata["source_mrope_positions"],
        "image_token_id": metadata["image_token_id"],
        "embedding_shape": metadata["embedding_shape"],
        "deepstack_visual_indexes": metadata["deepstack_visual_indexes"],
        "kv_heads": text["num_key_value_heads"], "head_dim": text["head_dim"],
        "num_hidden_layers": text["num_hidden_layers"],
        "legacy_reused_layers": metadata["reused_layers"],
        "legacy_write_layers": metadata["legacy_write_layers"],
        "recomputed_visual_rows": metadata["recomputed_visual_rows"],
        "verification": verification,
    }
    kv_metadata = SourceKVMetadata(
        key_representation_stage=KEY_STAGE, value_representation_stage=VALUE_STAGE,
        source_token_ids=source_ids, chat_template=rendered["chat_template"],
        qk_norm={"type": "per_head_RMSNorm", "eps": text["rms_norm_eps"]},
        rope={
            "theta": text["rope_theta"], "scaling": text["rope_scaling"],
            "max_position_embeddings": text["max_position_embeddings"],
        },
        details=metadata_fields,
    )
    versions = {name: version(name) for name in (
        "sglang", "torch", "transformers", "safetensors", "Pillow", "numpy",
        "sgl-kernel", "flashinfer-python",
    )}
    versions["python"] = ".".join(map(str, sys.version_info[:3]))
    srt = Path(__file__).resolve().parents[2]
    capture_sources = {
        name: _hash_file(srt / name) for name in [
            "models/qwen3.py", "models/qwen3_vl.py", "layers/attention/flashinfer_backend.py",
            "layers/rotary_embedding.py", "multimodal/processors/qwen_vl.py",
            "managers/mm_utils.py", "mem_cache/scene_graph_memory/capture.py",
            "mem_cache/scene_graph_memory/mapping.py",
        ]
    }
    fingerprints = {
        "model_path": str(model_path), "architecture": model["architectures"][0],
        "model": content_fingerprint({"weights": weight_hashes, "config": processor_hashes["config.json"]}),
        "model_weights_sha256": weight_hashes,
        "processor": content_fingerprint({"files": processor_hashes, "versions": versions}),
        "processor_files_sha256": processor_hashes,
        "source_prefix": content_fingerprint({
            "ids": source_ids, "template": rendered["chat_template"],
            "mrope": metadata["source_mrope_positions"],
        }),
        "capture": content_fingerprint({"sources": capture_sources, "stage": captured_stage}),
        "capture_sources_sha256": capture_sources,
        "versions": versions,
    }
    config_dict = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(config).items()
    }
    memory = assemble_memory(
        original, geometry, source_indices, mrope_positions, masks, source_kv, kv_metadata,
        fingerprints, config_dict, overlap_threshold=config.mask_token_overlap_threshold,
        contact_mode=config.contact_mode, segmentation=segmentation,
    )
    if timings is not None:
        timings["physical_index_s"] = perf_counter() - mark
    return memory
