"""Private model subprocess for the single-image memory builder."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from time import perf_counter

import torch
from safetensors.torch import load_file

from .capture import CAPTURE_ENV


def run(directory: Path):
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.utils.hf_transformers_utils import get_processor

    config_file = directory / "worker.json"
    config = json.loads(config_file.read_text())
    model_path = config["model_path"]
    processor = get_processor(model_path, local_files_only=True)
    messages = [{"role": "user", "content": [{"type": "image"}]}]
    template = processor.chat_template or processor.tokenizer.chat_template
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, chat_template=template
    )
    config_file.write_text(json.dumps(dict(config, chat_template=template, rendered_prompt=prompt)))

    os.environ[CAPTURE_ENV] = str(directory)
    # The dedicated worker has no shared requests or cached embeddings. A zero
    # budget disables the outer MultiModalCache; the capture also bypasses the
    # fork's independent MockKVManager vision-embedding and decoder K/V lookups.
    os.environ["SGLANG_VLM_CACHE_SIZE_MB"] = "0"
    asyncio.set_event_loop(asyncio.new_event_loop())
    startup = perf_counter()
    phase_times = {}
    with Engine(
        model_path=model_path,
        dtype=config["dtype"],
        tp_size=1,
        pp_size=1,
        device="cuda",
        attention_backend="flashinfer",
        mm_attention_backend="triton_attn",
        sampling_backend="pytorch",
        mem_fraction_static=config["mem_fraction_static"],
        max_total_tokens=config["context_length"] + 128,
        max_prefill_tokens=config["context_length"],
        context_length=config["context_length"],
        max_running_requests=1,
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        random_seed=0,
        log_level="info",
    ) as engine:
        engine_startup_s = perf_counter() - startup
        outputs = {}
        phases = ("baseline", "capture", "reuse") if config["verify_capture"] else ("capture",)
        for mode in phases:
            # Engine.generate is synchronous and completes before changing mode.
            (directory / "request.json").write_text(json.dumps({
                "mode": mode, "context_length": config["context_length"],
                "max_image_tokens": config["max_image_tokens"],
            }))
            started = perf_counter()
            result = engine.generate(
                prompt=prompt,
                image_data=str(directory / "submitted.png"),
                sampling_params={"max_new_tokens": 1, "temperature": 0},
            )
            phase_times[mode] = perf_counter() - started
            output_ids = result.get("output_ids") or []
            completion_tokens = result.get("meta_info", {}).get("completion_tokens")
            if not (directory / f"{mode}.json").is_file() or completion_tokens != 1 or not output_ids:
                raise RuntimeError(f"Worker did not finish a full {mode} prefill: {result}")
            # This version's detokenizer includes a read-back prefix in output_ids.
            outputs[mode] = {"output_ids": output_ids[-1:], "text": result["text"]}

    validation = {"phases": outputs, "engine_startup_s": engine_startup_s, "phase_s": phase_times}
    if config["verify_capture"]:
        baseline = load_file(str(directory / "baseline.safetensors"))["logits"].float()
        from safetensors import safe_open

        with safe_open(directory / "capture.safetensors", framework="pt") as captured:
            enabled = captured.get_tensor("logits").float()
        reused = load_file(str(directory / "reuse.safetensors"), device="cpu")["logits"].float()
        if not all(torch.isfinite(logits).all() for logits in (baseline, enabled, reused)):
            raise RuntimeError("Non-finite prefill logits")
        torch.testing.assert_close(baseline, enabled, atol=0, rtol=0)
        if outputs["baseline"]["output_ids"] != outputs["capture"]["output_ids"]:
            raise RuntimeError("Capture changed the deterministic greedy output")
        warm = json.loads((directory / "reuse.json").read_text())
        validation.update(
            logit_atol=0, logit_rtol=0,
            logit_max_abs_diff=(baseline - enabled).abs().max().item(),
            logits_shape=list(baseline.shape),
            reuse_layers=len(warm["reused_layers"]),
            reuse_recomputed_visual_rows=warm["recomputed_visual_rows"],
            full_prefix_reuse_max_abs_diff=(baseline - reused).abs().max().item(),
            full_prefix_reuse_mean_abs_diff=(baseline - reused).abs().mean().item(),
            full_prefix_reuse_greedy_equal=outputs["baseline"]["output_ids"] == outputs["reuse"]["output_ids"],
        )
    # CUDA discovery here reads device properties without retaining a model.
    validation["runtime"] = {
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (directory / "verification.json").write_text(json.dumps(validation, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    run(parser.parse_args().directory.resolve())


if __name__ == "__main__":
    main()
