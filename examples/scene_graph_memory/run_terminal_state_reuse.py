#!/usr/bin/env python3
"""Run terminal perception-state reuse on a locked Mini-o3 validation cohort."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import re
import statistics
import time
import weakref
from collections import defaultdict
from pathlib import Path

from sglang.srt.mem_cache.scene_graph_memory.schema import content_fingerprint

from terminal_evidence import (
    ANSWER,
    GROUNDING,
    REQUEST_SUFFIX,
    STOP_STRINGS,
    STATE_PROMPT_VERSION,
    CropTools,
    Observation,
    alternate_terminal,
    canonical_messages,
    canonical_state,
    parse_action,
)

DATA_ROOT = Path("/home/hdd4/lpz")
GQA_ROOT = Path("/home/hdd3/lpz/GQA")
MODEL_ROOT = Path("/home/hdd3/yaoyf/Mini-o3")
MODES = ("full_history", "terminal_kv_slice", "tool_result_cache", "canonical_state_kv")
GENERATION = {
    "max_turns": 8, "max_new_tokens": 512, "temperature": 1.0,
    "dtype": "bfloat16", "attention_backend": "sdpa",
    "min_pixels": 40_000, "max_pixels": 2_000_000,
}
GQA_GENERATION = {
    **GENERATION, "max_turns": 4, "max_new_tokens": 192, "qa_max_new_tokens": 48,
    "temperature": 0.0, "qa_assistant_prefix": "<answer>", "allow_original_evidence": True,
    "grounding_hint": (
        "\nUse the grounding zoom tool to check one relevant region before giving the answer. "
        "Choose the region yourself from the image. Keep the answer short."
    ),
}
GQA_REQUEST_SUFFIX = (
    "\nGive only the short answer label, without reasoning, and close it with </answer>. "
    "If the visual evidence is insufficient, answer 'insufficient evidence'."
)
VISUALPROBE_PREVIEW_GENERATION = {
    **GENERATION, "max_turns": 4, "max_new_tokens": 256, "qa_max_new_tokens": 192,
    "temperature": 0.0, "allow_original_evidence": True, "trace_preview": True,
    "grounding_hint": (
        "\nUse the grounding zoom tool to inspect a relevant region before answering. "
        "Choose the region yourself from the image."
    ),
}
PARITY_TARGET = "<answer>Observed detail.</answer>"
# BF16 projection/SDPA can use different reductions with a split prefix. Keep
# both a pointwise and a globally normalized limit, and persist observed errors.
PARITY_LIMITS = {"atol": 0.15, "rtol": 0.01, "normalized_rmse": 0.01}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["visualprobe", "gqa"], default="visualprobe")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--questions-per-image", type=int, default=3)
    parser.add_argument("--skip-controlled-replay", action="store_true")
    parser.add_argument("--model-path", type=Path, default=MODEL_ROOT)
    parser.add_argument("--per-difficulty", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--visualprobe-split", choices=["easy", "medium", "hard"], action="append")
    parser.add_argument("--trace-preview", action="store_true", help="Bound the VisualProbe demo and archive complete inputs")
    parser.add_argument("--seed", type=int, choices=[42], default=42)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--rows-file", type=Path, help="Alternative local Mini-o3 validation annotations")
    parser.add_argument("--rows-key", help="Key of a nested JSON annotation list")
    parser.add_argument("--field", action="append", default=[], metavar="CANONICAL=SOURCE")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    args.data_root = args.data_root or (GQA_ROOT if args.dataset == "gqa" else DATA_ROOT)
    args.generation = GQA_GENERATION if args.dataset == "gqa" else GENERATION
    if args.trace_preview:
        if args.dataset != "visualprobe":
            parser.error("--trace-preview requires VisualProbe")
        args.generation = VISUALPROBE_PREVIEW_GENERATION
    if args.visualprobe_split and (args.dataset != "visualprobe" or args.rows_file):
        parser.error("--visualprobe-split requires the standard VisualProbe layout")
    if args.dataset == "gqa":
        if not 1 <= args.num_images <= 10 or not 2 <= args.questions_per_image <= 5:
            parser.error("GQA requires at most 10 images and 2 to 5 questions per image")
        if args.rows_file or args.rows_key or args.field:
            parser.error("GQA uses the local validation archives")
    args.modes = args.modes.split(",")
    if len(set(args.modes)) != len(args.modes) or any(mode not in MODES for mode in args.modes):
        parser.error("--modes must be a unique comma-separated selection of known modes")
    if args.prepare_only == bool(args.manifest):
        parser.error("use --prepare-only to lock the cohort or explicitly pass --manifest to evaluate")
    if not args.prepare_only and (args.rows_file or args.rows_key or args.field):
        parser.error("annotation options are only used to prepare a locked manifest")
    try:
        args.fields = dict(field.split("=", 1) for field in args.field)
    except ValueError:
        parser.error("--field must be CANONICAL=SOURCE")
    return args


def model_identity(root):
    root = root.resolve()
    checkpoint = root if (root / "config.json").is_file() else root / "Mini-o3-7B-v1"
    code_dir = root / "code" if (root / "code").is_dir() else root.parent / "code"
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("model_type") != "qwen2_5_vl" or checkpoint.name != "Mini-o3-7B-v1":
        raise ValueError("expected the specified Mini-o3-7B-v1 Qwen2.5-VL checkpoint")
    files = {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(checkpoint.iterdir())
        if file.suffix in {".json", ".jinja", ".txt"} and file.is_file()
    }
    if not {"model.safetensors.index.json", "tokenizer_config.json", "preprocessor_config.json"} <= files.keys():
        raise ValueError("Mini-o3 model index, tokenizer or image processor is missing")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shards = {}
    for name in sorted(set(index["weight_map"].values())):
        stat = (checkpoint / name).stat()
        shards[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if not shards or not (code_dir / "verl/trainer/constants.py").is_file():
        raise ValueError("missing local Mini-o3 checkpoint shards or official tool code")
    identity = {
        "checkpoint": str(checkpoint), "code_dir": str(code_dir), "config_sha256": files,
        "shard_file_identity": shards, "model_type": config["model_type"],
        "num_hidden_layers": config.get("num_hidden_layers", config.get("text_config", {}).get("num_hidden_layers")),
        "num_key_value_heads": config.get("num_key_value_heads", config.get("text_config", {}).get("num_key_value_heads")),
    }
    identity["fingerprint"] = content_fingerprint(identity)
    return identity


def timed(function, device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, (time.perf_counter() - started) * 1000


def allocated(device):
    import torch

    return torch.cuda.memory_allocated(device) if device.type == "cuda" else None


def reserved(device):
    import torch

    return torch.cuda.memory_reserved(device) if device.type == "cuda" else None


def peak(device):
    import torch

    return torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None


def reset_peak(device):
    import torch

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def rng(runtime, seed, sample_id, request):
    import torch

    entropy = hashlib.sha256(f"{seed}\0{sample_id}\0{request}".encode()).digest()
    return torch.Generator(device=runtime.device).manual_seed(int.from_bytes(entropy[:8], "big"))


def assistant_history_ids(result, runtime):
    ids = list(result.generated_ids)
    # The template contributes a single closing EOM. An emitted EOM is already
    # recorded in raw output and must not be inserted a second time.
    eom = runtime.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if ids and ids[-1] == eom:
        ids.pop()
    return ids


def prep(runtime, messages, observations, assistant_ids=None, add_generation_prompt=False, assistant_prefix=""):
    return runtime.prepare(
        messages, [item.image for item in observations],
        assistant_token_ids=assistant_ids,
        add_generation_prompt=add_generation_prompt,
        assistant_prefix=assistant_prefix,
    )


def persist_observation(directory, observation):
    observation.image.save(directory / observation.metadata()["file"])


def acquire(runtime, tools, sample, image, directory, seed, *, generation=GENERATION):
    """Make one autonomous search. Keep only the last pre-answer observation boundary."""
    from sglang.srt.mem_cache.scene_graph_memory.terminal_kv import cache_stats

    directory.mkdir(parents=True, exist_ok=True)
    observations = [Observation.original(image)]
    persist_observation(directory, observations[0])
    messages = tools.initial_messages(sample["question"] + generation.get("grounding_hint", ""))
    assistant_ids = {}
    search_rng = rng(runtime, seed, sample["sample_key"], "search")
    trace = {
        "path_kind": "autonomous", "status": "running", "turns": [],
        "initial_messages": messages[:], "observations": [observations[0].metadata()],
        "terminal_boundary": None, "verification_status": "unverified",
    }
    write_json(directory / "trace.json", trace)
    state = boundary = boundary_state = None
    boundary_messages = boundary_observations = boundary_ids = None
    reset_peak(runtime.device)
    started = time.perf_counter()
    if generation.get("allow_original_evidence"):
        boundary_messages, boundary_observations, boundary_ids = messages[:], observations[:], {}
        boundary, trace["original_observation_processor_ms"] = timed(
            lambda: prep(runtime, boundary_messages, boundary_observations), runtime.device,
        )
        boundary_state, trace["original_observation_prefill_ms"] = timed(
            lambda: runtime.prefill(boundary), runtime.device,
        )
        state = runtime.branch(boundary_state)
    for turn in range(1, generation["max_turns"] + 1):
        prepared, prepare_ms = timed(lambda: prep(
            runtime, messages, observations, assistant_ids, add_generation_prompt=True
        ), runtime.device)
        turn_trace = {"turn": turn, "input": prepared.metadata, "processor_ms": prepare_ms}
        trace["turns"].append(turn_trace)
        write_json(directory / "trace.json", trace)
        result = runtime.generate(
            prepared, state=state, generator=search_rng,
            max_new_tokens=generation["max_new_tokens"], temperature=generation["temperature"],
            stop_strings=("</grounding>", "</answer>"),
        )
        state = result.state
        turn_trace.update({
            "generated_ids": list(result.generated_ids), "generated_text": result.text,
            "generation": result.metrics, "cache": cache_stats(state.cache),
        })
        grounding = GROUNDING.search(result.text)
        answer = ANSWER.search(result.text)
        if not grounding or (answer and answer.start() < grounding.start()):
            trace["status"] = "model_stopped_with_answer" if answer else "incomplete_tool_or_answer"
            write_json(directory / "trace.json", trace)
            break
        if turn == generation["max_turns"]:
            trace["status"] = "turn_limit"
            write_json(directory / "trace.json", trace)
            break

        assistant_ids[len(messages)] = assistant_history_ids(result, runtime)
        messages.append({"role": "assistant", "content": result.text})
        tool_start = time.perf_counter()
        try:
            action = parse_action(grounding[1])
            turn_trace["action"] = action
            observation = tools.observe(observations, action, len(messages))
        except (ValueError, AssertionError) as exc:
            turn_trace["tool_failure"] = str(exc)
            messages.append(tools.failure(exc))
        else:
            observations.append(observation)
            messages.append(tools.response(turn, observation))
            persist_observation(directory, observation)
            trace["observations"].append(observation.metadata())
            turn_trace["observation_id"] = observation.index
            boundary_messages, boundary_observations, boundary_ids = (
                messages[:], observations[:], assistant_ids.copy()
            )
            boundary, boundary_prepare_ms = timed(
                lambda: prep(runtime, boundary_messages, boundary_observations, boundary_ids),
                runtime.device,
            )
            if boundary.logical_length <= prepared.logical_length:
                raise ValueError("new crop did not extend the actual prompt")
            turn_trace["boundary_processor_ms"] = boundary_prepare_ms
            boundary_state, boundary_vision_ms = timed(
                lambda: runtime.prefill(boundary, state=state), runtime.device
            )
            turn_trace["new_observation_prefill_ms"] = boundary_vision_ms
            turn_trace["new_observation_visual_tokens"] = (
                boundary.image_spans[-1].stop - boundary.image_spans[-1].start
            )
            state = runtime.branch(boundary_state)
        turn_trace["tool_and_evidence_ms"] = (time.perf_counter() - tool_start) * 1000
        write_json(directory / "trace.json", trace)

    trace["elapsed_ms"] = (time.perf_counter() - started) * 1000
    trace["acquisition_gpu_peak_bytes"] = peak(runtime.device)
    if boundary is None:
        trace["terminal_status"] = "no_successful_crop"
        write_json(directory / "trace.json", trace)
        del state
        return None, trace

    # The latest observation's cache was sealed before entering lookahead.
    # Decode writes replace rows on a separate layer container, so even the
    # backing storage of the sealed cache contains no final-answer rows.
    del state, result
    runtime.verify_prefix(boundary, boundary_state)
    original_only = boundary_observations[-1].parent is None
    trace["terminal_status"] = "whole_frame_original_evidence" if original_only else "model_requested_crop"
    trace["acquisition_new_visual_encodes"] = int("original_observation_prefill_ms" in trace) + sum(
        turn.get("generation", {}).get("vision_input_count", 0) + ("new_observation_prefill_ms" in turn)
        for turn in trace["turns"]
    )
    trace["terminal_boundary"] = {
        "phase": (
            "after_original_observation_before_assistant_generation" if original_only
            else "after_tool_observation_before_assistant_generation"
        ),
        "observation_id": boundary_observations[-1].index,
        "input": boundary.metadata, "cache": cache_stats(boundary_state.cache),
        "lookahead_storage_separate": True, "answer_tokens_discarded": True,
    }
    write_json(directory / "trace.json", trace)
    return [boundary_state, boundary, boundary_messages, boundary_observations, boundary_ids], trace


def replay(runtime, tools, source, original_trace, directory, *, generation=GENERATION):
    from sglang.srt.mem_cache.scene_graph_memory.terminal_kv import cache_stats

    _, _, messages, observations, ids = source
    terminal = observations[-1]
    if terminal.parent is None:
        raise ValueError("no model-requested crop for pixel-matched replay")
    cut = terminal.message_index - 1
    replay_messages = messages[:cut]
    earlier_ids = {index: value for index, value in ids.items() if index < cut}
    if len(earlier_ids) + 2 >= generation["max_turns"]:
        raise ValueError("controlled path would leave no answer turn within the shared budget")
    path = alternate_terminal(tools, observations[:-1], terminal, cut)
    directory.mkdir(parents=True, exist_ok=True)
    replay_observations = observations[:-1]
    actions = []
    for action, observation in path:
        replay_messages.append({
            "role": "assistant",
            "content": "<grounding>" + json.dumps(action, separators=(",", ":")) + "</grounding>",
        })
        replay_messages.append(tools.response((len(replay_messages) - 1) // 2, observation))
        replay_observations.append(observation)
        actions.append({"tool_action": action, "observation_id": observation.index})
    for observation in replay_observations:
        persist_observation(directory, observation)
    prepared, prepare_ms = timed(lambda: prep(runtime, replay_messages, replay_observations, earlier_ids),
                                 runtime.device)
    state, prefill_ms = timed(lambda: runtime.prefill(prepared), runtime.device)
    trace = {
        "path_kind": "controlled_tool_replay", "status": "paired",
        "declared_autonomous": False, "actions": actions,
        "acquisition_message_history": replay_messages,
        "preserved_assistant_token_ids": earlier_ids,
        "observations": [item.metadata() for item in replay_observations],
        "original_terminal_observation_id": original_trace["terminal_boundary"]["observation_id"],
        "terminal_equivalence": {"size": list(terminal.image.size),
                                 "pixel_sha256": terminal.image_hash, "bytewise_equal": True},
        "terminal_boundary": {"input": prepared.metadata, "cache": cache_stats(state.cache)},
        "processor_ms": prepare_ms, "prefill_ms": prefill_ms,
    }
    write_json(directory / "trace.json", trace)
    return [state, prepared, replay_messages, replay_observations, earlier_ids], trace


def retained_indices(runtime, prepared, messages, observations, assistant_ids):
    # Keep system + exact terminal tool response, including its template and
    # vision boundary markers. Assistant action/reasoning and earlier views go.
    system = prep(runtime, messages[:1], [])
    cut = observations[-1].message_index
    previous = prep(runtime, messages[:cut], observations[:-1], assistant_ids)
    if not (system.logical_length <= previous.logical_length < prepared.logical_length):
        raise ValueError("indexed terminal tool spans are invalid")
    runtime.verify_prepared_prefix(system, prepared)
    runtime.verify_prepared_prefix(previous, prepared)
    return [*range(system.logical_length), *range(previous.logical_length, prepared.logical_length)], (
        [0, system.logical_length], [previous.logical_length, prepared.logical_length]
    )


def storage_refs(cache):
    return [weakref.ref(tensor) for layer in cache.layers for tensor in (layer.keys, layer.values)]


def released(refs, base_refs, device, gpu_before, reserved_before, old_storage, kept_storage):
    _, collect_ms = timed(gc.collect, device)
    gpu_after = allocated(device)
    return {
        "live_old_tensor_references": sum(ref() is not None for ref in refs),
        "live_old_view_base_references": sum(ref() is not None for ref in base_refs),
        "gpu_allocated_before_bytes": gpu_before,
        "gpu_allocated_after_bytes": gpu_after,
        "gpu_reserved_before_bytes": reserved_before,
        "gpu_reserved_after_bytes": reserved(device),
        "gpu_allocated_delta_bytes": gpu_before - gpu_after if gpu_before is not None else None,
        "old_storage_bytes": old_storage,
        "retained_storage_bytes": kept_storage,
        "synchronized_lifetime_probe_ms": collect_ms,
    }


def sample_requests(sample):
    if "requests" in sample:
        return sample["requests"]
    return [
        {"name": "original", "question": sample["question"], "answer": sample["answer"]},
        {"name": "followup", **sample["followup"]},
    ]


def append_request(messages, question, assistant_prefix=""):
    suffix = GQA_REQUEST_SUFFIX if assistant_prefix else REQUEST_SUFFIX
    return [*messages, {"role": "user", "content": question + suffix}]


def score(text, reference):
    if not isinstance(reference, (str, int, float)) or isinstance(reference, bool):
        return None
    if text is None:
        return False
    if isinstance(reference, str) and re.fullmatch(r"[A-Z]", reference.strip()):
        choice = re.match(r"^\(([A-Z])\)|^([A-Z])\.", text.strip())
        if choice:
            return (choice[1] or choice[2]) == reference.strip()

    def normalize(value):
        normalized = str(value).casefold().strip().rstrip(".")
        return re.sub(r"\s+", " ", normalized)
    return normalize(text) == normalize(reference)


def run_request(runtime, mode, state, prepared, sample, request_name, path_kind, context):
    from sglang.srt.mem_cache.scene_graph_memory.terminal_kv import cache_stats

    initial = cache_stats(state.cache) if state is not None else None
    if state is not None:
        retained = set(state.retained_indices)
        visual_prefix_tokens = sum(
            index in retained for span in state.image_spans for index in range(span.start, span.stop)
        )
    else:
        visual_prefix_tokens = 0
    reset_peak(runtime.device)
    generation = context["generation"]
    assistant_prefix = generation.get("qa_assistant_prefix", "")
    started = time.perf_counter()
    result = runtime.generate(
        prepared, state=state, generator=rng(runtime, context["seed"], sample["sample_key"], request_name),
        max_new_tokens=generation.get("qa_max_new_tokens", generation["max_new_tokens"]),
        temperature=generation["temperature"],
        stop_strings=STOP_STRINGS,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    answer = ANSWER.search(assistant_prefix + result.text)
    new_evidence = bool(GROUNDING.search(result.text) or "<need_evidence>" in result.text)
    answer_text = answer[1].strip() if answer and not new_evidence else None
    if assistant_prefix and not answer and not new_evidence and result.metrics["stop_reason"] == "stop":
        # An end-of-message token also closes a short GQA label.
        raw = result.text.strip()
        if raw and len(raw.split()) <= 6 and not any(tag in raw for tag in ("<", ">")):
            answer_text = raw
    new_evidence |= answer_text is not None and answer_text.casefold().rstrip(".") == "insufficient evidence"
    if new_evidence:
        answer_text = None
    status = "needs_evidence" if new_evidence else "answered" if answer_text else "incomplete"
    request = next(item for item in sample_requests(sample) if item["name"] == request_name)
    reference = request["answer"]
    row = {
        "sample_key": sample["sample_key"],
        "sample_id": sample["id"], "difficulty": sample["difficulty"], "path": path_kind,
        "mode": mode, "request": request_name, "status": status, "answer": answer_text,
        "question_id": request.get("question_id"), "question": request["question"],
        "reference": reference, "evidence_kind": context.get("evidence_kind"),
        "qa_assistant_prefix": assistant_prefix,
        "matches_reference": score(answer_text, reference),
        "scoring": "exact_match_with_minio3_choice_format" if reference is not None else "manual_only_unscored",
        "fallback": "new_evidence_requested_no_automatic_extra_generation" if new_evidence else None,
        "generation": result.metrics, "generated_text": result.text,
        "generated_ids": list(result.generated_ids),
        "prompt_fingerprint": prepared.fingerprint,
        "request_ms": elapsed_ms, "steady_gpu_before_bytes": context["steady_gpu_before_bytes"],
        "gpu_peak_bytes": peak(runtime.device), "prefix_logical_kv_bytes": initial["logical_bytes"] if initial else 0,
        "prefix_allocated_kv_bytes": initial["storage_bytes"] if initial else 0,
        "prefix_gpu_kv_bytes": initial["storage_bytes"] if initial and runtime.device.type == "cuda" else 0,
        "retained_visual_tokens": visual_prefix_tokens,
        "retained_text_tokens": (initial["sequence_length"] if initial else 0) - visual_prefix_tokens,
        "new_visual_tokens": result.metrics["vision_input_tokens"],
        "new_text_prefill_tokens": result.metrics["suffix_tokens"] - result.metrics["vision_input_tokens"],
        "new_decoded_kv_tokens": result.metrics["generated_tokens"] - 1,
        "approximate": mode == "terminal_kv_slice" and len(state.retained_indices) < state.logical_length,
    }
    if generation.get("trace_preview"):
        row["input"] = prepared.metadata
    return row, result.state


def cache_pointers(stats):
    return {
        (entry["device"], entry["storage_ptr"])
        for layer in stats["layers"] for entry in (layer["key"], layer["value"])
    }


def cache_base_refs(cache):
    return [
        weakref.ref(value._base)
        for layer in cache.layers for value in (layer.keys, layer.values)
        if value._base is not None
    ]


def mode_requests(runtime, mode, state, inputs, sample, path_kind, timing):
    """Continue from immutable real K/V; each request owns its appended rows."""
    rows = []
    for name, prepared in inputs:
        timing["steady_gpu_before_bytes"] = allocated(runtime.device)
        start = time.perf_counter()
        working = runtime.branch(state) if state is not None else None
        branch_ms = (time.perf_counter() - start) * 1000
        row, updated = run_request(runtime, mode, working, prepared, sample, name, path_kind, timing)
        del updated, working
        row["cache_branch_ms"] = branch_ms
        row["request_ms"] += branch_ms
        if request_dir := timing.get("request_trace_dir"):
            write_json(request_dir / f"{mode}-{name}.json", row)
        rows.append(row)
    return rows


def logit_metrics(reference, comparison):
    import torch

    expected = torch.cat(reference).double()
    actual = torch.cat(comparison).double()
    delta = (expected - actual).abs()
    rmse = (delta.square().mean() / expected.square().mean().clamp_min(1e-12)).sqrt().item()
    pointwise = torch.allclose(
        actual, expected, atol=PARITY_LIMITS["atol"], rtol=PARITY_LIMITS["rtol"]
    )
    return {
        "max_absolute_error": delta.max().item(),
        "mean_absolute_error": delta.mean().item(),
        "normalized_rmse": rmse,
        "top1_match_fraction": (actual.argmax(-1) == expected.argmax(-1)).double().mean().item(),
        "tolerance": PARITY_LIMITS,
        "passed": pointwise and rmse <= PARITY_LIMITS["normalized_rmse"],
    }


def parity_check(runtime, canonical, inputs, bank_dir, metadata, directory):
    from sglang.srt.mem_cache.scene_graph_memory.terminal_kv import load_cache

    targets = runtime.processor.tokenizer(PARITY_TARGET, add_special_tokens=False).input_ids
    prepared = next(value for name, value in inputs if name == "original")
    cold = runtime.teacher_forced(prepared, targets=targets)
    cold_logits = cold.logits
    del cold
    bank, _ = load_cache(bank_dir, device=runtime.device, expected_metadata=metadata)
    continued = runtime.teacher_forced(prepared, targets=targets, state=runtime.restore(canonical, bank))
    check = {
        "comparison": "C cold canonical prompt versus D loaded prefix and same forced suffix",
        "input_fingerprint": prepared.fingerprint,
        "targets": targets,
        **logit_metrics(cold_logits, continued.logits),
    }
    write_json(directory / "canonical_parity.json", check)
    return check


def path_modes(runtime, source, trace, sample, directory, args, identity, bank_root, *, parity):
    from sglang.srt.mem_cache.scene_graph_memory.terminal_kv import (
        cache_stats,
        copy_cache,
        load_cache,
        save_cache,
    )

    original, prefix, messages, observations, assistant_ids = source
    source[0] = None
    terminal = observations[-1]
    generation = getattr(args, "generation", GENERATION)
    assistant_prefix = generation.get("qa_assistant_prefix", "")
    evidence_kind = "terminal_crop" if terminal.parent is not None else "original_image_no_crop"
    tools_only = {"seed": args.seed, "generation": generation, "evidence_kind": evidence_kind}
    if generation.get("trace_preview"):
        tools_only["request_trace_dir"] = directory / "request_traces"
    questions = [(request["name"], request["question"]) for request in sample_requests(sample)
                 if request["question"]]
    order = random.Random(f"{args.seed}:{sample['sample_key']}:requests")
    order.shuffle(questions)

    state_data, state_ms = timed(lambda: canonical_state(observations[0], terminal), runtime.device)
    write_json(directory / "state.json", {
        **state_data, "source_observation": terminal.metadata(),
        "selection_policy": (
            "last_successful_crop" if terminal.parent is not None else "whole_frame_original_evidence"
        ),
        "available_observation_count": len(observations) - 1,
        "followup_locked_before_z": True,
    })
    path_kind = trace["path_kind"]
    rows = []
    conversions = {mode: {} for mode in args.modes}
    path_info = {
        "state_hash": state_data["state_hash"],
        "evidence_hash": state_data["state"]["evidence_hash"],
        "canonicalization_ms": state_ms,
        "evidence_png_bytes": (directory / terminal.metadata()["file"]).stat().st_size,
        "full_history_tokens": prefix.logical_length,
        "full_history_visual_tokens": sum(span.stop - span.start for span in prefix.image_spans),
        "full_history_kv": cache_stats(original.cache),
        "source_is_pre_answer": True, "evidence_kind": evidence_kind,
        "prefix_compatibility": "ids_attention_mask_mrope_image_grid_pixel_and_patch_hash",
        "requests": [name for name, _ in questions],
    }
    path_info["state_metadata_bytes"] = (directory / "state.json").stat().st_size
    path_info["full_history_text_tokens"] = (
        prefix.logical_length - path_info["full_history_visual_tokens"]
    )
    write_json(directory / "path_metrics.json", path_info)

    historical_modes = {"full_history", "terminal_kv_slice"} & set(args.modes)
    compact = None
    full_logits = None
    if historical_modes:
        historical = []
        for name, question in questions:
            prepared, ms = timed(
                lambda: prep(
                    runtime, append_request(messages, question, assistant_prefix), observations,
                    assistant_ids, add_generation_prompt=True, assistant_prefix=assistant_prefix,
                ), runtime.device
            )
            historical.append((name, prepared))
            path_info.setdefault("historical_prompt_prepare_ms", {})[name] = ms
        if "full_history" in args.modes:
            rows.extend(mode_requests(
                runtime, "full_history", original, historical, sample, path_kind, tools_only,
            ))

        if "terminal_kv_slice" in args.modes:
            reset_peak(runtime.device)
            (indices, spans), row_select_ms = timed(
                lambda: retained_indices(runtime, prefix, messages, observations, assistant_ids), runtime.device,
            )
            compact, copy_ms = timed(lambda: runtime.compact(original, indices), runtime.device)
            compact_stats = cache_stats(compact.cache)
            if compact_stats["logical_bytes"] != compact_stats["storage_bytes"]:
                raise ValueError("compacted terminal K/V still has oversized underlying storage")
            if cache_pointers(compact_stats) & cache_pointers(cache_stats(original.cache)):
                raise ValueError("terminal K/V storage aliases original history")
            terminal_visual = prefix.image_spans[-1]
            conversions["terminal_kv_slice"] = {
                "select_ms": row_select_ms, "compact_copy_ms": copy_ms, "copied_kv_bytes": compact_stats["storage_bytes"],
                "peak_bytes": peak(runtime.device), "retained_spans": spans,
                "retained_source_indices": indices,
                "retained_logical_mrope_positions": prefix.position_ids[..., indices].tolist(),
                "key_stage": "post_mrope", "value_stage": "raw_projection",
                "original_cache_layout": "batch_head_physical_token_head_dim",
                "visual_tokens": terminal_visual.stop - terminal_visual.start,
                "text_tokens": len(indices) - terminal_visual.stop + terminal_visual.start,
                "compact_bank": compact_stats,
            }
            if parity and "full_history" in args.modes:
                original_input = next(value for name, value in historical if name == "original")
                targets = runtime.processor.tokenizer(PARITY_TARGET, add_special_tokens=False).input_ids
                full_logits = runtime.teacher_forced(
                    original_input, targets=targets, state=runtime.branch(original)
                ).logits

    old_stats = cache_stats(original.cache)
    old_refs, old_base_refs = storage_refs(original.cache), cache_base_refs(original.cache)
    gpu_before = allocated(runtime.device)
    reserved_before = reserved(runtime.device)
    del original
    path_info["old_storage_release"] = released(
        old_refs, old_base_refs, runtime.device, gpu_before, reserved_before, old_stats["storage_bytes"],
        cache_stats(compact.cache)["storage_bytes"] if compact is not None else 0,
    )
    if path_info["old_storage_release"]["live_old_tensor_references"] or path_info[
        "old_storage_release"
    ]["live_old_view_base_references"]:
        raise ValueError("old history tensors are still referenced at the release boundary")

    if compact is not None:
        rows.extend(mode_requests(
            runtime, "terminal_kv_slice", compact, historical, sample, path_kind, tools_only,
        ))
        if full_logits is not None:
            sparse = runtime.teacher_forced(original_input, targets=targets, state=runtime.branch(compact))
            approximation = logit_metrics(full_logits, sparse.logits)
            # No equivalence test for sparse context; report measured drift only.
            approximation.pop("passed")
            approximation.pop("tolerance")
            approximation.update({
                "comparison": "A entire acquired context versus B terminal slice",
                "input_fingerprint": original_input.fingerprint, "targets": targets,
            })
            path_info["terminal_slice_approximation"] = approximation
            write_json(directory / "terminal_slice_approximation.json", approximation)
            del sparse
        del compact

    canonical_modes = [mode for mode in args.modes if mode in ("tool_result_cache", "canonical_state_kv")]
    if canonical_modes:
        order.shuffle(canonical_modes)
        canon_messages = canonical_messages(state_data)
        canonical, base_prepare_ms = timed(lambda: prep(runtime, canon_messages, [terminal]), runtime.device)
        inputs = []
        for name, question in questions:
            prepared, ms = timed(
                lambda: prep(
                    runtime, append_request(canon_messages, question, assistant_prefix), [terminal],
                    add_generation_prompt=True, assistant_prefix=assistant_prefix,
                ), runtime.device
            )
            runtime.verify_prepared_prefix(canonical, prepared)
            inputs.append((name, prepared))
            path_info.setdefault("canonical_prompt_prepare_ms", {})[name] = ms
        path_info["canonical_setup_prepare_ms"] = base_prepare_ms
        path_info["canonical_prefix_tokens"] = canonical.logical_length
        path_info["canonical_prefix_visual_tokens"] = sum(
            span.stop - span.start for span in canonical.image_spans
        )
        write_json(directory / "canonical_prefix.json", canonical.metadata)
        metadata = {
            "model_fingerprint": identity["fingerprint"], "state_hash": state_data["state_hash"],
            "state_prompt_version": STATE_PROMPT_VERSION,
            "evidence_hash": state_data["state"]["evidence_hash"],
            "prefix": json.loads(json.dumps(canonical.metadata)),
            "dtype": str(runtime.model.dtype), "position_encoding": "qwen2_5_vl_mrope_three_axis",
            "processor": {"min_pixels": GENERATION["min_pixels"], "max_pixels": GENERATION["max_pixels"]},
        }
        for mode in canonical_modes:
            if mode == "tool_result_cache":
                conversions[mode] = {"canonicalize_ms": state_ms, "old_decoder_kv_bytes": 0}
                rows.extend(mode_requests(
                    runtime, mode, None, inputs, sample, path_kind, tools_only,
                ))
                continue

            reset_peak(runtime.device)
            bank_dir = bank_root / state_data["state_hash"]
            conversion = {"canonicalize_ms": state_ms, "path_independent_bank_hit": bank_dir.is_dir()}
            if not bank_dir.is_dir():
                canonical_cache, conversion["canonical_prefill_ms"] = timed(
                    lambda: runtime.prefill(canonical), runtime.device
                )
                conversion["canonical_vision_encodes"] = len(canonical.image_spans)
                cpu, conversion["cpu_snapshot_ms"] = timed(
                    lambda: copy_cache(canonical_cache.cache, device="cpu"), runtime.device
                )
                del canonical_cache
                conversion["cpu_snapshot_bytes"] = cache_stats(cpu)["storage_bytes"]
                _, conversion["disk_save_ms"] = timed(
                    lambda: save_cache(bank_dir, cpu, metadata=metadata), runtime.device
                )
                del cpu
            (bank, _), conversion["disk_load_and_restore_ms"] = timed(
                lambda: load_cache(bank_dir, device=runtime.device, expected_metadata=metadata), runtime.device
            )
            live, conversion["attach_ms"] = timed(lambda: runtime.restore(canonical, bank), runtime.device)
            conversion["load_kv_bytes"] = cache_stats(bank)["storage_bytes"]
            conversion["kv_h2d_bytes"] = conversion["load_kv_bytes"] if runtime.device.type == "cuda" else 0
            conversion["bank_file_bytes"] = sum(file.stat().st_size for file in bank_dir.iterdir())
            conversion["build_load_peak_bytes"] = peak(runtime.device)
            del bank
            rows.extend(mode_requests(
                runtime, mode, live, inputs, sample, path_kind, tools_only,
            ))
            del live
            conversions[mode] = conversion

        if parity and "canonical_state_kv" in canonical_modes:
            check = parity_check(runtime, canonical, inputs, bank_dir, metadata, directory)
            path_info["canonical_parity"] = check
            if not check["passed"]:
                note = "C/D logits exceeded diagnostic tolerance; see canonical_parity.json"
                if "requests" not in sample and not generation.get("trace_preview"):
                    raise ValueError(note)
                print(note, flush=True)

    acquisition_ms = trace.get("elapsed_ms", trace.get("processor_ms", 0) + trace.get("prefill_ms", 0))
    path_info["conversion"] = conversions
    path_info["mode_order"] = list(dict.fromkeys(row["mode"] for row in rows))
    for mode, conversion in conversions.items():
        mode_rows = [row for row in rows if row["mode"] == mode]
        if not mode_rows:
            continue
        prep_times = path_info.get(
            "canonical_prompt_prepare_ms" if mode in canonical_modes else "historical_prompt_prepare_ms", {}
        )
        conversion_ms = sum(value for key, value in conversion.items() if key.endswith("_ms"))
        episode_ms = acquisition_ms + conversion_ms
        if mode in canonical_modes:
            episode_ms += path_info["canonical_setup_prepare_ms"]
        for row in mode_rows:
            row["request_prepare_ms"] = prep_times.get(row["request"], 0)
            row["conversion_ms_once_per_episode"] = conversion_ms
            row["acquisition_ms_once_per_episode"] = acquisition_ms
            row["old_storage_released_before_mode"] = mode != "full_history"
            row["conversion_kv_h2d_bytes_once_per_episode"] = conversion.get("kv_h2d_bytes", 0)
            episode_ms += row["request_ms"] + row["request_prepare_ms"]
        for row in mode_rows:
            row["episode_ms"] = episode_ms

    comparison_modes = ("tool_result_cache", "canonical_state_kv")
    paired = {mode: [row for row in rows if row["mode"] == mode] for mode in comparison_modes}
    baseline, saved_prefix = (paired[mode] for mode in comparison_modes)
    if baseline and saved_prefix:
        path_info["cost_D_minus_C"] = {
            "mode_episode_ms": saved_prefix[0]["episode_ms"] - baseline[0]["episode_ms"],
            "all_post_build_requests_ms": sum(row["request_ms"] for row in saved_prefix)
            - sum(row["request_ms"] for row in baseline),
            "canonical_conversion_ms": saved_prefix[0]["conversion_ms_once_per_episode"]
            - baseline[0]["conversion_ms_once_per_episode"],
            "requests_compared": len(baseline),
            "episode_cost_is_component_sum": True,
        }
    write_json(directory / "path_metrics.json", path_info)
    return rows, path_info


def failed_rows(sample, modes, status, reason, path="autonomous"):
    for mode in modes:
        for question in sample_requests(sample):
            request = question["name"]
            yield {
                "sample_id": sample["id"], "sample_key": sample["sample_key"], "difficulty": sample["difficulty"],
                "mode": mode, "path": path, "request": request, "status": status,
                "reason": reason, "answer": None, "question_id": question.get("question_id"),
                "question": question["question"], "reference": question["answer"],
                "matches_reference": score(None, question["answer"]) if status != "not_run" else None,
            }


def mode_summary(rows):
    cohorts = defaultdict(list)
    for row in rows:
        cohorts[(str(row["difficulty"]), row["mode"], row["request"], row["path"])].append(row)
    summary = []
    for (difficulty, mode, request, path), group in sorted(cohorts.items()):
        successful = [entry for entry in group if entry.get("generation")]
        scored = [entry["matches_reference"] for entry in group if entry["matches_reference"] is not None]
        ttft = [entry["generation"]["first_token_seconds"] * 1000 for entry in successful]
        tpot = [
            (entry["generation"]["generation_seconds"] - entry["generation"]["first_token_seconds"])
            * 1000 / (entry["generation"]["generated_tokens"] - 1)
            for entry in successful if entry["generation"]["generated_tokens"] > 1
        ]
        summary.append({
            "difficulty": difficulty, "mode": mode, "request": request, "path": path,
            "n": len(group), "executed": len(successful), "answered": sum(e["status"] == "answered" for e in group),
            "scored": len(scored), "exact_match_count": sum(scored),
            "median_ttft_ms": statistics.median(ttft) if ttft else None,
            "median_tpot_ms": statistics.median(tpot) if tpot else None,
            "median_request_ms": statistics.median(e["request_ms"] for e in successful) if successful else None,
            "median_estimated_mode_episode_ms": statistics.median(e["episode_ms"] for e in successful) if successful else None,
        })
    return summary


def dump_results(output, rows, pairs, status):
    raw = output / "requests.jsonl"
    raw.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    columns = [
        "sample_key", "sample_id", "difficulty", "path", "mode", "request", "status",
        "question_id", "question", "reference", "evidence_kind",
        "answer", "matches_reference", "request_ms", "request_prepare_ms", "episode_ms",
        "prefix_gpu_kv_bytes", "prefix_allocated_kv_bytes", "gpu_peak_bytes",
        "ttft_ms", "tpot_ms", "prefill_ms", "physical_prefix_tokens", "fresh_tokens", "visual_encodes",
    ]
    with (output / "requests.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output_row = row.copy()
            if generation := row.get("generation"):
                output_row.update({
                    "ttft_ms": generation["first_token_seconds"] * 1000,
                    "prefill_ms": generation["prefill_seconds"] * 1000,
                    "physical_prefix_tokens": generation["prefix_physical_tokens"],
                    "fresh_tokens": generation["suffix_tokens"],
                    "visual_encodes": generation["vision_input_count"],
                })
                if generation["generated_tokens"] > 1:
                    output_row["tpot_ms"] = (
                        (generation["generation_seconds"] - generation["first_token_seconds"]) * 1000
                        / (generation["generated_tokens"] - 1)
                    )
            writer.writerow(output_row)
    summary = mode_summary(rows)
    write_json(output / "summary.json", {"status": status, "by_difficulty_mode": summary})
    report = [
        "# 终端感知状态复用",
        "",
        f"运行状态 `{status}`。TTFT 和 TPOT 单位为 ms，延迟只统计实际执行的请求。"
        "非原题的人工核验样本无可用独立参考时不计入正确数。"
        "本批次仅用于机制检查。",
        "",
        "| 难度 | 路径 | 模式 | 问题 | 执行/计划 | 正确/可评分 | TTFT 中位数 | TPOT 中位数 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summary:
        ttft = item["median_ttft_ms"]
        tpot = item["median_tpot_ms"]
        values = [
            item["difficulty"], item["path"], item["mode"], item["request"],
            f"{item['executed']}/{item['n']}", f"{item['exact_match_count']}/{item['scored']}",
            f"{ttft:.2f}" if ttft is not None else "—",
            f"{tpot:.2f}" if tpot is not None else "—",
        ]
        report.append("| " + " | ".join(values) + " |")
    (output / "report.md").write_text("\n".join(report) + "\n")
    write_json(output / "failures.json", [row for row in rows if row["status"] != "answered"])
    write_json(output / "pairs.json", pairs)
    write_json(output / "unscorable.json", [row for row in rows if row["matches_reference"] is None])


def main(argv=None):
    import terminal_dataset

    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / "selected_manifest.json"
    config_file = args.output_dir / "experiment_config.json"
    generation = args.generation
    gqa = args.dataset == "gqa"
    sampling = (
        {"image_count": args.num_images, "questions_per_image": args.questions_per_image}
        if gqa else {"per_difficulty": args.per_difficulty}
    )
    config = {"dataset": args.dataset, "data_root": str(args.data_root.resolve()),
              "modes": args.modes, "seed": args.seed, **sampling,
              "device": args.device, "generation": generation,
              "controlled_replay": not args.skip_controlled_replay}
    if args.visualprobe_split:
        config["visualprobe_splits"] = args.visualprobe_split
    try:
        identity = model_identity(args.model_path)
        config["model"] = identity
        shared_config = {
            "model_fingerprint": identity["fingerprint"], "generation": generation,
            "state_prompt_version": STATE_PROMPT_VERSION,
            "followup_budget": args.questions_per_image - 1 if gqa else 1,
            "alternate_path_budget": int(not args.skip_controlled_replay),
        }
        if args.visualprobe_split:
            shared_config["visualprobe_splits"] = args.visualprobe_split
        if args.prepare_only:
            if gqa:
                manifest = terminal_dataset.lock_gqa_manifest(
                    args.data_root, lock_path, seed=args.seed, experiment=shared_config, **sampling,
                )
            else:
                manifest = terminal_dataset.lock_manifest(
                    args.data_root, rows_file=args.rows_file, output=lock_path, seed=args.seed,
                    per_difficulty=args.per_difficulty, experiment=shared_config,
                    fields=args.fields, rows_key=args.rows_key, visualprobe_splits=args.visualprobe_split,
                )
            config.update(status="locked", selected_manifest_fingerprint=manifest["fingerprint"])
            write_json(config_file, config)
            print(f"Locked {len(manifest['samples'])} samples at {lock_path}", flush=True)
            return 0
        manifest = terminal_dataset.load_locked_manifest(args.manifest)
        if (
            manifest["dataset"]["root"] != str(args.data_root.resolve())
            or manifest["sampling"]["seed"] != args.seed
            or any(manifest["sampling"].get(key) != value for key, value in sampling.items())
            or manifest.get("experiment") != shared_config
        ):
            raise ValueError("locked dataset, generation, sampling or Mini-o3 identity differs from this run")
        if lock_path.exists() and lock_path.read_bytes() != args.manifest.read_bytes():
            raise ValueError("output directory contains a different locked cohort")
        if not lock_path.exists():
            write_json(lock_path, manifest)
        if (args.output_dir / "samples").exists():
            raise ValueError("experiment output already contains traces; use a new output directory")
    except (OSError, ValueError, RuntimeError) as exc:
        config.update(status="blocked_before_inference", reason=str(exc))
        write_json(config_file, config)
        print(f"Blocked before inference: {exc}", flush=True)
        return 2

    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, StopStringCriteria

    from terminal_runtime import MiniO3Runtime

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        config.update(status="blocked_before_inference", reason="a CUDA device is required for the selected BF16 checkpoint")
        write_json(config_file, config)
        print(config["reason"], flush=True)
        return 2
    torch.cuda.set_device(device)
    config.update({
        "status": "loading",
        "selected_manifest_fingerprint": manifest["fingerprint"],
        "loaded_samples": len(manifest["samples"]),
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)},
    })
    write_json(config_file, config)
    rows, pairs = [], []
    results_status = "complete"
    try:
        tools = CropTools(identity["code_dir"])
        model_start = time.perf_counter()
        processor = AutoProcessor.from_pretrained(identity["checkpoint"], local_files_only=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            identity["checkpoint"], dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
        ).eval().to(device)
        runtime = MiniO3Runtime(
            model, processor, device=device, min_pixels=generation["min_pixels"], max_pixels=generation["max_pixels"],
        )
        config["model_and_processor_load_ms"] = (time.perf_counter() - model_start) * 1000
        start = time.perf_counter()
        # Shared tokenizer setup. Hold no per-sample KV and never run a model
        # forward for warmup outside the locked cohort.
        StopStringCriteria(processor.tokenizer, ["</grounding>", "</answer>"])
        StopStringCriteria(processor.tokenizer, list(STOP_STRINGS))
        config["shared_stop_criteria_setup_ms"] = (time.perf_counter() - start) * 1000
        config["status"] = "running"
        write_json(config_file, config)
    except (OSError, RuntimeError, ValueError) as exc:
        config.update(status="blocked_while_loading", reason=str(exc))
        write_json(config_file, config)
        print(f"Could not initialize Mini-o3: {exc}", flush=True)
        return 2

    if set(args.modes) != set(MODES):
        config["first_sample_gate"] = "checked_for_requested_mode_subset"
    for index, sample in enumerate(manifest["samples"]):
        directory = args.output_dir / "samples" / f"{index:02d}-{sample['sample_key']}"
        autonomous = directory / "autonomous"
        print(f"[{index + 1}/{len(manifest['samples'])}] {sample['difficulty']} {sample['id']}", flush=True)
        path_rows = []
        with torch.inference_mode():
            try:
                original, trace = acquire(
                    runtime, tools, sample, terminal_dataset.load_image(manifest, sample), autonomous, args.seed,
                    generation=generation,
                )
                if original is None:
                    path_rows = list(failed_rows(sample, args.modes, "unverified", trace["terminal_status"]))
                    pairs.append({"sample_key": sample["sample_key"], "status": "pair_invalid",
                                  "reason": "no_verified_pixel_source_terminal_crop"})
                else:
                    path_rows, original_info = path_modes(
                        runtime, original, trace, sample, autonomous, args, identity,
                        directory / "state_banks", parity=index == 0,
                    )
                    replay_dir = directory / "controlled_tool_replay"
                    try:
                        if args.skip_controlled_replay:
                            raise ValueError("controlled replay disabled for this run")
                        alternative, alternate_trace = replay(
                            runtime, tools, original, trace, replay_dir, generation=generation,
                        )
                        alt_rows, alternate_info = path_modes(
                            runtime, alternative, alternate_trace, sample, replay_dir, args, identity,
                            directory / "state_banks", parity=False,
                        )
                        if any(alternate_info[key] != original_info[key] for key in ("state_hash", "evidence_hash")):
                            raise ValueError("controlled replay produced a different canonical state")
                    except (ValueError, RuntimeError) as exc:
                        pairs.append({"sample_key": sample["sample_key"],
                                      "status": "not_requested" if args.skip_controlled_replay else "pair_invalid",
                                      "reason": str(exc)})
                    else:
                        path_rows.extend(alt_rows)
                        existing = {(row["mode"], row["request"]): row for row in path_rows if row["path"] == "autonomous"}
                        pairs.append({
                            "sample_key": sample["sample_key"], "status": "paired", "state_hash": original_info["state_hash"],
                            "comparisons": [
                                {"mode": row["mode"], "request": row["request"],
                                 "both_answered": row["answer"] is not None and existing[(row["mode"], row["request"])]["answer"] is not None,
                                 "same_answer_text": row["answer"] is not None and row["answer"] == existing[(row["mode"], row["request"])]["answer"]}
                                for row in alt_rows
                            ],
                        })
            except (OSError, RuntimeError, ValueError) as exc:
                path_rows.extend(failed_rows(sample, args.modes, "error", str(exc)))
                if not any(row["sample_key"] == sample["sample_key"] for row in pairs):
                    pairs.append({"sample_key": sample["sample_key"], "status": "pair_invalid", "reason": str(exc)})
                write_json(directory / "error.json", {"reason": str(exc), "type": type(exc).__name__})
                print(f"Sample failed: {exc}", flush=True)
        rows.extend(path_rows)
        print(
            f"  {sum(bool(row.get('generation')) for row in path_rows)} requests executed; "
            f"{sum(row.get('matches_reference') is True for row in path_rows)} exact matches",
            flush=True,
        )
        if index == 0 and not gqa and not args.trace_preview and not all(
            any(row["mode"] == mode and row.get("generation") and row["path"] == "autonomous" for row in path_rows)
            for mode in args.modes
        ):
            results_status = "first_locked_sample_did_not_complete_required_checks"
            for pending in manifest["samples"][1:]:
                rows.extend(failed_rows(pending, args.modes, "not_run", results_status))
            dump_results(args.output_dir, rows, pairs, results_status)
            break
        dump_results(args.output_dir, rows, pairs, "in_progress")

    dump_results(args.output_dir, rows, pairs, results_status)
    config["status"] = results_status
    write_json(config_file, config)
    return 0 if results_status == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
