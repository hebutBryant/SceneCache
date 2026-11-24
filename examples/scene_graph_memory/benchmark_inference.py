"""Benchmark seven isolated visual KV modes on matched image-first questions."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import random
import signal
import socket
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import requests
from benchmark_trace import (
    DISABLE_DECODER_CACHE_ENV,
    DISABLE_VISION_CACHE_ENV,
    PROFILE_RANGE,
)
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODES = (
    "cold_full",
    "feature_only",
    "legacy_full_kv",
    "pipelined_full_kv",
    "region_reference",
    "region_pipeline",
    "region_pipeline_triton",
)
LEGACY = MODES[:3]
REGION = MODES[4:]
PRIME = "这张图片是什么场景？只回答场景名称。"
SCHEDULE_FIELDS = (
    "ttft_ms",
    "total_ms",
    "prefill_ms",
    "decode_ms",
    "decode_tail_ms",
    "tpot_ms",
    "decode_step_ms",
)
OPTIMIZED_COUNTERS = (
    "n_all",
    "n_active",
    "n_recompute",
    "layers",
    "h2d_bytes",
    "h2d_index_bytes",
    "copy_calls",
    "pinned_bytes",
    "gpu_staging_bytes",
    "source_cpu_bytes",
    "gpu_kv_bytes",
    "sync_calls",
)
GPU_PEAK_FIELDS = (
    "process_gpu_peak_allocated_bytes",
    "process_gpu_peak_reserved_bytes",
)
COMPACT_FIELDS = (
    "prefill_kv_length",
    "first_decode_kv_length",
    "last_decode_kv_length",
    "decode_h2d_bytes",
)


def dump(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def replace_json(path, value):
    temp = path.with_suffix(".tmp")
    dump(temp, value)
    temp.replace(path)


def normalize_answer(answer):
    return "".join(
        char
        for char in unicodedata.normalize("NFKC", answer).casefold()
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merge", type=Path, nargs="+", help="Merge previously completed timed runs"
    )
    parser.add_argument("--memory-dir", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New directory for samples, logs and summary",
    )
    parser.add_argument(
        "--model-path", type=Path, help="Default: model recorded in the memory manifest"
    )
    parser.add_argument("--cases", type=Path, default=HERE / "market_qa.json")
    parser.add_argument(
        "--diagnostic-cases",
        type=Path,
        nargs="+",
        default=[],
        help="Additional variable-length QA case lists to evaluate once in the warmed server",
    )
    parser.add_argument(
        "--verify-after",
        type=Path,
        help="Save final reference and optimized tensor probes separately from timed samples",
    )
    parser.add_argument(
        "--check-cancellation",
        choices=REGION,
        help="Run an SSE disconnect and changed-selection check on the warmed region server",
    )
    parser.add_argument(
        "--gpu", help="Physical GPU index or UUID visible to the server"
    )
    parser.add_argument("--port", type=int, default=30008)
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument(
        "--interleave-full-region",
        action="store_true",
        help="Compare pipelined_full_kv and region_pipeline in one isolated worker",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Allow reduced counts for diagnostic smoke runs",
    )
    parser.add_argument(
        "--nsys",
        type=Path,
        help="Nsight Systems executable. Capture one post-warmup prefill per mode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--fixed-output-tokens",
        type=int,
        help="Exact generated length for performance-only runs",
    )
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--mem-fraction-static", type=float, default=0.97)
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8")),
    )
    parser.add_argument("--active-budget", type=float)
    args = parser.parse_args()
    if not args.merge and (args.memory_dir is None or args.gpu is None):
        parser.error("running the benchmark requires --memory-dir and --gpu")
    if args.nsys is not None and (
        not args.smoke or args.rounds != 1 or not args.nsys.is_file()
    ):
        parser.error("--nsys requires a local executable, --smoke, and --rounds=1")
    if args.smoke:
        if args.warmup < 1 or args.rounds < 1:
            parser.error(
                "smoke runs require at least one warmup and one sample per mode"
            )
    elif args.warmup < 5 or args.rounds < 20:
        parser.error(
            "formal runs require at least five warmups and twenty samples per mode"
        )
    if args.fixed_output_tokens is not None and args.fixed_output_tokens < 2:
        parser.error("--fixed-output-tokens must be at least two")
    if args.interleave_full_region and set(args.modes) != {
        "pipelined_full_kv", "region_pipeline"
    }:
        parser.error("--interleave-full-region requires pipelined_full_kv and region_pipeline")
    if (
        args.check_cancellation is not None
        and args.check_cancellation not in args.modes
    ):
        parser.error("--check-cancellation must name a benchmarked region mode")
    return args


def load_cases(path, manifest, need_regions):
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("The cases file must contain a non-empty list")
    ids = set()
    ids_in_memory = {row["region_id"] for row in manifest["regions"]}
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("id"), str)
            or not case["id"]
            or case["id"] in ids
        ):
            raise ValueError("Case IDs must be nonempty and unique")
        ids.add(case["id"])
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"Case {case['id']} has no question")
        accepted = case.get("accepted")
        if (
            not isinstance(accepted, list)
            or not accepted
            or any(
                not isinstance(answer, str) or not normalize_answer(answer)
                for answer in accepted
            )
        ):
            raise ValueError(
                f"Case {case['id']} requires exact normalized accepted answers"
            )
        regions = case.get("region_ids")
        if need_regions and (
            not isinstance(regions, list)
            or not regions
            or not set(regions) <= ids_in_memory
        ):
            raise ValueError(
                f"Case {case['id']} selects missing regions in this memory"
            )
    return cases


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = math.floor(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def percentiles(values):
    return dict(
        p50=percentile(values, 0.5),
        p95=percentile(values, 0.95),
        min=min(values),
        max=max(values),
        stddev=statistics.pstdev(values),
    )


def summarize(samples, selected_modes):
    summary = {}
    for mode in selected_modes:
        rows = [
            row for row in samples if row["condition"] == mode and not row["warmup"]
        ]
        if not rows:
            continue
        graded = [row["correct"] for row in rows if row["correct"] is not None]
        summary[mode] = {
            "n": len(rows),
            "correct": sum(graded) if graded else None,
            "accuracy": statistics.mean(graded) if graded else None,
            "answers": dict(Counter(row["answer"] for row in rows)),
            "latency_ms": {
                name: percentiles(
                    [row[name] for row in rows if row.get(name) is not None]
                )
                for name in SCHEDULE_FIELDS
                if any(row.get(name) is not None for row in rows)
            },
            "counter_medians": {
                name: statistics.median(
                    row[name] for row in rows if row[name] is not None
                )
                for name in (
                    "output_tokens",
                    "decode_steps",
                    "visual_tokens",
                    "active_visual_tokens",
                    "recomputed_visual_tokens",
                    "legacy_kv_reads",
                    "legacy_kv_read_bytes",
                    "legacy_kv_writes",
                    "legacy_kv_write_bytes",
                    "vision_cpu_reads",
                    "vision_cpu_writes",
                    *OPTIMIZED_COUNTERS,
                    *GPU_PEAK_FIELDS,
                    *COMPACT_FIELDS,
                )
                if all(row.get(name) is not None for row in rows)
            },
            "runtime_metric_medians": {},
            "by_case": {},
        }
        for case_id in sorted({row["case_id"] for row in rows}):
            subset = [row for row in rows if row["case_id"] == case_id]
            graded = [row["correct"] for row in subset if row["correct"] is not None]
            summary[mode]["by_case"][case_id] = {
                "n": len(subset),
                "correct": sum(graded) if graded else None,
                "answers": dict(Counter(row["answer"] for row in subset)),
                "latency_ms": {
                    name: percentiles(
                        [row[name] for row in subset if row.get(name) is not None]
                    )
                    for name in SCHEDULE_FIELDS
                    if any(row.get(name) is not None for row in subset)
                },
                "active_visual_tokens": statistics.median(
                    row["active_visual_tokens"]
                    for row in subset
                    if row.get("active_visual_tokens") is not None
                )
                if any(row.get("active_visual_tokens") is not None for row in subset)
                else None,
            }
        counters = {}
        for sample in rows:
            for event in sample["runtime_metrics"]:
                stage = (
                    event.get("stage") or event.get("event") or event.get("mode", "")
                )
                transfer = event.get("transfer") or {}
                fields = {
                    **event,
                    **{f"transfer.{name}": value for name, value in transfer.items()},
                }
                for name, value in fields.items():
                    if (
                        name != "time_ns"
                        and isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    ):
                        counters.setdefault(
                            f"{stage}.{name}" if stage else name, []
                        ).append(value)
        for name, values in sorted(counters.items()):
            summary[mode]["runtime_metric_medians"][name] = statistics.median(values)
    return summary


def paired_against_full(samples):
    formal = [
        row for row in samples if not row["warmup"] and row["correct"] is not None
    ]
    full = {
        (row["round"], row["case_id"]): row
        for row in formal
        if row["condition"] == "cold_full"
    }
    paired = {}
    for mode in MODES[1:]:
        pairs = [
            (full[key], row)
            for row in formal
            if row["condition"] == mode
            and (key := (row["round"], row["case_id"])) in full
        ]
        if pairs:
            paired[mode] = {
                "n": len(pairs),
                "both_correct": sum(
                    base["correct"] and row["correct"] for base, row in pairs
                ),
                "both_incorrect": sum(
                    not base["correct"] and not row["correct"] for base, row in pairs
                ),
                "full_only_correct": sum(
                    base["correct"] and not row["correct"] for base, row in pairs
                ),
                "mode_only_correct": sum(
                    not base["correct"] and row["correct"] for base, row in pairs
                ),
                "exact_same_answer": sum(
                    base["answer"] == row["answer"] for base, row in pairs
                ),
            }
    return paired


def paired_performance(samples):
    formal = {
        (row["condition"], row["round"], row["case_id"]): row
        for row in samples
        if not row["warmup"]
    }
    comparisons = (
        ("cold_full", "feature_only"),
        ("feature_only", "legacy_full_kv"),
        ("legacy_full_kv", "pipelined_full_kv"),
        ("pipelined_full_kv", "region_pipeline"),
        ("region_reference", "region_pipeline"),
        ("region_pipeline", "region_pipeline_triton"),
    )
    results = []
    for reference, candidate in comparisons:
        paired = [
            (start, formal[candidate, round_index, case_id])
            for (mode, round_index, case_id), start in formal.items()
            if mode == reference and (candidate, round_index, case_id) in formal
        ]
        if not paired:
            continue
        values = {}
        for name in SCHEDULE_FIELDS:
            ratios = [
                b[name] / a[name]
                for a, b in paired
                if a.get(name) and b.get(name) is not None
            ]
            if ratios:
                values[name] = percentiles(ratios)
        results.append(
            {
                "reference": reference,
                "candidate": candidate,
                "n": len(paired),
                "candidate_over_reference_latency": values,
            }
        )
    return results


class TraceCursor:
    def __init__(self, filename):
        self.filename = filename
        self.offset = 0

    def read(self):
        if not self.filename.is_file():
            return []
        with self.filename.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read()
        last_line = data.rfind(b"\n") + 1
        self.offset += last_line
        return [
            json.loads(line) for line in data[:last_line].splitlines() if line.strip()
        ]


@contextmanager
def launch_server(args, output, model, mode, capture_count=1):
    output.mkdir()
    bootstrap = output / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "sitecustomize.py").write_text(
        "from benchmark_trace import install\ninstall()\n"
    )
    env = os.environ.copy()
    for name in (
        "SGLANG_SCENE_GRAPH_CAPTURE_DIR",
        "SCENECACHE_BENCHMARK_MIXED",
        "SGLANG_VISUAL_KV_MODE",
        DISABLE_DECODER_CACHE_ENV,
        DISABLE_VISION_CACHE_ENV,
        "SGLANG_VISUAL_KV_MEMORY",
        "SGLANG_VISUAL_KV_SELECTION",
        "SGLANG_VISUAL_KV_METRICS",
    ):
        env.pop(name, None)
    env.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_HOME=str(args.cuda_home),
        CUDACXX=str(args.cuda_home / "bin/nvcc"),
        CUDA_TOOLKIT_ROOT_DIR=str(args.cuda_home),
        PATH=os.pathsep.join(
            [str(args.cuda_home / "bin"), str(Path(sys.executable).parent), env["PATH"]]
        ),
        LD_LIBRARY_PATH=os.pathsep.join(
            filter(None, [str(args.cuda_home / "lib64"), env.get("LD_LIBRARY_PATH")])
        ),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4",
        SGLANG_CPU_WORKERS="2",
        SGLANG_VLM_CACHE_SIZE_MB="128",
        MAX_JOBS="4",
        SCENECACHE_REPORT_DIR=str(output),
        PYTHONPATH=os.pathsep.join(
            str(path) for path in (bootstrap, HERE, ROOT / "scripts", ROOT / "python")
        ),
        FLASHINFER_WORKSPACE_BASE=str(ROOT / ".runtime/flashinfer"),
        TRITON_CACHE_DIR=str(ROOT / ".runtime/triton"),
    )
    if mode not in LEGACY:
        env["SGLANG_VISUAL_KV_MODE"] = mode
        env["SGLANG_VISUAL_KV_METRICS"] = str(output / "runtime_metrics.jsonl")
    if mode in REGION:
        env["SGLANG_VISUAL_KV_MEMORY"] = str(args.memory_dir.resolve())
        env["SGLANG_VISUAL_KV_SELECTION"] = str(output / "selection.json")
    if getattr(args, "interleave_full_region", False):
        env["SCENECACHE_BENCHMARK_MIXED"] = "1"
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model),
        "--served-model-name",
        model.name,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "flashinfer",
        "--mm-attention-backend",
        "triton_attn",
        "--sampling-backend",
        "pytorch",
        "--context-length",
        str(args.context_length),
        "--max-total-tokens",
        str(args.context_length + 128),
        "--max-running-requests",
        "1",
        "--chunked-prefill-size",
        "-1",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--random-seed",
        str(args.seed),
    ]
    server_pid_file = output / "server.pid"
    if args.nsys is not None:
        env.update(SGLANG_VISUAL_KV_PROFILE="1", NSYS_NVTX_PROFILER_REGISTER_ONLY="0")
        launcher = (
            "import os, sys; from pathlib import Path; "
            "Path(sys.argv[1]).write_text(str(os.getpid()) + '\\n'); "
            "os.execv(sys.executable, [sys.executable, *sys.argv[2:]])"
        )
        range_end = f"repeat:{capture_count}" if capture_count > 1 else "stop"
        command = [
            str(args.nsys.resolve()),
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            "--capture-range=nvtx",
            f"--nvtx-capture={PROFILE_RANGE}",
            f"--capture-range-end={range_end}",
            f"--output={output / 'prefill_profile'}",
            sys.executable,
            "-c",
            launcher,
            str(server_pid_file),
            *command[1:],
        ]
    file = output / "server.log"
    with file.open("w") as log, requests.Session() as session:
        session.trust_env = False
        base = f"http://127.0.0.1:{args.port}"
        sock = socket.socket()
        try:
            if sock.connect_ex(("127.0.0.1", args.port)) == 0:
                raise RuntimeError(f"Port {args.port} is occupied")
        finally:
            sock.close()
        server = subprocess.Popen(
            command,
            env=env,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        if args.nsys is None:
            server_pid_file.write_text(f"{server.pid}\n")
        print(
            f"Started {mode} server PID={server.pid}, physical GPU {args.gpu}",
            flush=True,
        )
        try:
            deadline = time.monotonic() + 360
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}; see {file}")
                try:
                    if session.get(base + "/health", timeout=2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError(f"Server did not become ready; see {file}")
            yield session, base, command
        finally:
            if server.poll() is None:
                try:
                    os.kill(int(server_pid_file.read_text()), signal.SIGTERM)
                except (FileNotFoundError, ProcessLookupError):
                    server.terminate()
                try:
                    server.wait(timeout=30 if args.nsys is not None else 12)
                except subprocess.TimeoutExpired:
                    pass
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            server.wait()


def ask(
    session,
    base,
    image,
    model_name,
    question,
    max_tokens,
    request_id,
    fixed_tokens=None,
):
    started_ns = time.monotonic_ns()
    first_ns = None
    last_ns = None
    usage = None
    chunks = []
    finish_reason = None
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": [image, {"type": "text", "text": question}]}
        ],
        "rid": request_id,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if fixed_tokens is not None:
        payload.update(
            max_tokens=fixed_tokens, min_tokens=fixed_tokens, ignore_eos=True
        )
    with session.post(
        base + "/v1/chat/completions", json=payload, stream=True, timeout=180
    ) as response:
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        for line in response.iter_lines(chunk_size=128):
            if not line.startswith(b"data: "):
                continue
            if line[6:] == b"[DONE]":
                break
            item = json.loads(line[6:])
            usage = item.get("usage") or usage
            for choice in item.get("choices", []):
                chunk = choice.get("delta", {}).get("content")
                if chunk:
                    last_ns = time.monotonic_ns()
                    if first_ns is None:
                        first_ns = last_ns
                    chunks.append(chunk)
                finish_reason = choice.get("finish_reason") or finish_reason
    finish_ns = time.monotonic_ns()
    if usage is None or not finish_reason:
        raise RuntimeError(f"Incomplete completion: {question!r} {chunks}")
    output_tokens = usage["completion_tokens"]
    if fixed_tokens is not None and output_tokens != fixed_tokens:
        raise RuntimeError(
            f"Expected {fixed_tokens} generated tokens, got {output_tokens} ({finish_reason})"
        )
    return {
        "answer": "".join(chunks),
        "finish_reason": finish_reason,
        "start_ns": started_ns,
        "first_text_ns": first_ns,
        "last_text_ns": last_ns,
        "finish_ns": finish_ns,
        "ttft_ms": (first_ns - started_ns) / 1e6 if first_ns is not None else None,
        "total_ms": (finish_ns - started_ns) / 1e6,
        "decode_tail_ms": (finish_ns - first_ns) / 1e6
        if first_ns is not None
        else None,
        "tpot_ms": (last_ns - first_ns) / (1e6 * (output_tokens - 1))
        if first_ns is not None and output_tokens > 1
        else None,
        "input_tokens": usage["prompt_tokens"],
        "output_tokens": output_tokens,
    }


def selection_for(case, args):
    selection = {
        "region_ids": case["region_ids"],
        "expand_hops": case.get("expand_hops", 0),
    }
    budget = case.get("active_budget", args.active_budget)
    if budget is not None:
        selection["active_budget"] = budget
    return selection


def observations(request_id, traces, timings, runtime, manifest, mode):
    events = [event for event in traces.read() if event["request_id"] == request_id]
    batches = [event for event in timings.read() if event["request_id"] == request_id]
    runtime_events = runtime.read()
    prefill = [event for event in events if event["event"] == "prefill"]
    if len(prefill) != 1:
        raise RuntimeError(
            f"Expected a single image prefill for {request_id}, saw {len(prefill)}"
        )
    event = prefill[0]
    if event["image_grids"] != [manifest["geometry"]["grid_thw"]]:
        raise RuntimeError(
            f"Runtime image grid changed from the persisted bank: {event['image_grids']}"
        )
    ((start, end),) = event["image_ranges"]
    active_tokens = end - start + 1
    image_tokens = len(manifest["tokens"])
    mask = event["first_layer_compute_mask"]
    recomputed = active_tokens if mask is None else sum(mask[start : end + 1])
    hits = sum(e["event"] == "embedding_cache_get" and e["hit"] is True for e in events)
    reads = [e for e in events if e["event"] == "get_kv" and e["kind"] == "kv"]
    writes = [e for e in events if e["event"] == "write_kv" and e["kind"] == "kv"]
    vision_reads = sum(e["event"] == "get_kv" and e["kind"] == "vision" for e in events)
    vision_writes = sum(
        e["event"] == "write_kv" and e["kind"] == "vision" for e in events
    )
    if mode == "cold_full" and (vision_reads or vision_writes):
        raise RuntimeError(
            f"Unexpected CPU visual feature bank activity for {request_id}: {vision_reads} reads, {vision_writes} writes"
        )
    stages = {
        stage: [batch for batch in batches if batch["stage"] == stage]
        for stage in ("prefill", "decode")
    }
    if len(stages["prefill"]) != 1:
        raise RuntimeError(f"No unique scheduler prefill stage for {request_id}")
    expected = {
        "cold_full": (0, 0, 0),
        "feature_only": (1, 0, 0),
        "legacy_full_kv": (1, 2 * len(manifest["source_bank"]["layers"]), 0),
    }
    if mode in expected and (hits, len(reads), len(writes)) != expected[mode]:
        raise RuntimeError(
            f"Cache controls for {request_id} did not hold: {hits} feature hits, {len(reads)} KV reads, {len(writes)} KV writes"
        )
    decode_ms = sum(b["finish_ns"] - b["start_ns"] for b in stages["decode"]) / 1e6
    decode_steps = len(stages["decode"])
    optimized_counters = {}
    if mode not in LEGACY:
        candidates = [
            row
            for row in runtime_events
            if row.get("event", row.get("stage")) == "optimized_prefill"
        ]
        if len(candidates) != 1 or candidates[0].get("mode") != mode:
            raise RuntimeError(
                f"Missing optimized {mode} prefill telemetry for {request_id}: {runtime_events}"
            )
        raw = candidates[0]
        if (raw["n_all"], raw["n_active"]) != (image_tokens, active_tokens):
            raise RuntimeError(
                f"Invalid visual grid accounting for {request_id}: {raw}"
            )
        recomputed = raw["n_recompute"]
        optimized_counters = {name: raw.get(name) for name in OPTIMIZED_COUNTERS}
        decode = [
            row for row in runtime_events if row.get("event") == "optimized_decode"
        ]
        optimized_counters["prefill_kv_length"] = raw["kv_length"]
        scheduler_lengths = [row.get("kv_length") for row in stages["decode"]]
        if scheduler_lengths and all(length is not None for length in scheduler_lengths):
            expected_lengths = list(
                range(raw["kv_length"] + 1, raw["kv_length"] + decode_steps + 1)
            )
            if scheduler_lengths != expected_lengths:
                raise RuntimeError(f"Unexpected scheduler KV lengths for {request_id}: {scheduler_lengths}")
            optimized_counters.update(
                first_decode_kv_length=scheduler_lengths[0],
                last_decode_kv_length=scheduler_lengths[-1],
            )
        if mode == "pipelined_full_kv" and active_tokens != image_tokens:
            raise RuntimeError(f"Full-KV mode was compacted for {request_id}")
        if mode in REGION:
            expected_lengths = range(
                raw["kv_length"] + 1, raw["kv_length"] + decode_steps + 1
            )
            if [row["kv_length"] for row in decode] != list(expected_lengths) or any(
                (row["n_active"], row["h2d_bytes"], row["mode"])
                != (active_tokens, 0, mode)
                for row in decode
            ):
                raise RuntimeError(
                    f"Compact decode telemetry does not match prefill for {request_id}: {decode}"
                )
            optimized_counters.update(
                first_decode_kv_length=expected_lengths[0] if decode else None,
                last_decode_kv_length=expected_lengths[-1] if decode else None,
                decode_h2d_bytes=sum(row["h2d_bytes"] for row in decode),
            )
    # Older trace records can be read without implying allocator values were captured.
    gpu_peaks = {
        name: max(b[name] for b in batches)
        for name in GPU_PEAK_FIELDS
        if all(name in b for b in batches)
    }
    return {
        "embedding_hits": hits,
        "legacy_kv_reads": len(reads),
        "legacy_kv_read_bytes": sum(r["bytes"] for r in reads),
        "legacy_kv_writes": len(writes),
        "legacy_kv_write_bytes": sum(r["bytes"] for r in writes),
        "vision_cpu_reads": vision_reads,
        "vision_cpu_writes": vision_writes,
        "visual_tokens": image_tokens,
        "active_visual_tokens": active_tokens,
        "recomputed_visual_tokens": recomputed,
        "prefill_ms": sum(b["finish_ns"] - b["start_ns"] for b in stages["prefill"])
        / 1e6,
        "decode_ms": decode_ms,
        "decode_steps": decode_steps,
        "decode_step_ms": decode_ms / decode_steps if decode_steps else None,
        "scheduler_batches": batches,
        "runtime_metrics": runtime_events,
        **optimized_counters,
        **gpu_peaks,
    }


def run_group(
    args,
    metadata,
    group,
    cases,
    diagnostics,
    manifest,
    model,
    image,
    write,
    write_diagnostic,
):
    mode = "pipelined_full_kv" if "pipelined_full_kv" in group else group[0]
    if args.interleave_full_region:
        mode, name = "region_pipeline", "mixed_group"
    elif len(group) == 1:
        name = mode
    elif mode in REGION:
        name = "region_group"
    else:
        name = "baseline" if mode in LEGACY else "full_group"
    root = args.out / name
    with launch_server(args, root, model, mode, capture_count=len(group)) as (
        session,
        base,
        command,
    ):
        metadata["server_commands"][name] = command
        traces = TraceCursor(root / "trace.jsonl")
        timings = TraceCursor(root / "scheduler_timing.jsonl")
        runtime = TraceCursor(root / "runtime_metrics.jsonl")
        rng = random.Random(args.seed)
        cases_by_round = list(cases)
        current_selection = None
        full_kv_primed = False

        def run_case(index, case, fixed_tokens, diagnostic_suite=None, verify=False):
            nonlocal current_selection, full_kv_primed
            selection = selection_for(case, args) if mode in REGION else None
            if selection != current_selection:
                replace_json(root / "selection.json", selection)
                current_selection = selection
            order = list(group)
            if len(order) > 1:
                offset = index % len(order)
                order = order[offset:] + order[:offset]
            for condition in order:
                probe_path = None
                if verify:
                    if condition in LEGACY[:2]:
                        continue
                    probe_path = args.verify_after / f"{condition}.pt"
                    if probe_path.exists():
                        raise FileExistsError(
                            f"Verification probe already exists: {probe_path}"
                        )
                phase = f"d_{diagnostic_suite}" if diagnostic_suite is not None else "r"
                request_id = f"{phase}{index:04d}_{condition}_{case['id']}"
                if condition in ("feature_only", "legacy_full_kv", "pipelined_full_kv") and (
                    not args.interleave_full_region or not full_kv_primed
                ):
                    replace_json(
                        root / "control.json",
                        {
                            "request_id": request_id + "_prime",
                            "reset": True,
                            "bench_mode": "legacy",
                        },
                    )
                    ask(
                        session,
                        base,
                        image,
                        model.name,
                        PRIME,
                        args.max_new_tokens,
                        request_id + "_prime",
                    )
                    traces.read()
                    timings.read()
                    runtime.read()
                    full_kv_primed = True
                replace_json(
                    root / "control.json",
                    {
                        "request_id": request_id,
                        "condition": condition,
                        "reset": condition in ("cold_full", "feature_only"),
                        "disable_decoder_cache": condition
                        in ("cold_full", "feature_only"),
                        "disable_vision_cache": condition == "cold_full",
                        "keep_embedding": condition == "feature_only",
                        "capture_profile": args.nsys is not None
                        and diagnostic_suite is None
                        and index == args.warmup,
                        "bench_mode": "legacy" if condition in LEGACY else condition,
                        "verify_dir": str(probe_path.parent)
                        if probe_path is not None
                        else None,
                    },
                )
                answer = ask(
                    session,
                    base,
                    image,
                    model.name,
                    case["question"],
                    args.max_new_tokens,
                    request_id,
                    fixed_tokens=fixed_tokens,
                )
                measured = observations(
                    request_id, traces, timings, runtime, manifest, condition
                )
                if probe_path is not None and not probe_path.is_file():
                    raise RuntimeError(
                        f"Missing verification probe for {request_id}: {probe_path}"
                    )
                if (
                    fixed_tokens is not None
                    and measured["decode_steps"] != fixed_tokens - 1
                ):
                    raise RuntimeError(
                        f"Fixed-length decode did not execute one step per continuation token: {measured['decode_steps']}"
                    )
                row = {
                    "request_id": request_id,
                    "case_id": case["id"],
                    "condition": condition,
                    "round": index,
                    "warmup": diagnostic_suite is None and index < args.warmup,
                    "question": case["question"],
                    "accepted": case["accepted"],
                    "correct": None
                    if fixed_tokens is not None
                    else normalize_answer(answer["answer"])
                    in map(normalize_answer, case["accepted"]),
                    "selection": selection if condition in REGION else None,
                    **answer,
                    **measured,
                }
                if diagnostic_suite is None:
                    write(row)
                    phase_name = "warmup" if row["warmup"] else "sample"
                else:
                    row["diagnostic_suite"] = diagnostic_suite
                    if probe_path is not None:
                        row["verification_probe"] = str(probe_path)
                    write_diagnostic(row)
                    phase_name = "diagnostic"
                print(
                    f"{request_id} {phase_name} {row['total_ms']:.1f} ms {row['answer']!r}",
                    flush=True,
                )

        for index in range(args.warmup + args.rounds):
            phase_index = index if index < args.warmup else index - args.warmup
            if phase_index % len(cases_by_round) == 0:
                rng.shuffle(cases_by_round)
            run_case(
                index,
                cases_by_round[phase_index % len(cases_by_round)],
                args.fixed_output_tokens,
            )
        for suite, suite_cases in diagnostics.items():
            for index, case in enumerate(suite_cases):
                run_case(index, case, None, diagnostic_suite=suite)
        if args.verify_after is not None:
            run_case(
                0, cases[0], None, diagnostic_suite="tensor_verification", verify=True
            )
        if args.check_cancellation in group:
            result_path = args.out / "cancellation.json"
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "benchmark_cancellation.py"),
                    "--base",
                    base,
                    "--selection",
                    str(root / "selection.json"),
                    "--memory-dir",
                    str(args.memory_dir),
                    "--model",
                    model.name,
                    "--mode",
                    args.check_cancellation,
                    "--out",
                    str(result_path),
                ],
                check=True,
            )
            metadata["cancellation_result"] = str(result_path)


@contextmanager
def output_writer(output, samples, name="samples"):
    csv_fields = [
        "request_id",
        "case_id",
        "condition",
        "round",
        "warmup",
        "correct",
        "answer",
        "finish_reason",
        "input_tokens",
        "output_tokens",
        "decode_steps",
        *SCHEDULE_FIELDS,
        "embedding_hits",
        "visual_tokens",
        "active_visual_tokens",
        "recomputed_visual_tokens",
        "legacy_kv_reads",
        "legacy_kv_read_bytes",
        "legacy_kv_writes",
        "legacy_kv_write_bytes",
        "vision_cpu_reads",
        "vision_cpu_writes",
        "start_ns",
        "first_text_ns",
        "last_text_ns",
        "finish_ns",
        *OPTIMIZED_COUNTERS,
        *GPU_PEAK_FIELDS,
        *COMPACT_FIELDS,
        "diagnostic_suite",
        "verification_probe",
    ]
    with ExitStack() as stack:
        jsonl = stack.enter_context(
            (output / f"{name}.jsonl").open("w", encoding="utf-8")
        )
        csv_file = stack.enter_context(
            (output / f"{name}.csv").open("w", encoding="utf-8", newline="")
        )
        csv_writer = csv.DictWriter(
            csv_file, fieldnames=csv_fields, extrasaction="ignore"
        )
        csv_writer.writeheader()

        def write(row):
            samples.append(row)
            jsonl.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            jsonl.flush()
            csv_writer.writerow(row)
            csv_file.flush()

        yield write


def write_summary(output, metadata, samples, diagnostic_samples=()):
    diagnostics = {}
    for suite in sorted({row["diagnostic_suite"] for row in diagnostic_samples}):
        rows = [row for row in diagnostic_samples if row["diagnostic_suite"] == suite]
        diagnostics[suite] = {
            "results": summarize(rows, metadata["modes"]),
            "matched_by_case_and_round_vs_cold_full": paired_against_full(rows),
        }
    dump(
        output / "summary.json",
        {
            **metadata,
            "results": summarize(samples, metadata["modes"]),
            "matched_by_case_and_round_vs_cold_full": paired_against_full(samples),
            "matched_latency_by_case_and_round": paired_performance(samples),
            "diagnostics": diagnostics,
        },
    )


def merge_runs(paths, output):
    runs = [json.loads((path / "summary.json").read_text()) for path in paths]
    first = runs[0]
    settings = (
        "cases",
        "model",
        "gpu",
        "request_order",
        "temperature",
        "max_output_tokens",
        "memory_fingerprint",
        "warmup_per_mode",
        "formal_per_mode",
        "seed",
        "prefill_decode_clock",
        "ttft_clock",
        "fixed_output_tokens",
        "decoder_cache_bypass",
        "versions",
        "profiling",
        "verification_mode",
    )
    if any(
        any(run.get(field) != first.get(field) for field in settings)
        for run in runs[1:]
    ):
        raise ValueError("Benchmark runs have unmatched cases, settings or hardware")
    seen = set()
    rows = []
    diagnostic_rows = []
    diagnostic_suites = {}
    for run, path in zip(runs, paths):
        modes = run["modes"]
        counts = Counter()
        if seen.intersection(modes):
            raise ValueError("A benchmark mode appears in multiple runs")
        seen.update(modes)
        for line in (path / "samples.jsonl").read_text().splitlines():
            row = json.loads(line)
            counts[row["condition"], row["warmup"]] += 1
            rows.append(row)
        expected = {
            (mode, warm): run["warmup_per_mode"] if warm else run["formal_per_mode"]
            for mode in modes
            for warm in (True, False)
        }
        if counts != expected:
            raise ValueError(f"Incomplete benchmark samples in {path}")
        for name, cases in run.get("diagnostic_suites", {}).items():
            if diagnostic_suites.setdefault(name, cases) != cases:
                raise ValueError(f"Diagnostic case suite {name} changed between runs")
        diagnostic_file = path / "diagnostic_samples.jsonl"
        if diagnostic_file.is_file():
            diagnostic_rows.extend(
                map(json.loads, diagnostic_file.read_text().splitlines())
            )
    metadata = {
        **first,
        "modes": [mode for mode in MODES if mode in seen],
        "merged_runs": [str(path.resolve()) for path in paths],
        "server_commands": {
            key: value for run in runs for key, value in run["server_commands"].items()
        },
        "diagnostic_suites": diagnostic_suites,
        "vision_cache_bypass": sorted(
            {
                mode
                for run in runs
                for mode in run.get("vision_cache_bypass", [])
                if mode in run["modes"]
            }
        ),
        "verification_after_directories": [
            run["verification_after_directory"]
            for run in runs
            if run.get("verification_after_directory")
        ],
        "cancellation_results": [
            run["cancellation_result"] for run in runs if run.get("cancellation_result")
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    with output_writer(output, samples) as write:
        for row in rows:
            write(row)
    diagnostics = []
    if diagnostic_rows:
        with output_writer(output, diagnostics, "diagnostic_samples") as write:
            for row in diagnostic_rows:
                write(row)
    write_summary(output, metadata, samples, diagnostics)


def main():
    args = arguments()
    args.out = args.out.resolve()
    if args.merge:
        merge_runs(args.merge, args.out)
        return
    if len(set(args.modes)) != len(args.modes):
        raise ValueError("Modes must not be duplicated")
    manifest = json.loads((args.memory_dir / "manifest.json").read_text())
    model = (args.model_path or Path(manifest["fingerprints"]["model_path"])).absolute()
    need_regions = any(mode in REGION for mode in args.modes)
    cases = load_cases(args.cases, manifest, need_regions)
    diagnostics = {
        path.stem: load_cases(path, manifest, need_regions)
        for path in args.diagnostic_cases
    }
    if len(diagnostics) != len(args.diagnostic_cases):
        raise ValueError("Diagnostic case files must have distinct names")
    if args.verify_after is not None:
        if "tensor_verification" in diagnostics:
            raise ValueError(
                "The diagnostic suite name tensor_verification is reserved for --verify-after"
            )
        args.verify_after = args.verify_after.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    submitted = args.out / "submitted.png"
    with Image.open(args.memory_dir / "image.png") as photo:
        photo = photo.convert("RGB")
        max_edge = manifest["config"].get("max_image_edge")
        if max_edge is not None:
            photo.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        photo.save(submitted)
    image = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,"
            + base64.b64encode(submitted.read_bytes()).decode()
        },
    }
    metadata = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "gpu": args.gpu,
        "request_order": "image then text",
        "interleave_full_region": args.interleave_full_region,
        "temperature": 0,
        "prime_question": PRIME,
        "max_output_tokens": args.fixed_output_tokens or args.max_new_tokens,
        "fixed_output_tokens": args.fixed_output_tokens,
        "memory_fingerprint": manifest["fingerprint"],
        "model": str(model),
        "modes": args.modes,
        "warmup_per_mode": args.warmup,
        "formal_per_mode": args.rounds,
        "seed": args.seed,
        "smoke": args.smoke,
        "profiling": bool(args.nsys or os.getenv("SGLANG_VISUAL_KV_PROFILE")),
        "verification_mode": bool(os.getenv("SGLANG_VISUAL_KV_VERIFY_DIR")),
        "verification_after_directory": str(args.verify_after)
        if args.verify_after is not None
        else None,
        "shared_gpu": "The benchmark does not reserve the device; check concurrent utilization for interpretation.",
        "prefill_decode_clock": "Scheduler.run_batch wall time through a CUDA synchronization after each batch",
        "process_gpu_peak_definition": "Server worker's PyTorch allocator peak after reset immediately before prefill, through the last decode step",
        "ttft_clock": "Client elapsed wall time through first nonempty streamed text",
        "tpot_clock": "Client first-to-last nonempty text interval divided by generated tokens minus one",
        "cold_definition": "Model resident, image features and decoder KV cold for this image. Weight initialization is excluded.",
        "decoder_cache_bypass": ["cold_full", "feature_only"],
        "vision_cache_bypass": ["cold_full"],
        "versions": {
            package: version(package)
            for package in (
                "torch",
                "sglang",
                "transformers",
                "flashinfer-python",
                "triton",
            )
        },
        "cases": cases,
        "diagnostic_suites": diagnostics,
        "server_commands": {},
    }
    groups = [args.modes] if args.interleave_full_region else [
        group
        for group in (
            [mode for mode in args.modes if mode not in REGION],
            [mode for mode in args.modes if mode in REGION],
        )
        if group
    ]
    samples = []
    diagnostic_samples = []
    with ExitStack() as stack:
        write = stack.enter_context(output_writer(args.out, samples))
        write_diagnostic = (
            stack.enter_context(
                output_writer(args.out, diagnostic_samples, "diagnostic_samples")
            )
            if diagnostics or args.verify_after is not None
            else None
        )
        for modes in groups:
            try:
                run_group(
                    args,
                    metadata,
                    modes,
                    cases,
                    diagnostics,
                    manifest,
                    model,
                    image,
                    write,
                    write_diagnostic,
                )
            finally:
                write_summary(args.out, metadata, samples, diagnostic_samples)


if __name__ == "__main__":
    main()
