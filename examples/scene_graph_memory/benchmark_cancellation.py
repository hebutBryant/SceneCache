#!/usr/bin/env python3
"""Check stream cancellation and live region-selection replacement on an idle benchmark server."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4

from PIL import Image
import requests

from benchmark_inference import REGION, TraceCursor, ask, launch_server, replace_json


def _selection(manifest, region_ids, hops):
    by_id = {region["region_id"]: region for region in manifest["regions"]}
    ids = set(region_ids)
    if unknown := ids - by_id.keys():
        raise ValueError(f"Unknown region IDs {sorted(unknown)}")
    for _ in range(hops):
        neighbors = {
            neighbor
            for edge in manifest["edges"]
            for node, neighbor in ((edge["region_a"], edge["region_b"]),
                                   (edge["region_b"], edge["region_a"]))
            if node in ids
        }
        if neighbors <= ids:
            break
        ids.update(neighbors)
    indices = {index for name in ids for index in by_id[name]["token_indices"]}
    return {
        "config": {"region_ids": list(region_ids), "expand_hops": hops},
        "expected_region_ids": sorted(ids),
        "expected_active_rows": len(indices),
    }


def _selections(manifest, first_ids, first_hops, next_ids, next_hops):
    candidates = [
        region["region_id"] for region in manifest["regions"]
        if len(region["token_indices"]) >= 2
    ]
    if not first_ids:
        if not candidates:
            raise ValueError("Choose a region selection with at least two active visual rows")
        first_ids = [candidates[0]]
    first = _selection(manifest, first_ids, first_hops)
    if not next_ids:
        for region in candidates:
            candidate = _selection(manifest, [region], next_hops)
            if (candidate["expected_region_ids"] != first["expected_region_ids"]
                    and candidate["expected_active_rows"] != first["expected_active_rows"]):
                next_ids = [region]
                break
    second = _selection(manifest, next_ids or [], next_hops)
    if min(first["expected_active_rows"], second["expected_active_rows"]) < 2:
        raise ValueError("Both selections need at least two active rows to exercise KV reuse")
    if (first["expected_region_ids"] == second["expected_region_ids"]
            or first["expected_active_rows"] == second["expected_active_rows"]):
        raise ValueError("Choose distinct region selections with different active row counts")
    return first, second


def _benchmark_image(memory_dir, manifest):
    with Image.open(memory_dir / "image.png") as stored:
        image = stored.convert("RGB")
    max_edge = manifest["config"].get("max_image_edge")
    if max_edge is not None:
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode()},
    }


def _get_load(session, base, timeout):
    response = session.get(base + "/get_load", timeout=timeout)
    response.raise_for_status()
    ranks = response.json()
    if not isinstance(ranks, list) or not ranks:
        raise RuntimeError(f"Unexpected /get_load response: {ranks!r}")
    return ranks


def _wait_idle(session, base, timeout):
    started = time.monotonic()
    polls = 0
    while (remaining := timeout - (time.monotonic() - started)) > 0:
        ranks = _get_load(session, base, timeout=max(remaining, 0.05))
        polls += 1
        if all(rank["num_reqs"] == 0 and rank["num_waiting_reqs"] == 0 for rank in ranks):
            return {"ranks": ranks, "polls": polls, "elapsed_ms": (time.monotonic() - started) * 1000}
        time.sleep(0.1)
    raise TimeoutError(f"Scheduler did not become idle in {timeout}s after {polls} polls")


def _allocated_rows(idle):
    return {rank["dp_rank"]: rank["num_tokens"] for rank in idle["ranks"]}


def _stream_then_disconnect(session, base, image, model, question, request_id, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [image, {"type": "text", "text": question}]}],
        "rid": request_id,
        "temperature": 0,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.monotonic_ns()
    first_chunk = None
    with session.post(base + "/v1/chat/completions", json=payload, stream=True, timeout=180) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data:"):
                continue
            content = line[5:].strip()
            if content == b"[DONE]":
                break
            message = json.loads(content)
            for choice in message.get("choices", []):
                if choice.get("finish_reason"):
                    raise RuntimeError("The long request already finished before disconnect")
                text = choice.get("delta", {}).get("content")
                if text:
                    first_chunk = text
                    break
            if first_chunk is not None:
                # Do not consume the rest of this SSE response. No explicit
                # /abort_request is sent; the HTTP disconnect exercises cleanup.
                break
    if first_chunk is None:
        raise RuntimeError("No nonempty text arrived before the stream ended")
    return {
        "request_id": request_id,
        "first_text": first_chunk,
        "close_after_first_text_ms": (time.monotonic_ns() - started) / 1e6,
        "requested_output_tokens": max_tokens,
    }


def _check_request(runtime, scheduler, request_id, plan, mode, total_rows, bytes_per_visual_row):
    records = [
        row for row in runtime.read()
        if row.get("event", row.get("stage")) == "optimized_prefill"
    ]
    if len(records) != 1:
        raise RuntimeError(f"{request_id}: expected one optimized visual prefill, got {len(records)}")
    record = records[0]
    if (
        record["mode"] != mode or record["n_all"] != total_rows
        or record["n_active"] != plan["expected_active_rows"]
        or sorted(record["region_ids"]) != plan["expected_region_ids"]
        or not 0 < record["n_recompute"] < record["n_active"]
        or record["h2d_bytes"] != (record["n_active"] - record["n_recompute"]) * bytes_per_visual_row
    ):
        raise RuntimeError(f"{request_id}: expected {plan}, saw {record}")
    batches = scheduler.read()
    if any(row["request_id"] != request_id for row in batches):
        raise RuntimeError(f"{request_id}: concurrent scheduler activity makes the sample ambiguous")
    prefill_count = sum(row["stage"] == "prefill" for row in batches)
    if prefill_count != 1:
        raise RuntimeError(f"{request_id}: expected one scheduler prefill, got {prefill_count}")
    return {
        "request_id": request_id,
        "selection": plan,
        "prefill_batches": prefill_count,
        "decode_steps": sum(row["stage"] == "decode" for row in batches),
        "metrics": record,
    }


def _restore_file(path, content):
    if content is None:
        path.unlink(missing_ok=True)
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        stream.write(content)
    Path(stream.name).replace(path)


def run(args):
    manifest = json.loads((args.memory_dir / "manifest.json").read_text())
    first, second = _selections(
        manifest, args.initial_regions, args.initial_hops, args.next_regions, args.next_hops
    )
    selection = args.selection.resolve()
    control = selection.with_name("control.json")
    if control == selection:
        raise ValueError("--selection must identify the server selection file, not control.json")
    runtime = TraceCursor(args.metrics or selection.with_name("runtime_metrics.jsonl"))
    scheduler = TraceCursor(args.scheduler or selection.with_name("scheduler_timing.jsonl"))
    model = args.model or Path(manifest["fingerprints"]["model_path"]).name
    image = _benchmark_image(args.memory_dir, manifest)
    total_rows = len(manifest["tokens"])
    bytes_per_visual_row, remainder = divmod(manifest["summary"]["kv_bank_bytes"], total_rows)
    if remainder or bytes_per_visual_row <= 0:
        raise ValueError("Source bank does not have a consistent byte count per visual row")
    base = args.base.rstrip("/")
    nonce = uuid4().hex[:12]

    saved = {path: path.read_bytes() if path.exists() else None for path in (selection, control)}
    with requests.Session() as session:
        session.trust_env = False
        idle_before = _wait_idle(session, base, args.idle_timeout)
        runtime.read()
        scheduler.read()
        try:
            canceled_id = f"scg_cancel_{nonce}"
            replace_json(selection, first["config"])
            replace_json(control, {"request_id": canceled_id, "bench_mode": args.mode})
            disconnected = _stream_then_disconnect(
                session, base, image, model, args.question, canceled_id, args.abort_tokens
            )
            # Switch the selection while the server may still be finishing the
            # disconnect. Delay moving the trace request ID until it is idle.
            replace_json(selection, second["config"])
            idle_after_abort = _wait_idle(session, base, args.idle_timeout)
            canceled = _check_request(
                runtime, scheduler, canceled_id, first, args.mode, total_rows, bytes_per_visual_row
            )
            if canceled["decode_steps"] >= args.abort_tokens - 1:
                raise RuntimeError("The long request ran its full decode budget, so early cancellation was not observed")
            if _allocated_rows(idle_after_abort) != _allocated_rows(idle_before):
                raise RuntimeError("Allocator occupancy did not return to the idle baseline after disconnect")
            print(
                f"Disconnected after first text; scheduler ran {canceled['decode_steps']} of "
                f"{args.abort_tokens - 1} possible decode steps.", flush=True,
            )

            followups = []
            for index in range(2):
                request_id = f"scg_follow_{nonce}_{index + 1}"
                replace_json(control, {"request_id": request_id, "bench_mode": args.mode})
                answer = ask(
                    session, base, image, model, args.question, args.followup_tokens,
                    request_id, fixed_tokens=args.followup_tokens,
                )
                idle = _wait_idle(session, base, args.idle_timeout)
                measured = _check_request(
                    runtime, scheduler, request_id, second, args.mode, total_rows, bytes_per_visual_row
                )
                if measured["decode_steps"] != args.followup_tokens - 1:
                    raise RuntimeError(f"{request_id}: expected {args.followup_tokens - 1} decode steps")
                if _allocated_rows(idle) != _allocated_rows(idle_before):
                    raise RuntimeError(f"{request_id}: allocator occupancy did not return to baseline")
                followups.append({**measured, "answer": answer, "idle": idle})
                print(
                    f"{request_id}: {answer['output_tokens']} output tokens, "
                    f"{measured['metrics']['n_active']} active visual rows.", flush=True,
                )
        finally:
            for path, content in saved.items():
                _restore_file(path, content)

    return {
        "status": "passed",
        "method": "Real SSE client disconnect after first nonempty text; no explicit abort API call.",
        "server": base,
        "mode": args.mode,
        "memory_fingerprint": manifest["fingerprint"],
        "before": idle_before,
        "disconnected_request": {**disconnected, **canceled},
        "after_disconnect": idle_after_abort,
        "followups": followups,
        "selection_restored": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use on a benchmark server launched with benchmark_trace instrumentation and no other clients.",
    )
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--base", help="Existing benchmark-compatible server address")
    location.add_argument("--gpu", help="Physical GPU index or UUID to launch its own server")
    parser.add_argument("--selection", type=Path, help="Selection JSON used by an existing server")
    parser.add_argument("--memory-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=REGION, default="region_pipeline_triton")
    parser.add_argument("--model", help="Override the manifest model name in HTTP requests")
    parser.add_argument("--metrics", type=Path, help="Default: runtime_metrics.jsonl beside the selection")
    parser.add_argument("--scheduler", type=Path, help="Default: scheduler_timing.jsonl beside the selection")
    parser.add_argument("--initial-regions", nargs="+")
    parser.add_argument("--initial-hops", type=int, default=0)
    parser.add_argument("--next-regions", nargs="+")
    parser.add_argument("--next-hops", type=int, default=0)
    parser.add_argument("--abort-tokens", type=int, default=512)
    parser.add_argument("--followup-tokens", type=int, default=8)
    parser.add_argument("--idle-timeout", type=float, default=30.0)
    parser.add_argument("--question", default="请描述这张图片里的场景。")
    parser.add_argument("--out", required=True, type=Path)
    server = parser.add_argument_group("Optional isolated launch with --gpu")
    server.add_argument("--port", type=int, default=30009)
    server.add_argument("--context-length", type=int, default=2048)
    server.add_argument("--mem-fraction-static", type=float, default=0.97)
    server.add_argument("--seed", type=int, default=42)
    server.add_argument("--model-path", type=Path, help="Default: model recorded in the manifest")
    server.add_argument("--cuda-home", type=Path, default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8")))
    server.add_argument("--server-dir", type=Path, help="New directory for launch and benchmark logs")
    args = parser.parse_args(argv)
    if min(args.initial_hops, args.next_hops) < 0:
        parser.error("Region hops must be nonnegative")
    if min(args.abort_tokens, args.followup_tokens) < 2 or args.idle_timeout <= 0:
        parser.error("Token limits must be at least two and the idle timeout must be positive")
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.base:
        if not args.selection:
            parser.error("--base requires the existing server's --selection path")
        result = run(args)
    else:
        if args.selection or args.metrics or args.scheduler:
            parser.error("Use --server-dir to choose where the self-launched server writes its logs")
        manifest = json.loads((args.memory_dir / "manifest.json").read_text())
        model = (args.model_path or Path(manifest["fingerprints"]["model_path"])).absolute()
        directory = (args.server_dir or args.out.with_name(args.out.stem + "_server")).resolve()
        directory.parent.mkdir(parents=True, exist_ok=True)
        args.nsys = None
        with launch_server(args, directory, model, args.mode) as (_, args.base, command):
            args.selection = directory / "selection.json"
            result = run(args)
            result["server_dir"] = str(directory)
            result["server_command"] = command
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Cancellation checks passed. Evidence written to {args.out}")


if __name__ == "__main__":
    main()
