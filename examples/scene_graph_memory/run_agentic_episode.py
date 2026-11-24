"""Build a question-scoped semantic index and replay one agent's visual operations."""

import argparse
import base64
from collections import defaultdict
from io import BytesIO
import json
from math import isfinite
import os
from pathlib import Path
import re
from statistics import median
from time import perf_counter
from types import SimpleNamespace

from PIL import Image, ImageOps
import requests

from benchmark_inference import ROOT, TraceCursor, ask, launch_server, replace_json
from sglang.srt.mem_cache.scene_graph_memory.agentic_graph import (
    GraphBudget,
    build_agentic_graph,
    make_sam3_segmenter,
)
from sglang.srt.mem_cache.scene_graph_memory.agentic_index import AgenticEpisode, AgenticKVIndex
from sglang.srt.mem_cache.scene_graph_memory.agentic_plan_import import import_scene2graph_plan
from sglang.srt.mem_cache.scene_graph_memory.builder import BuildConfig, build_image_memory
from sglang.srt.mem_cache.scene_graph_memory.schema import image_sha256
from sglang.srt.mem_cache.scene_graph_memory.selection import _selection
from sglang.srt.mem_cache.scene_graph_memory.store import inspect_summary, save_memory
from sglang.srt.mem_cache.scene_graph_memory.visualization import write_visualization


ARMS = ("A_full", "B_crop_cache", "C_geometry", "D_semantic")
INSUFFICIENT = "INSUFFICIENT"


def image_input(image, max_edge):
    submitted = image.copy()
    submitted.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    payload = BytesIO()
    submitted.save(payload, format="PNG")
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(payload.getvalue()).decode("ascii")
    }}


class Trace:
    def __init__(self, directory):
        self.path = directory / "episode_trace.jsonl"
        self.started = perf_counter()
        self.events = []

    def log(self, event, **data):
        row = {"event": event, "episode_s": perf_counter() - self.started, **data}
        self.events.append(row)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if event in {"vlm_reason", "sam3", "source_bank", "inference", "graph_ready", "worker_ready"}:
            print(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str), flush=True)


class LocalReasoner:
    def __init__(self, session, base, model, max_tokens, trace):
        self.session = session
        self.base = base.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.trace = trace

    def __call__(self, messages):
        started = perf_counter()
        try:
            response = self.session.post(
                f"{self.base}/v1/chat/completions",
                json={"model": self.model, "messages": messages, "temperature": 0,
                      "max_tokens": self.max_tokens, "response_format": {"type": "json_object"}},
                timeout=180,
            )
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            content = choice["message"]["content"]
        except (requests.RequestException, ValueError, KeyError, IndexError) as error:
            failure = getattr(error, "response", None)
            self.trace.log("vlm_reason", elapsed_s=perf_counter() - started,
                           endpoint=self.base, model=self.model, error=str(error),
                           response_excerpt=failure.text[:500] if failure is not None else None)
            raise RuntimeError(f"VLM call failed: {error}") from error
        self.trace.log("vlm_reason", elapsed_s=perf_counter() - started,
                       endpoint=self.base, model=self.model, usage=body.get("usage"),
                       finish_reason=choice.get("finish_reason"), result=content)
        return content


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--existing-plan", type=Path, help="Plan JSON or a Scene2Graph archive")
    parser.add_argument("--resume-build", type=Path, help="Replay the same question with a validated episode build")
    parser.add_argument("--observations", type=Path, help="JSON with prior visual observations")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vlm-url", default="http://127.0.0.1:30000")
    parser.add_argument("--gpu", required=True, help="Physical GPU for SAM3, capture and QA")
    parser.add_argument("--port", type=int, default=30019)
    parser.add_argument("--sam3-checkpoint", type=Path)
    parser.add_argument("--sam3-source", type=Path)
    parser.add_argument("--sam3-python", type=Path)
    parser.add_argument("--sam3-device", default="cuda:0")
    parser.add_argument("--sam3-timeout", type=float, default=300.0)
    parser.add_argument("--cuda-home", type=Path, default=Path("/usr/local/cuda-12.8"))
    parser.add_argument("--max-image-edge", type=int, default=1344)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--max-image-tokens", type=int, default=1536)
    parser.add_argument("--mem-fraction-static", type=float, default=0.97)
    parser.add_argument("--max-vlm-calls", type=int, default=3)
    parser.add_argument("--max-sam-calls", type=int, default=2)
    parser.add_argument("--max-concepts", type=int, default=8)
    parser.add_argument("--max-regions", type=int, default=16)
    parser.add_argument("--max-evidence-regions", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--max-operations", type=int, default=3, help="Follow-ups before final answer")
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--max-reason-tokens", type=int, default=768)
    parser.add_argument("--token-budget", type=int, default=400)
    parser.add_argument("--background-tokens", type=int, default=16)
    parser.add_argument("--max-retrieval-expansions", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-capture", action="store_true")
    parser.add_argument("--replay-rounds", type=int, default=1)
    parser.add_argument("--closed-loop", action="store_true",
                        help="Run a separate semantic episode using prior live tool answers")
    parser.add_argument("--accepted", action="append", default=[], help="Accepted final answer, repeatable")
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    args = parser.parse_args()
    if (min(args.port, args.max_image_edge, args.max_image_tokens, args.max_answer_tokens,
            args.max_reason_tokens, args.token_budget, args.replay_rounds, args.max_candidates) < 1
            or min(args.max_operations, args.background_tokens) < 0):
        parser.error("Expected positive resource limits and nonnegative operation/background caps")
    if not isfinite(args.sam3_timeout) or args.sam3_timeout <= 0:
        parser.error("--sam3-timeout must be positive")
    if len(args.arms) != len(set(args.arms)):
        parser.error("Evaluation arms must be unique")
    if args.resume_build and (args.existing_plan or args.observations or args.verify_capture):
        parser.error("Replay of a saved build keeps its original plan and capture")
    if not args.resume_build and not all((args.sam3_checkpoint, args.sam3_source, args.sam3_python)):
        parser.error("A new build requires --sam3-checkpoint, --sam3-source and --sam3-python")
    return args


def operation_items(plan, question, maximum):
    operations = []
    for operation in plan.get("operations", [])[:maximum]:
        prompt = operation.get("question")
        if operation.get("kind") != "answer" and prompt and prompt.casefold().strip() != question.casefold().strip():
            operations.append({**operation, "question": prompt})
    planned_answer = next((operation for operation in plan.get("operations", [])
                           if operation.get("kind") == "answer"), {})
    if not planned_answer:
        named = [concept for concept in plan.get("concepts", [])
                 if isinstance(concept, str) and normalize(concept) in normalize(question)]
        if named:
            planned_answer = {"concepts": named}
    operations.append({**planned_answer, "id": "final_question", "question": question, "final": True})
    return operations


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def is_insufficient(answer):
    return normalize(answer).startswith(INSUFFICIENT.casefold())


def geometry_regions(memory, operation, plan):
    concepts = plan.get("concepts", [])
    target_ids = operation.get("concepts") or operation.get("concept_ids") or operation.get("targets") or []
    wanted = []
    for item in concepts:
        if isinstance(item, str):
            key, phrase = item, item
        else:
            key = item.get("id")
            phrase = item.get("concept") or item.get("label") or item.get("name") or item.get("text")
        if phrase and (not target_ids or key in target_ids or phrase in target_ids):
            wanted.append(normalize(phrase))
    # Prefer concepts actually mentioned in the question when no explicit targets exist.
    matches = [label for label in wanted if label in normalize(operation["question"])]
    wanted = wanted if target_ids or not matches else matches
    return [
        region for region in memory.regions
        if any(label in normalize(region.label or "") or normalize(region.label or "") in label
               for label in wanted if region.label)
    ]


def crop_image(image, regions):
    if not regions:
        return image, None
    boxes = [r.bbox_xyxy for r in regions]
    x0, y0 = min(box[0] for box in boxes), min(box[1] for box in boxes)
    x1, y1 = max(box[2] for box in boxes), max(box[3] for box in boxes)
    border = max(8, round(max(x1 - x0, y1 - y0) * 0.12))
    box = (max(0, x0 - border), max(0, y0 - border), min(image.width, x1 + border),
           min(image.height, y1 + border))
    return image.crop(box), box


def geometry_selection(regions, index_dir, token_budget):
    if not regions:
        return None, "no_geometry_region_matches"
    for hops in (1, 0):
        requested = {"region_ids": [region.region_id for region in regions], "expand_hops": hops}
        try:
            _, resolved, _, active = _selection(str(index_dir), json.dumps(requested).encode())
        except ValueError as error:
            return None, str(error)
        if len(active) <= token_budget:
            return requested, {"region_ids": list(resolved), "source_visual_indices": list(active)}
    return None, "geometry_regions_exceed_token_budget"


class InferenceReplay:
    def __init__(self, server, traces, timings, runtime, image, model, args, trace, out):
        self.session, self.base = server[:2]
        self.traces, self.timings, self.runtime = traces, timings, runtime
        self.image = image
        self.full_image = image_input(image, args.max_image_edge)
        self.model, self.args, self.trace, self.out = model, args, trace, out
        self.count = 0
        self.steps = []

    def call(self, arm, operation, *, round_index, selection=None, crop=None, reset=False, reason=None,
             active=None):
        self.count += 1
        rid = f"ep{self.count:04d}_{arm}"
        mode = "region_pipeline" if selection is not None else "pipelined_full_kv"
        if selection is not None:
            replace_json(self.out / "selection.json", selection)
        replace_json(self.out / "control.json", {
            "request_id": rid, "condition": arm, "bench_mode": mode, "reset": reset,
        })
        payload = self.full_image if crop is None else image_input(crop, self.args.max_image_edge)
        answer = ask(
            self.session, self.base, payload, self.model,
            operation["question"] + f" Reply briefly. If the image evidence is missing, reply exactly {INSUFFICIENT}.",
            self.args.max_answer_tokens, rid,
        )
        runtime = self.runtime.read()
        stages = [series for series in self.timings.read() if series["request_id"] == rid]
        native = [row for row in self.traces.read() if row["request_id"] == rid]
        phase = [record for record in native if record["event"] == "prefill"]
        if len(phase) != 1:
            raise RuntimeError(f"Expected one actual multimodal prefill for {rid}, got {len(phase)}")
        if reset and not any(row["event"] == "cache_reset" for row in native):
            raise RuntimeError(f"Missing cache isolation for {rid}")
        active_rows = sum(last - start + 1 for start, last in phase[0]["image_ranges"])
        optimized = next((row for row in runtime if row["event"] == "optimized_prefill"), None)
        fallback = next((row["reason"] for row in runtime if row["event"] == "visual_kv_fallback"), None)
        if selection is not None and (optimized is None or optimized["n_active"] != len(active)):
            raise RuntimeError(f"Region selection did not reach the real attention backend for {rid}")
        reads = [event for event in native if event["event"] == "get_kv" and event["kind"] == "kv"]
        writes = [event for event in native if event["event"] == "write_kv" and event["kind"] == "kv"]
        embedding_hits = sum(event["event"] == "embedding_cache_get" and event["hit"] for event in native)
        mask = phase[0]["first_layer_compute_mask"]
        recomputed = active_rows if mask is None else sum(
            sum(mask[start:last + 1]) for start, last in phase[0]["image_ranges"]
        )
        h2d = sum(event["bytes"] for event in reads)
        index_h2d = 0
        bank_rows = []
        if optimized is not None:
            active_rows = optimized["n_active"]
            recomputed = optimized["n_recompute"]
            h2d = optimized["h2d_bytes"]
            index_h2d = optimized["h2d_index_bytes"]
            if active is not None:
                bank_rows = active[recomputed:]
        decode = [stage for stage in stages if stage["stage"] == "decode"]
        decode_events = [row for row in runtime if row["event"] == "optimized_decode"]
        if selection is not None and (
            len(decode_events) != len(decode)
            or any(row["n_active"] != active_rows or row["h2d_bytes"] for row in decode_events)
        ):
            raise RuntimeError(f"Region decode telemetry mismatch for {rid}")
        decode_ms = sum(row["finish_ns"] - row["start_ns"] for row in decode) / 1e6
        record = {
            "arm": arm, "round": round_index, "operation": operation["id"],
            "request_id": rid, "mode": mode, "cache_cold": reset,
            "answer": answer["answer"], "output_tokens": answer["output_tokens"],
            "ttft_ms": answer["ttft_ms"], "client_tpot_ms": answer["tpot_ms"],
            "total_ms": answer["total_ms"], "decode_steps": len(decode),
            "server_tpot_ms": decode_ms / len(decode) if decode else None,
            "n_all": active_rows if optimized is None else optimized["n_all"],
            "n_active": active_rows, "n_recompute": recomputed, "kv_h2d_bytes": h2d,
            "index_h2d_bytes": index_h2d, "kv_d2h_bytes": sum(event["bytes"] for event in writes),
            "embedding_cache_hits": embedding_hits, "read_source_bank_rows": bank_rows,
            "backend_reported": optimized["backend"] if optimized else "legacy_flashinfer",
            "source_reported": optimized["source"] if optimized else "native_full_prefill",
            "decode_kv_lengths": [item["kv_length"] for item in decode],
            "decode_region_kv_h2d_bytes": sum(row["h2d_bytes"] for row in decode_events) if selection is not None else None,
            "fallback": fallback,
            "selection_reason": reason,
            "input_image_sha256": image_sha256(self.image if crop is None else crop),
            "gpu_peak_allocated_bytes": max(stage["process_gpu_peak_allocated_bytes"] for stage in stages),
            "gpu_peak_reserved_bytes": max(stage["process_gpu_peak_reserved_bytes"] for stage in stages),
        }
        self.steps.append(record)
        self.trace.log("inference", **{**record, "read_source_bank_rows": len(bank_rows)})
        return record


def save_json(path, document):
    replace_json(path, document)


def run(args):
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    trace = Trace(args.out)
    model_path = args.model_path.absolute()
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    trace.log("episode_start", image=str(args.image.resolve()), image_hash=image_sha256(image),
              question=args.question, gpu=args.gpu, arms=args.arms)
    os.environ.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": args.gpu,
        "CUDA_HOME": str(args.cuda_home), "CUDACXX": str(args.cuda_home / "bin/nvcc"),
        "PATH": f"{args.cuda_home / 'bin'}:{os.environ['PATH']}",
        "FLASHINFER_WORKSPACE_BASE": str(ROOT / ".runtime/flashinfer"),
        "TRITON_CACHE_DIR": str(ROOT / ".runtime/triton"),
    })
    reference_cost = None
    if args.resume_build is None:
        existing = None
        plan_import_s = 0.0
        if args.existing_plan:
            started = perf_counter()
            existing = (import_scene2graph_plan(args.existing_plan, image=image, question=args.question)
                        if args.existing_plan.is_dir() else json.loads(args.existing_plan.read_text()))
            plan_import_s = perf_counter() - started
        observations = json.loads(args.observations.read_text()) if args.observations else None

        segmenter = make_sam3_segmenter(
            image, sam3_source=args.sam3_source, sam3_python=args.sam3_python,
            sam3_checkpoint=args.sam3_checkpoint, sam3_device=args.sam3_device,
            sam3_max_candidates=args.max_candidates, sam3_timeout=args.sam3_timeout,
        )
        segmentations = []

        def segment(concepts):
            started = perf_counter()
            try:
                masks, provenance = segmenter(concepts)
            except (RuntimeError, ValueError, OSError) as error:
                trace.log("sam3", elapsed_s=perf_counter() - started, concepts=concepts, error=str(error))
                raise
            segmentations.append(provenance)
            trace.log("sam3", elapsed_s=perf_counter() - started, concepts=concepts,
                      retained=len(masks), backend=provenance.get("backend"))
            return masks, provenance

        budget = GraphBudget(max_vlm_calls=args.max_vlm_calls, max_sam_calls=args.max_sam_calls,
                             max_regions=args.max_regions, max_concepts=args.max_concepts,
                             max_evidence_regions=args.max_evidence_regions)
        with requests.Session() as session:
            session.trust_env = False
            reason = LocalReasoner(session, args.vlm_url, model_path.name, args.max_reason_tokens, trace)
            started = perf_counter()
            graph = build_agentic_graph(image, question=args.question, reason=reason, segment=segment,
                                        existing_plan=existing, observations=observations, budget=budget)
            graph_s = perf_counter() - started

        graph_document = graph.manifest()
        save_json(args.out / "semantic_graph.json", graph_document)
        evidence_dir = args.out / "evidence"
        evidence_dir.mkdir()
        for board_id, board in graph.evidence_images.items():
            board.save(evidence_dir / f"{board_id}.png")
        plan = graph_document["plan"]
        save_json(args.out / "plan.json", plan)
        operations = operation_items(plan, args.question, args.max_operations)
        trace.log("graph_ready", elapsed_s=graph_s, masks=len(graph.masks), operations=operations)

        config = BuildConfig(model_path=model_path, device="cuda:0", dtype="bfloat16",
                             context_length=args.context_length, max_image_tokens=args.max_image_tokens,
                             max_image_edge=args.max_image_edge, mem_fraction_static=args.mem_fraction_static,
                             verify_capture=args.verify_capture)
        phases = {}
        started = perf_counter()
        memory = build_image_memory(
            image, config, prepared_masks=(graph.masks, {"backend": "plan_guided_sam3",
                "batches": segmentations, "graph_schema_version": graph_document["schema_version"]}),
            timings=phases,
        )
        index_dir = args.out / "memory"
        save_started = perf_counter()
        save_memory(memory, index_dir)
        phases["validate_and_save_s"] = perf_counter() - save_started
        build_s = perf_counter() - started
        capture = memory.kv_metadata.details["verification"]
        stats = inspect_summary(memory)
        trace.log("source_bank", elapsed_s=build_s, phases=phases, capture=capture,
                  source_bank_bytes=stats["kv_bank_bytes"], visual_rows=len(memory.tokens))
        write_visualization(memory, args.out / "mask_token_alignment.html")

        started = perf_counter()
        graph_document = graph.manifest(memory)
        save_json(args.out / "semantic_graph.json", graph_document)
        del memory
        index = AgenticKVIndex(index_dir, graph_document)
        memory = index.memory
        save_json(args.out / "semantic_kv_index.json", index.to_dict())
        join_s = perf_counter() - started
        trace.log("semantic_index", elapsed_s=join_s, memory_fingerprint=memory.fingerprint)
    else:
        source = args.resume_build.resolve()
        graph_document = json.loads((source / "semantic_graph.json").read_text(encoding="utf-8"))
        plan = graph_document["plan"]
        if plan["question"] != args.question or graph_document["image_hash"] != image_sha256(image):
            raise ValueError("A resumed build must match the original image pixels and episode question")
        reference_cost = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        index_dir = source / "memory"
        started = perf_counter()
        index = AgenticKVIndex(index_dir, graph_document)
        join_s = perf_counter() - started
        memory = index.memory
        if memory.config["max_image_edge"] != args.max_image_edge:
            raise ValueError("A resumed build must keep the original pre-processing image edge")
        phases = reference_cost["physical_phases_s"]
        graph_s = reference_cost["graph_wall_s"]
        build_s = reference_cost["build_wall_s"]
        plan_import_s = 0.0
        stats = inspect_summary(memory)
        capture = memory.kv_metadata.details["verification"]
        operations = operation_items(plan, args.question, args.max_operations)
        save_json(args.out / "semantic_graph.json", graph_document)
        save_json(args.out / "plan.json", plan)
        save_json(args.out / "semantic_kv_index.json", index.to_dict())
        write_visualization(memory, args.out / "mask_token_alignment.html")
        trace.log("saved_build_resumed", directory=str(source), elapsed_load_s=join_s,
                  memory_fingerprint=memory.fingerprint)
    save_json(args.out / "agent_operations.json", operations)
    episode = AgenticEpisode(index)
    launch_args = SimpleNamespace(
        gpu=args.gpu, nsys=None, cuda_home=args.cuda_home, context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static, seed=args.seed, memory_dir=index_dir,
        interleave_full_region=True, port=args.port,
    )
    server_dir = args.out / "worker"
    arm_spans = []
    started = perf_counter()
    retrieval_seconds = defaultdict(float)
    with launch_server(launch_args, server_dir, model_path, "region_pipeline") as server:
        worker_startup_s = perf_counter() - started
        runtime = TraceCursor(server_dir / "runtime_metrics.jsonl")
        bank_load = [row for row in runtime.read() if row["event"] == "visual_kv_memory_load"]
        server_load_s = sum(row["elapsed_ms"] for row in bank_load) / 1000
        trace.log("worker_ready", elapsed_s=worker_startup_s, source_load_and_verify_s=server_load_s,
                  command=server[2])
        runner = InferenceReplay(server, TraceCursor(server_dir / "trace.jsonl"),
                                 TraceCursor(server_dir / "scheduler_timing.jsonl"), runtime,
                                 image, model_path.name, args, trace, server_dir)
        for turn in range(args.replay_rounds):
            order = args.arms[turn % len(args.arms):] + args.arms[:turn % len(args.arms)]
            workloads = [(arm, False) for arm in order]
            if args.closed_loop and turn == args.replay_rounds - 1:
                workloads.append(("D_semantic", True))
            for arm, adaptive in workloads:
                report_arm = "D_closed_loop" if adaptive else arm
                episode_start = perf_counter()
                first = True
                notes = []
                episode.reset()
                for item in operations:
                    operation = item if item.get("concepts") or not item.get("final") else None
                    request_item = item
                    if notes:
                        history = "\n".join(notes)
                        request_item = {**item, "question": (
                            f"{item['question']}\n"
                            f"Earlier image observations (check against visible evidence):\n{history}"
                        )}
                    selection = None
                    active = None
                    crop = None
                    reason = None
                    start_retrieval = perf_counter()
                    if arm in {"B_crop_cache", "C_geometry"}:
                        targets = geometry_regions(memory, item, plan)
                        if arm == "C_geometry":
                            selection, reason = geometry_selection(targets, index_dir, args.token_budget)
                            if selection:
                                active = reason["source_visual_indices"]
                        else:
                            crop, box = crop_image(image, targets)
                            reason = {"crop_original_xyxy": box,
                                      "region_ids": [region.region_id for region in targets],
                                      "independent_crop_prefix": box is not None}
                            if box is None:
                                crop = None
                                reason["fallback"] = "no_crop_region_matches"
                    elif arm == "D_semantic":
                        for depth in range(args.max_retrieval_expansions + 1):
                            retrieved = episode.retrieve(item["question"], plan, operation=operation,
                                                       token_budget=min(args.token_budget, len(memory.tokens)), max_hops=depth,
                                                       background_tokens=args.background_tokens)
                            reason = retrieved.to_dict()
                            trace.log("retrieval", arm=report_arm, operation=item["id"], round=turn,
                                      expansion=depth, result=reason)
                            if not retrieved.fallback:
                                selection = retrieved.selection_dict()
                                active = selection["source_visual_indices"]
                                break
                    retrieval_seconds[report_arm] += perf_counter() - start_retrieval
                    row = runner.call(report_arm, request_item, round_index=turn, selection=selection, crop=crop,
                                      reset=first, reason=reason, active=active)
                    first = False
                    if (selection is not None or crop is not None) and is_insufficient(row["answer"]):
                        trace.log("fallback", arm=report_arm, operation=item["id"],
                                  reason="model_reported_insufficient_evidence")
                        if arm == "D_semantic" and args.max_retrieval_expansions:
                            started_expand = perf_counter()
                            expanded = episode.retrieve(item["question"], plan, operation=operation,
                                                      token_budget=min(args.token_budget, len(memory.tokens)), max_hops=1,
                                                      background_tokens=args.background_tokens)
                            expanded_selection = expanded.selection_dict() if not expanded.fallback else None
                            retrieval_seconds[report_arm] += perf_counter() - started_expand
                            trace.log("retrieval", arm=report_arm, operation=item["id"], round=turn,
                                      expansion=1, result=expanded.to_dict())
                            if expanded_selection and expanded_selection != selection:
                                row = runner.call(report_arm, request_item, round_index=turn, selection=expanded_selection,
                                                  active=expanded_selection["source_visual_indices"],
                                                  reason=expanded.to_dict())
                        if is_insufficient(row["answer"]):
                            row = runner.call(report_arm, request_item, round_index=turn, reason="native_full_after_insufficient")
                    if adaptive and not item.get("final") and not is_insufficient(row["answer"]):
                        notes.append(f"{item['id']}: {row['answer'].strip()}")
                        trace.log("observation_committed", arm=report_arm, operation=item["id"],
                                  request_id=row["request_id"], answer=row["answer"])
                arm_spans.append({"arm": report_arm, "round": turn, "elapsed_s": perf_counter() - episode_start,
                                  "adaptive_prior_observations": len(notes),
                                  "retrieval_decisions": len(episode.history),
                                  "final_answer": row["answer"],
                                  "correct": (normalize(row["answer"]) in {normalize(a) for a in args.accepted})
                                             if args.accepted else None})
    episode.reset()
    trace.log("worker_released", load_process_closed=True)
    summarize(args, trace, runner.steps, arm_spans, graph_s, plan_import_s, build_s, phases,
              capture, join_s, server_load_s, worker_startup_s, len(memory.tokens), stats["kv_bank_bytes"],
              retrieval_seconds, reference_cost)


def summarize(args, trace, steps, spans, graph_s, plan_import_s, build_s, phases, capture, join_s,
              load_s, worker_s, n_visual, bank_bytes, retrieval_seconds, reference_cost):
    sam_total = sum(row["elapsed_s"] for row in trace.events if row["event"] == "sam3")
    reason_calls = [row["elapsed_s"] for row in trace.events if row["event"] == "vlm_reason"]
    planning_s = plan_import_s if args.existing_plan else next(iter(reason_calls), 0.0)
    verify_s = sum(reason_calls) if args.existing_plan else sum(reason_calls[1:])
    if reference_cost is not None:
        planning_s = reference_cost["planning_s"]
        sam_total = reference_cost["sam3_s"]
        verify_s = reference_cost["vlm_validation_s"]
    validation_s = sum(value for phase, value in capture["phase_s"].items() if phase != "capture")
    totals = {}
    for arm in dict.fromkeys(span["arm"] for span in spans):
        rows = [row for row in steps if row["arm"] == arm]
        applicable = [span for span in spans if span["arm"] == arm]
        startup_cost = planning_s
        if arm != "A_full":
            startup_cost += sam_total
        if arm in ("C_geometry", "D_semantic", "D_closed_loop"):
            startup_cost += build_s + load_s
        if arm in ("D_semantic", "D_closed_loop"):
            startup_cost += verify_s + join_s
        used_rows = {row for sample in rows for row in sample["read_source_bank_rows"]}
        warm_calls = [row for row in rows if row["kv_h2d_bytes"] > 0]
        active_rows = sum(row["n_active"] for row in rows)
        recomputed_rows = sum(row["n_recompute"] for row in rows)
        cumulative, accumulated_s, accumulated_active, accumulated_reuse = [], 0.0, 0, 0
        for call_index, row in enumerate(rows, 1):
            accumulated_s += row["total_ms"] / 1000
            accumulated_active += row["n_active"]
            accumulated_reuse += row["n_active"] - row["n_recompute"]
            cumulative.append({
                "inference_calls": call_index, "operation": row["operation"],
                "build_and_http_inference_s": startup_cost + accumulated_s,
                "visual_rows_reused_fraction": accumulated_reuse / accumulated_active if accumulated_active else 0,
            })
        totals[arm] = {
            "episodes": [{**span, "serial_cold_equivalent_s": startup_cost + span["elapsed_s"]}
                         for span in applicable],
            "serial_build_charge_s": startup_cost, "wall_grounding_and_build_measured_once": True,
            "calls": len(rows), "actual_hits": sum(row["kv_h2d_bytes"] > 0 for row in rows),
            "active_visual_rows": active_rows,
            "recomputed_visual_rows": recomputed_rows,
            "visual_rows_reused_fraction": (active_rows - recomputed_rows) / active_rows if active_rows else 0,
            "cumulative_observed_calls": cumulative,
            "actual_kv_h2d_bytes": sum(row["kv_h2d_bytes"] for row in rows),
            "actual_kv_d2h_bytes": sum(row["kv_d2h_bytes"] for row in rows),
            "gpu_peak_allocated_bytes_shared_worker": max(row["gpu_peak_allocated_bytes"] for row in rows),
            "retrieval_s": retrieval_seconds[arm],
            "source_bank_untouched_fraction": 1 - len(used_rows) / n_visual
                if arm in ("C_geometry", "D_semantic", "D_closed_loop") else None,
            "warm_medians_ms": {
                key: median(values) if (values := [row[key] for row in warm_calls if row[key] is not None]) else None
                for key in ("ttft_ms", "server_tpot_ms", "client_tpot_ms", "total_ms")
            },
        }
    document = {
        "measurement_scope": "Same generated operation transcript replayed serially; one dedicated QA "
                             "worker, reset between arms, one canonical CPU source bank for C/D. "
                             "Cold-equivalent charges use the measured serial dependencies; "
                             "QA model server startup is reported separately. The capture subprocess "
                             "full wall time is charged including any requested verification.",
        "resumed_build": str(args.resume_build.resolve()) if args.resume_build else None,
        "historical_build_cost_used": reference_cost is not None,
        "episode_wall_s_all_arms": perf_counter() - trace.started,
        "plan_source": (reference_cost["plan_source"] if reference_cost else
                        ("existing" if args.existing_plan else "online_vlm")),
        "planning_s": planning_s, "sam3_s": sam_total, "vlm_validation_s": verify_s,
        "graph_wall_s": graph_s, "build_wall_s": build_s, "physical_phases_s": phases,
        "capture_validation_extra_s": validation_s, "capture_worker": capture,
        "semantic_index_s": join_s, "qa_worker_startup_s": worker_s,
        "qa_source_load_and_verify_s": load_s, "source_bank_bytes": bank_bytes,
        "source_visual_rows": n_visual, "physical_gpu": args.gpu, "arms": totals,
    }
    save_json(args.out / "inference_steps.json", steps)
    save_json(args.out / "summary.json", document)
    trace.log("episode_end", summary=str(args.out / "summary.json"))
    print(json.dumps(document, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_arguments())
