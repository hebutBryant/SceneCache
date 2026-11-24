#!/usr/bin/env python3
"""Compare same-image KV reuse against KV recomputation with vision embeddings held fixed."""

import argparse
import base64
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL = "/home/hdd4/lpz/Qwen3-VL-8B-Instruct"
LETTERS = "ABCD"


def compact_trace(events):
    prefill, = (event for event in events if event["event"] == "prefill")
    first, last = prefill["image_ranges"][0]
    temporal, height, width = prefill["image_grids"][0]
    mask = prefill["first_layer_compute_mask"]
    mask = mask[first:last + 1] if mask else [True] * (last - first + 1)
    assert len(mask) == temporal * height * width // 4
    return {
        "total": len(mask), "recomputed": sum(mask), "reused": len(mask) - sum(mask),
        "image_range": [first, last],
        "layers": prefill["cached_layers"], "reads": sum(
            event["event"] == "get_kv" and event["kind"] == "kv" for event in events
        ),
        "writes": sum(event["event"] == "write_kv" and event["kind"] == "kv" for event in events),
        "embedding_hits": sum(event["event"] == "embedding_cache_get" and event["hit"] for event in events),
        "input_tokens": prefill["total_input_tokens"],
        "rows": height // 2, "cols": width // 2, "mask": mask,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, default=30002)
    parser.add_argument("--cases", type=Path, default=ROOT / "scripts/kv_failure_cases.json")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/kv_failure")
    parser.add_argument("--max-edge", type=int, default=1344)
    parser.add_argument("--verify", type=int, default=5, help="maximum candidate regressions to repeat")
    parser.add_argument("--verify-results", type=Path, help="repeat candidate pairs from a prior screen")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bootstrap = output / "bootstrap"
    bootstrap.mkdir(exist_ok=True)
    (bootstrap / "sitecustomize.py").write_text("from scenecache_report_trace import install\ninstall()\n")
    trace_path = output / "trace.jsonl"
    trace_path.write_text("")
    (output / "control.json").unlink(missing_ok=True)
    env = os.environ.copy()
    env.update(
        SCENECACHE_REPORT_DIR=str(output),
        PYTHONPATH=os.pathsep.join(map(str, [bootstrap, ROOT / "scripts", ROOT / "python"])),
        CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_HOME="/usr/local/cuda-12.8", CUDACXX="/usr/local/cuda-12.8/bin/nvcc",
        CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda-12.8",
        LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:" + env.get("LD_LIBRARY_PATH", ""),
        PATH="/usr/local/cuda-12.8/bin:" + env["PATH"],
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4",
        SGLANG_CPU_WORKERS="2", SGLANG_VLM_CACHE_SIZE_MB="128", MAX_JOBS="4",
        FLASHINFER_WORKSPACE_BASE=str(ROOT / ".runtime/flashinfer"),
        TRITON_CACHE_DIR=str(ROOT / ".runtime/triton"),
    )
    command = [
        sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL,
        "--served-model-name", "Qwen3-VL-8B-Instruct", "--host", "127.0.0.1", "--port", str(args.port),
        "--dtype", "bfloat16", "--attention-backend", "flashinfer", "--mm-attention-backend", "triton_attn",
        "--sampling-backend", "pytorch", "--mem-fraction-static", "0.96", "--max-total-tokens", "8192",
        "--context-length", "4096", "--max-running-requests", "1", "--chunked-prefill-size", "4096",
        "--disable-cuda-graph", "--disable-radix-cache", "--trust-remote-code", "--random-seed", "42",
    ]
    session = requests.Session()
    session.trust_env = False
    base = f"http://127.0.0.1:{args.port}"
    data = json.loads(args.verify_results.read_text()) if args.verify_results else {
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "gpu": args.gpu, "model": "Qwen3-VL-8B-Instruct", "temperature": 0, "seed": 42,
        "command": command, "images": [], "cases": [], "requests": {},
    }
    image_content = {}
    counter = max((int(key.split("_", 1)[0]) for key in data["requests"]), default=0)

    def save():
        temp = output / "results.tmp"
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        temp.replace(output / "results.json")

    with trace_path.open() as trace_file, (output / "server.log").open("w") as log:
        server = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Demo server PID={server.pid}, GPU={args.gpu}, port={args.port}", flush=True)

        def ask(label, image, question, options=None, reset=False, keep_embedding=True, image_first=False):
            nonlocal counter
            counter += 1
            request_id = f"{counter:04d}_{label}"
            control = {"request_id": request_id, "reset": reset, "keep_embedding": keep_embedding}
            temp = output / "control.tmp"
            temp.write_text(json.dumps(control))
            temp.replace(output / "control.json")
            prompt = question
            if options:
                prompt += "\n" + "\n".join(f"{letter}. {value}" for letter, value in zip(LETTERS, options))
                prompt += "\n只回答选项字母，不要解释。"
            content = [{"type": "text", "text": prompt}, image_content[image]]
            if image_first:
                content.reverse()
            started = time.perf_counter()
            response = session.post(base + "/v1/chat/completions", json={
                "model": data["model"], "messages": [{"role": "user", "content": content}],
                "temperature": 0, "max_tokens": 36 if options is None else 8,
            }, timeout=180)
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            reply = choice["message"]["content"].strip()
            events = [json.loads(line) for line in trace_file.readlines()]
            events = [event for event in events if event["request_id"] == request_id]
            trace = compact_trace(events)
            matches = re.fullmatch(r"(?:答案[：:]?\s*)?\(?([ABCD])\)?[.。]?", reply, flags=re.I)
            result = {
                "request_id": request_id, "question": prompt, "image": image, "image_first": image_first,
                "reset_image_kv": reset, "keep_embedding": keep_embedding,
                "text": reply, "letter": matches.group(1).upper() if matches else None,
                "trace": trace, "usage": payload.get("usage"), "logprobs": choice.get("logprobs"),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            data["requests"][request_id] = result
            print(f"{request_id} {reply!r} embedding={trace['embedding_hits']} KV={trace['reads']} mask={trace['recomputed']}/{trace['total']}", flush=True)
            return result

        def pair(case, cold, warm):
            assert cold["question"] == warm["question"]
            assert cold["image"] == warm["image"] and cold["image_first"] == warm["image_first"]
            assert cold["trace"]["embedding_hits"] == warm["trace"]["embedding_hits"] == 1
            assert cold["trace"]["reads"] == 0 and cold["trace"]["reused"] == 0
            assert warm["trace"]["reads"] == 72 and warm["trace"]["layers"] == 36
            assert warm["trace"]["reused"] > 0
            return {
                "cold": cold["text"], "warm": warm["text"], "cold_id": cold["request_id"],
                "warm_id": warm["request_id"], "trace": warm["trace"],
                "cold_correct": cold["letter"] == case["answer"], "warm_correct": warm["letter"] == case["answer"],
            }

        def check(case, layout_first=False, cold_first=False):
            image = case["image"]
            label = case["id"]
            if cold_first:
                cold = ask(label + "_recompute", image, case["question"], case["options"], reset=True,
                           image_first=layout_first)
            seed = ask(label + "_seed", image, case["main"], reset=True, image_first=layout_first)
            warm = ask(label + "_reuse", image, case["question"], case["options"], image_first=layout_first)
            if not cold_first:
                cold = ask(label + "_recompute", image, case["question"], case["options"], reset=True,
                           image_first=layout_first)
            result = pair(case, cold, warm)
            result.update(seed_id=seed["request_id"], seed=seed["text"])
            return result

        try:
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}. See {output / 'server.log'}")
                try:
                    if session.get(base + "/health", timeout=2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("Server not ready")
            config = json.loads(args.cases.read_text())
            for info in config:
                path = Path(info["path"])
                original = Image.open(path).convert("RGB")
                submitted = original.copy()
                submitted.thumbnail((args.max_edge, args.max_edge), Image.Resampling.LANCZOS)
                png = io.BytesIO()
                submitted.save(png, format="PNG")
                png_bytes = png.getvalue()
                image = info["id"]
                metadata = {key: info[key] for key in ["id", "label", "path"]} | {
                    "width": original.width, "height": original.height,
                    "submitted_size": list(submitted.size), "input_sha256": hashlib.sha256(png_bytes).hexdigest(),
                }
                image_content[image] = {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(png_bytes).decode()
                }}
                if args.verify_results:
                    expected = next(item for item in data["images"] if item["id"] == image)
                    assert metadata["input_sha256"] == expected["input_sha256"]
                    continue
                data["images"].append(metadata)
                seed = ask(image + "_main", image, info["main"], reset=True, keep_embedding=False)
                warms = {}
                for detail in info["details"]:
                    case_id = f"{image}_{detail['id']}"
                    warms[case_id] = ask(case_id + "_reuse", image, detail["question"], detail["options"])
                    assert warms[case_id]["trace"]["writes"] == 0
                for detail in info["details"]:
                    case_id = f"{image}_{detail['id']}"
                    cold = ask(case_id + "_recompute", image, detail["question"], detail["options"], reset=True)
                    case = detail | {"id": case_id, "image": image, "main": info["main"], "seed": seed["text"],
                                     "seed_id": seed["request_id"]}
                    case.update(pair(case, cold, warms[case_id]))
                    data["cases"].append(case)
                    save()
            candidates = [case for case in data["cases"] if case["cold_correct"] and not case["warm_correct"]
                          and data["requests"][case["warm_id"]]["letter"] is not None]
            # Confirm multiple photographs before spending remaining repetitions on one photograph.
            chosen = []
            for case in candidates:
                if not any(item["image"] == case["image"] for item in chosen):
                    chosen.append(case)
            chosen.extend(case for case in candidates if case not in chosen)
            for case in chosen[:args.verify]:
                if args.verify_results:
                    ask(case["id"] + "_prime", case["image"], case["main"], reset=True)
                case["repeats"] = [check(case, cold_first=True), check(case)]
                cold = data["requests"][case["repeats"][-1]["cold_id"]]
                same_question = ask(case["id"] + "_same_question", case["image"], case["question"], case["options"])
                case["same_question"] = pair(case, cold, same_question)
                case["image_first"] = check(case, layout_first=True)
                save()
            print(f"Completed {len(data['cases'])} case pairs, {len(candidates)} candidate regressions.", flush=True)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            server.wait()


if __name__ == "__main__":
    main()
