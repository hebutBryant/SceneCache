#!/usr/bin/env python3
"""Collect actual SceneCache traces for the supplied market image on a demo GPU."""

import argparse
import base64
import datetime
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL = "/home/hdd4/lpz/Qwen3-VL-8B-Instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="/home/lipz/test_market.jpg")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--pairs", type=int, default=4)
    args = parser.parse_args()
    root = ROOT / ".runtime/market_demo"
    root.mkdir(parents=True, exist_ok=True)
    trace_path = root / "trace.jsonl"
    trace_path.write_text("")
    (root / "bootstrap").mkdir(exist_ok=True)
    (root / "bootstrap/sitecustomize.py").write_text(
        "from scenecache_report_trace import install\ninstall()\n"
    )
    with Image.open(args.image) as original:
        original_size = list(original.size)
        image = original.convert("RGB")
        image.thumbnail((1024, 1024))
        image.save(root / "input.png")
        submitted_size = list(image.size)

    env = dict(os.environ)
    env.update({
        "SCENECACHE_REPORT_DIR": str(root),
        "PYTHONPATH": os.pathsep.join([str(root / "bootstrap"), str(ROOT / "scripts"), str(ROOT / "python")]),
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_HOME": "/usr/local/cuda-12.8",
        "CUDACXX": "/usr/local/cuda-12.8/bin/nvcc",
        "CUDA_TOOLKIT_ROOT_DIR": "/usr/local/cuda-12.8",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.8/lib64:" + env.get("LD_LIBRARY_PATH", ""),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "4", "SGLANG_CPU_WORKERS": "2", "MAX_JOBS": "4",
        "FLASHINFER_WORKSPACE_BASE": str(ROOT / ".runtime/flashinfer"),
        "TRITON_CACHE_DIR": str(ROOT / ".runtime/triton"),
    })
    env["PATH"] = "/usr/local/cuda-12.8/bin:" + env["PATH"]
    command = [
        sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL,
        "--served-model-name", "Qwen3-VL-8B-Instruct", "--host", "127.0.0.1",
        "--port", str(args.port), "--dtype", "bfloat16",
        "--attention-backend", "flashinfer", "--mm-attention-backend", "triton_attn",
        "--sampling-backend", "pytorch", "--mem-fraction-static", "0.95",
        "--max-total-tokens", "8192", "--context-length", "4096",
        "--max-running-requests", "1", "--chunked-prefill-size", "8000",
        "--disable-cuda-graph", "--disable-radix-cache", "--trust-remote-code",
        "--random-seed", "42",
    ]
    session = requests.Session()
    session.trust_env = False
    base = f"http://127.0.0.1:{args.port}"
    encoded = base64.b64encode((root / "input.png").read_bytes()).decode()
    image_content = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}
    data = {
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "image_path": args.image, "original_size": original_size,
        "submitted_size": submitted_size, "model_path": MODEL,
        "gpu": args.gpu, "max_tokens": 96, "temperature": 0,
        "recompute_ratio": 0.3, "requests": [], "command": command,
    }

    def request(label, question, reset=False, purpose="demo"):
        control = {"request_id": label, "reset": reset}
        temporary = root / "control.tmp"
        temporary.write_text(json.dumps(control))
        temporary.replace(root / "control.json")
        started = time.perf_counter()
        response = session.post(base + "/v1/chat/completions", json={
            "model": "Qwen3-VL-8B-Instruct",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": question}, image_content,
            ]}], "temperature": 0, "max_tokens": 96,
            "stream": True, "stream_options": {"include_usage": True},
        }, stream=True, timeout=180)
        response.raise_for_status()
        text_parts, usage, finish = [], None, None
        first = None
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data: "):
                continue
            raw = line[6:]
            if raw == b"[DONE]":
                break
            chunk = json.loads(raw)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if text:
                    if first is None:
                        first = time.perf_counter() - started
                    text_parts.append(text)
                finish = choice.get("finish_reason") or finish
        elapsed = time.perf_counter() - started
        result = {"id": label, "question": question, "purpose": purpose,
                  "cache_reset": reset, "text": "".join(text_parts),
                  "ttft_ms": round(first * 1000, 3) if first is not None else None,
                  "elapsed_ms": round(elapsed * 1000, 3), "usage": usage,
                  "finish_reason": finish}
        if not result["text"]:
            raise RuntimeError(f"Empty response: {label}")
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        events = [event for event in events if event["request_id"] == label]
        result["trace"] = events
        reads = [event for event in events if event["event"] == "get_kv"]
        writes = [event for event in events if event["event"] == "write_kv"]
        result["cache_reads"] = {kind: sum(e["kind"] == kind for e in reads) for kind in ["vision", "kv"]}
        result["cache_writes"] = {kind: sum(e["kind"] == kind for e in writes) for kind in ["vision", "kv"]}
        result["cache_write_bytes"] = sum(event["bytes"] for event in writes)
        result["vision_embedding_hits"] = sum(e["event"] == "embedding_cache_get" and e["hit"] for e in events)
        data["requests"].append(result)
        (root / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({key: result[key] for key in ["id", "text", "ttft_ms", "elapsed_ms", "cache_reads", "cache_writes"]}, ensure_ascii=False), flush=True)

    log = (root / "server.log").open("w")
    server = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, start_new_session=True)
    (root / "server.pid").write_text(str(server.pid) + "\n")
    try:
        print(f"Demo server PID={server.pid}; GPU={args.gpu}; port={args.port}", flush=True)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"Demo server exited: {server.returncode}; inspect {root / 'server.log'}")
            try:
                if session.get(base + "/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise RuntimeError("Demo server startup timeout")
        q1 = "请用两句话描述这张市场图片中的主要人物和蔬菜，只描述能直接看见的内容。"
        q2 = "图片中右侧前景戴米色头巾的女子正在与左侧男子进行什么互动？请只用一句话回答，不要猜测地点或身份。"
        request("warmup", q1, reset=True, purpose="warmup")
        request("A_first_question_cold", q1, reset=True)
        request("B_new_question_reuse", q2)
        request("C_new_question_cold", q2, reset=True)
        request("D_same_question_reuse", q2)
        for index in range(args.pairs):
            request(f"bench_{index}_cold", q2, reset=True, purpose="benchmark")
            request(f"bench_{index}_warm", q2, purpose="benchmark")
        summary = {}
        for condition in ["cold", "warm"]:
            rows = [r for r in data["requests"] if r["purpose"] == "benchmark" and r["id"].endswith(condition)]
            summary[condition] = {metric: statistics.median(r[metric] for r in rows) for metric in ["ttft_ms", "elapsed_ms"]}
            summary[condition]["n"] = len(rows)
        data["benchmark"] = summary
        (root / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        print("Complete: " + str(root / "results.json"), flush=True)
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=12)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        log.close()


if __name__ == "__main__":
    main()
