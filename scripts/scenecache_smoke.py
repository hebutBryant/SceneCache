#!/usr/bin/env python3
"""Exercise local text generation and the SceneCache image reorder workflow.

Usage: conda run -n scenecache python scripts/scenecache_smoke.py
The two images are bundled in the repository; no external image URLs are used.
Results are written to .runtime/smoke_results.json.
"""

import argparse
import base64
import io
import json
import time
from pathlib import Path

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def image_content(path):
    # Limit visual tokens and activation memory for this shared A6000.
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((448, 448))
        data = io.BytesIO()
        image.save(data, format="PNG")
    encoded = base64.b64encode(data.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    session = requests.Session()
    session.trust_env = False
    base_url = args.base_url.rstrip("/")
    health = session.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()
    models = session.get(f"{base_url}/v1/models", timeout=10)
    models.raise_for_status()
    model_name = models.json()["data"][0]["id"]
    results = {"base_url": base_url, "model": model_name, "requests": []}
    output_path = ROOT / ".runtime/smoke_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def complete(label, content):
        started = time.perf_counter()
        response = session.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": 96,
            },
            timeout=args.timeout,
        )
        if not response.ok:
            raise RuntimeError(f"{label}: HTTP {response.status_code}: {response.text[:2000]}")
        body = response.json()
        text = body["choices"][0]["message"]["content"]
        if not text or not text.strip():
            raise RuntimeError(f"{label}: empty completion: {body}")
        result = {
            "label": label,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "text": text,
            "usage": body.get("usage"),
            "finish_reason": body["choices"][0].get("finish_reason"),
        }
        results["requests"].append(result)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)

    complete("text", "请用一句话介绍你自己。")
    directory = ROOT / "examples/frontend_language/quick_start/images"
    cat = image_content(directory / "cat.jpeg")
    dog = image_content(directory / "dog.jpeg")
    prompt = {"type": "text", "text": "请按图片顺序，分别用一句中文描述每张图片。"}
    complete("images_cat_dog", [prompt, cat, dog])
    complete("images_dog_cat_reuse", [prompt, dog, cat])
    print(f"All three requests succeeded. Results: {output_path}")


if __name__ == "__main__":
    main()
