#!/usr/bin/env python3
"""Build a self-contained HTML report from the measured market-photo experiment."""

import base64
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def data_url(path, mime):
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main():
    runtime = ROOT / ".runtime/market_demo"
    data = json.loads((runtime / "results.json").read_text())
    requests = {r["id"]: r for r in data["requests"]}
    grid_event = next(e for e in requests["B_new_question_reuse"]["trace"] if e["event"] == "prefill")
    thw = grid_event["image_grids"][0]
    grid = {
        "thw": thw, "rows": thw[1] // 2, "cols": thw[2] // 2,
        "tokens": thw[0] * thw[1] * thw[2] // 4,
        "processed_size": [thw[2] * 16, thw[1] * 16],
        "intermediate_size": [round(size / 28) * 28 for size in data["submitted_size"]],
    }
    data["grid"] = grid
    data["stages"] = {}
    names = {"A": "A_first_question_cold", "B": "B_new_question_reuse",
             "C": "C_new_question_cold", "D": "D_same_question_reuse"}
    for key, name in names.items():
        row = dict(requests[name])
        prefill = next(e for e in row["trace"] if e["event"] == "prefill")
        start, end = prefill["image_ranges"][0]
        assert end - start + 1 == grid["tokens"]
        masks = prefill["first_layer_compute_mask"]
        image_mask = masks[start:end + 1] if masks is not None else [True] * grid["tokens"]
        row.update({
            "prefill": prefill, "image_mask": image_mask,
            "recomputed": sum(image_mask), "reused": len(image_mask) - sum(image_mask),
            "text_tokens": prefill["total_input_tokens"] - grid["tokens"],
            "computed_per_layer": [prefill["computed_tokens_per_layer"].get(str(layer), prefill["total_input_tokens"]) for layer in range(36)],
        })
        assert len(set(row["computed_per_layer"])) == 1
        data["stages"][key] = row
    cold = data["stages"]["A"]
    feature = next(e for e in cold["trace"] if e["event"] == "write_kv" and e["kind"] == "vision")
    kv = [e for e in cold["trace"] if e["event"] == "write_kv" and e["kind"] == "kv"]
    assert len(kv) == 72
    data["storage"] = {
        "feature_bytes": feature["bytes"], "feature_shape": feature["shape"],
        "kv_bytes": sum(e["bytes"] for e in kv),
        "host_bytes": sum(e["bytes"] for e in kv) + feature["bytes"],
    }
    assert data["stages"]["B"]["vision_embedding_hits"] == 1
    assert data["stages"]["B"]["cache_reads"]["kv"] == 72
    assert data["stages"]["C"]["cache_reads"]["kv"] == 0
    assert data["stages"]["B"]["recomputed"] == max(1, int(grid["tokens"] * 0.3))
    assert data["stages"]["C"]["text"] == data["stages"]["D"]["text"]
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    with Image.open(runtime / "input.png") as image:
        image = image.resize(tuple(grid["intermediate_size"]), Image.Resampling.BICUBIC)
        image.resize(tuple(grid["processed_size"]), Image.Resampling.BICUBIC).save(runtime / "actual_grid_preview.jpg", quality=92)
    template = (ROOT / "scripts/market_cache_report.template.html").read_text()
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html = template.replace("__REPORT_JSON__", encoded)
    html = html.replace("__ORIGINAL_IMAGE__", data_url(Path(data["image_path"]), "image/jpeg"))
    html = html.replace("__PROCESSED_IMAGE__", data_url(runtime / "actual_grid_preview.jpg", "image/jpeg"))
    assert "__REPORT_JSON__" not in html and "__PROCESSED_IMAGE__" not in html
    (output / "market_cache_report.html").write_text(html)
    (output / "market_cache_evidence.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print("Report:", output / "market_cache_report.html")
    print("Visual tokens:", grid["tokens"], "recomputed:", data["stages"]["B"]["recomputed"], "reused:", data["stages"]["B"]["reused"])
    print("Feature MiB:", feature["bytes"] / 1048576, "KV MiB:", data["storage"]["kv_bytes"] / 1048576)


if __name__ == "__main__":
    main()
