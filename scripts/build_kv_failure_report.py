#!/usr/bin/env python3
"""Render the measured KV reuse experiment as a standalone HTML report."""

import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = {
    "market_mask": "画面右侧、前景女子背后，白底黑点头巾下露出一只浅蓝色口罩。黑色出现在头巾斑点和附近物体上。模型选黑色的原因仍需通过注意力测量或干预验证。",
    "market_phone_case": "白色塑料袋旁露出的手机壳有清楚的棕黑格纹。它被手和塑料袋遮住一部分，占整张图的面积很小。",
    "market_phone_position": "格纹手机位于白色塑料袋提手左上方。两组的不同回答是一处左右关系翻转。",
    "cafe_lights": "货架及大菜单下沿共有三个清楚可见的白色圆盘形吊灯，从左到右在货架、大菜单和条纹墙柱附近。",
    "football_right_jersey": "右侧张开双臂、背对镜头的球员号码是 6。本题恰好由重算时答错变为复用时答对，报告保留这一反向结果。",
}


def roi_counts(case):
    x, y, width, height = case["box"]
    trace = case["trace"]
    rows, cols = trace["rows"], trace["cols"]
    selected = [
        recomputed for index, recomputed in enumerate(trace["mask"])
        if x <= (index % cols + 0.5) / cols <= x + width
        and y <= (index // cols + 0.5) / rows <= y + height
    ]
    return {"recomputed": sum(selected), "reused": len(selected) - sum(selected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / ".runtime/kv_failure/results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    parser.add_argument("--featured", help="Override selection with a visually checked case ID")
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    requests = data["requests"]
    cases = data["cases"]
    referenced_requests = set()
    for case in cases:
        case["cold_trace"] = requests[case["cold_id"]]["trace"]
        case["roi_counts"] = roi_counts(case)
        case["note"] = OBSERVATIONS.get(case["id"], "")
        if case["image"] in {"football", "cafe"}:
            case["seed_correct"] = True
        controls = [case.get("image_first", {}), case.get("same_question", {})]
        for comparison in [case, *case.get("repeats", []), *controls]:
            for field in ("seed_id", "cold_id", "warm_id"):
                if field in comparison:
                    referenced_requests.add(comparison[field])
    data["requests"] = {key: value for key, value in requests.items() if key in referenced_requests}
    regressions = [case for case in cases if case["cold_correct"] and not case["warm_correct"]]
    complete = [
        case for case in regressions
        if case.get("repeats") and all(r["cold"] == case["cold"] and r["warm"] == case["warm"] for r in case["repeats"])
    ]
    featured = next((case for case in cases if case["id"] == args.featured), None)
    if args.featured and featured is None:
        raise ValueError(f"Unknown featured case: {args.featured}")
    if featured is None:
        featured = (complete or regressions or cases)[0]
    data["featured"] = featured["id"]
    data["model_notes"] = "Qwen3-VL-8B-Instruct · BF16 · temperature=0 · 单请求"
    data["methodology"] = (
        "每张图先发送主体题（文字在前、图片在后），建立跨请求图像 KV。随后用相同的图片、"
        "细节问题和选项测试两种条件。复用组沿用主体题留下的图像 KV，"
        "重算组先清除图像 KV 再答题。两组都命中同一图的外层视觉嵌入缓存。"
        "测试关闭 Radix 前缀缓存及 CUDA Graph，单请求运行，temperature=0。"
        "模型输入按最长边 1344 像素缩放后再由服务预处理。标准答案按原图中的可见证据标注，"
        "细节题只允许回复一个选项字母。市场主体回答识别了菜市场和蔬菜场景，但附带的“露天”等推断不作为正确性依据。"
        "复测从主体题重新建缓存，交换冷算与复用的执行顺序，并补做同题建缓存及图片置前两项对照。"
    )
    data["verification"] = (
        f"探索性筛查含 {len(data['images'])} 张照片、{len(cases)} 道题。"
        f"发现 {len(regressions)} 道由正确变错误，其中 {len(complete)} 道完成两次同条件复测且选项一致。"
        "本组题是为寻找反例手工设计的，结果适用于这次输入、模型和重算策略。"
        "图层展示预填充时哪些视觉 token 重算或复用。模型内部的注意力权重未采集，"
        "所以无法从位置重合推断哪片像素导致选项翻转。"
    )
    data["source_links"] = [
        {"label": "VLCache 论文 §3.1–3.2 · 前缀不匹配与重用误差", "url": "https://arxiv.org/html/2512.12977v2#S3"},
        {"label": "SceneCache 源码 · 图像哈希查找与固定位置重算", "url": "https://github.com/hebutBryant/SceneCache/blob/9c2935359cd53d64c53a2a8a48a0728c232b1c4e/python/sglang/srt/layers/attention/flashinfer_backend.py#L1137"},
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = args.output / "kv_failure_evidence.json"
    evidence.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    for image in data["images"]:
        image["src"] = "data:image/jpeg;base64," + base64.b64encode(Path(image["path"]).read_bytes()).decode("ascii")
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = (ROOT / "scripts/kv_failure_report.template.html").read_text()
    assert template.count("__REPORT_DATA__") == 1
    report = args.output / "kv_failure_report.html"
    report.write_text(template.replace("__REPORT_DATA__", encoded))
    print(f"Featured: {featured['id']} | {len(regressions)} regressions, {len(complete)} repeated")
    print(report)
    print(evidence)


if __name__ == "__main__":
    main()
