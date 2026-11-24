"""Render an offline HTML report from a locked terminal-state experiment."""

import argparse
import base64
import io
import json
import os
import re
from collections import defaultdict
from html import escape
from pathlib import Path
from statistics import median
from urllib.parse import quote

from PIL import Image


MODES = {
    "full_history": "A · 完整历史",
    "terminal_kv_slice": "B · 末端 KV 截取",
    "tool_result_cache": "C · 工具结果缓存",
    "canonical_state_kv": "D · 重编码终态 KV",
}
PATHS = {"autonomous": "自主采集", "controlled_tool_replay": "受控裁剪回放"}
STATUSES = {
    "answered": "已回答",
    "needs_evidence": "请求更多证据",
    "unverified": "未验证",
    "incomplete": "生成未完成",
    "not_run": "未执行",
    "error": "执行失败",
    "complete": "已完成",
    "running": "运行中",
    "in_progress": "进行中",
    "model_stopped_with_answer": "模型生成了回答",
    "model_requested_crop": "模型选择的裁剪",
    "whole_frame_original_evidence": "原图证据",
    "no_successful_crop": "未取得有效裁剪",
    "turn_limit": "到达轮次上限",
    "first_locked_sample_did_not_complete_required_checks": "首个锁定样本未通过必要检查",
}
CSS = """
:root { color-scheme: light; --ink: #172f38; --muted: #50646c; --page: #edf3f2;
  --paper: #fff; --line: #cbdad7; --accent: #0e665d; --pass: #166443; --fail: #9f3828; }
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: var(--page); font: 15px/1.65
  system-ui, "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; }
.wrap { width: min(1170px, calc(100% - 40px)); margin: 0 auto; }
header { padding: 42px 0 36px; background: #123f3a; color: #f3fbf9; }
header p { color: #cdeae4; }
h1 { font-size: clamp(25px, 5vw, 42px); line-height: 1.18; margin: 0 0 12px; }
h2 { font-size: 23px; line-height: 1.3; margin: 0 0 10px; }
h3 { font-size: 17px; margin: 20px 0 7px; }
h4 { font-size: 15px; margin: 12px 0 6px; }
p { margin: 6px 0 10px; }
main { padding: 22px 0 40px; }
section { margin-bottom: 18px; padding: 22px; background: var(--paper); border: 1px solid var(--line);
  border-radius: 12px; min-width: 0; scroll-margin-top: 12px; }
a { color: var(--accent); text-underline-offset: .18em; }
:focus-visible { outline: 3px solid #c26b23; outline-offset: 3px; }
nav, .links { display: flex; gap: 8px 20px; flex-wrap: wrap; }
nav { margin-bottom: 16px; }
.muted { color: var(--muted); }
.scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
td, th { padding: 7px 9px; border-bottom: 1px solid var(--line); text-align: left; }
thead { background: #edf5f3; }
th { font-weight: 650; white-space: nowrap; }
.gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(245px, 100%), 1fr)); gap: 13px; }
figure { margin: 0; border: 1px solid var(--line); border-radius: 7px; overflow: hidden; background: #f8faf9; }
figure img { display: block; width: 100%; height: clamp(140px, 25vw, 270px); object-fit: contain; background: #e6eeec; }
figcaption { padding: 8px 10px; }
figure a { display: block; }
details { border: 1px solid var(--line); margin: 8px 0; border-radius: 8px; background: #fcfdfd; }
details[open] > summary { border-bottom: 1px solid var(--line); background: #edf5f3; }
summary { cursor: pointer; padding: 8px 11px; font-weight: 640; overflow-wrap: anywhere; }
details > div { padding: 10px 12px; min-width: 0; }
pre { margin: 6px 0; padding: 11px; white-space: pre-wrap; overflow-wrap: anywhere;
  border-radius: 5px; background: #edf3f2; color: var(--ink); font: 13px/1.55 ui-monospace, monospace; }
code { overflow-wrap: anywhere; }
.question { padding-left: 13px; border-left: 3px solid var(--accent); font-weight: 650; }
.good { color: var(--pass); }
.bad { color: var(--fail); }
.metric { font-variant-numeric: tabular-nums; color: var(--muted); }
.mode { font-weight: 650; white-space: nowrap; }
@media (max-width: 640px) { .wrap { width: calc(100% - 20px); } section { padding: 14px; } header { padding: 26px 0; } }
@media print { header { background: #fff; color: #172f38; } header p { color: #172f38; } body { background: white; } }
"""


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def show(value):
    if value is None or value == "":
        return "—"
    return escape(str(value))


def status(value):
    return show(STATUSES.get(value, value))


def number(value, precision=2):
    return f"{value:,.{precision}f}" if isinstance(value, (int, float)) else "—"


def seconds_ms(value):
    return value * 1000 if value is not None else None


def bytes_mib(value):
    return value / (1024 * 1024) if value is not None else None


def load_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def load_rows(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def request_key(row):
    return row["sample_key"], row.get("path", "autonomous"), row["mode"], row["request"]


def merge_request_traces(run_dir, rows):
    merged = {request_key(row): row for row in rows}
    sources = {}
    for path in sorted((run_dir / "samples").glob("*/*/request_traces/*.json")):
        recorded = load_json(path, {})
        key = request_key(recorded)
        summary = merged.get(key, {})
        if summary.get("generation"):
            merged[key] = {**recorded, **summary}
        else:
            merged[key] = recorded
        sources[key] = path
    return list(merged.values()), sources


def href(path, output):
    return escape(quote(os.path.relpath(path, output.parent), safe="/"), quote=True)


def link(path, label, output):
    if path.is_file():
        return f'<a href="{href(path, output)}">{escape(label)}</a>'
    return ""


def links(items, output):
    html = " ".join(link(path, label, output) for path, label in items if path.is_file())
    return f'<p class="links">{html}</p>' if html else ""


def disclosure(title, contents, *, css=""):
    return f'<details class="{css}"><summary>{title}</summary><div>{contents}</div></details>'


def pre(value):
    return f"<pre>{escape(value)}</pre>"


def json_disclosure(title, value, *, compact=False):
    if value is None:
        return ""
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":")) if compact else json_text(value)
    return disclosure(escape(title), pre(serialized))


def table(columns, rows):
    headers = "".join(f'<th scope="col">{column}</th>' for column in columns)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>'


def token_metrics(record):
    generation = record.get("generation") or {}
    count = generation.get("generated_tokens")
    after_first = generation.get("after_first_token_seconds")
    first = generation.get("first_token_seconds")
    elapsed = generation.get("generation_seconds")
    if after_first is None and first is not None and elapsed is not None:
        after_first = max(0, elapsed - first)
    return {
        "ttft": seconds_ms(first),
        "prefill": seconds_ms(generation.get("prefill_seconds")),
        "tpot": seconds_ms(after_first / (count - 1)) if count and count > 1 and after_first is not None else None,
        "generated": count,
        "kv_rows": generation.get("prefix_physical_tokens"),
        "kv_mib": bytes_mib(record.get("prefix_gpu_kv_bytes")),
        "vision": generation.get("vision_input_count"),
        "request": record.get("request_ms"),
    }


def middle(records, metric):
    values = [m[metric] for m in records if m[metric] is not None]
    return median(values) if values else None


def verdict(record):
    if record.get("matches_reference") is True:
        return '<span class="good">匹配标注</span>'
    if record.get("matches_reference") is False:
        return '<span class="bad">未匹配标注</span>'
    return '<span class="muted">未评分</span>'


def summary_tables(rows):
    by_path = defaultdict(list)
    for row in rows:
        by_path[row.get("path", "autonomous")].append(row)
    sections = []
    for path, path_rows in by_path.items():
        scored = []
        for mode, label in MODES.items():
            mode_rows = [row for row in path_rows if row.get("mode") == mode]
            if not mode_rows:
                continue
            executed = [row for row in mode_rows if row.get("generation")]
            metrics = [token_metrics(row) for row in executed]
            references = [row["matches_reference"] for row in mode_rows if row.get("matches_reference") is not None]
            scored.append([
                escape(label), f"{len(executed)} / {len(mode_rows)}",
                f"{sum(references)} / {len(references)}" if references else "—",
                *[number(middle(metrics, key)) for key in ("ttft", "prefill", "tpot")],
                number(middle(metrics, "kv_rows"), 1), number(middle(metrics, "kv_mib")),
            ])
        sections.append(
            f"<h3>{show(PATHS.get(path, path))}</h3>" + table(
                ["模式", "执行 / 记录", "标注匹配 / 已评分", "TTFT（ms）", "Prefill（ms）",
                 "TPOT（ms）", "复用 KV 行数", "GPU KV（MiB）"], scored
            )
        )
    return "".join(sections) or '<p class="muted">当前没有问答记录。</p>'


def preview(path):
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((1050, 850), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def photo(observation, path_dir, output, label):
    name = observation.get("file")
    if not name:
        return ""
    image = path_dir / Path(name).name
    if not image.is_file():
        return f'<p class="muted">{escape(label)}的图像文件缺失。</p>'
    size = observation.get("size_wh") or []
    dimensions = " × ".join(str(dimension) for dimension in size)
    box = observation.get("bbox_original_xyxy")
    coord = f" · 原图坐标 {show(box)}" if box is not None else ""
    return (
        f'<figure><a href="{href(image, output)}" title="打开原尺寸 PNG">'
        f'<img src="{preview(image)}" alt="{escape(label, quote=True)}" loading="lazy"></a>'
        f'<figcaption>{escape(label)} · {escape(dimensions)}{coord}</figcaption></figure>'
    )


def model_output(record):
    output = record.get("generated_text")
    generated_ids = record.get("generated_ids")
    if output is None:
        return '<p class="muted">此条记录没有已保存的模型生成文本。</p>'
    parts = []
    if prefix := record.get("qa_assistant_prefix"):
        parts.append(f"<p>固定输出前缀 <code>{escape(prefix)}</code></p>")
    if not re.search(r"<think(?:\s[^>]*)?>", output):
        parts.append('<p class="muted">本次输出未包含 <code>&lt;think&gt;</code> 标签。</p>')
    parts.append(pre(output) if output else "<p>模型生成文本为空。</p>")
    if generated_ids is not None:
        parts.append(json_disclosure("生成 token ID", generated_ids, compact=True))
    return "".join(parts)


def metric_line(metrics):
    return (
        '<p class="metric">'
        f'TTFT {number(metrics["ttft"])} ms · Prefill {number(metrics["prefill"])} ms'
        f' · TPOT {number(metrics["tpot"])} ms · 输出 token {show(metrics["generated"])}'
        f' · 复用 KV {show(metrics["kv_rows"])} 行</p>'
    )


def render_turn(record, observations, *, sealed_lookahead=False, turn_limit=False):
    metrics = token_metrics(record)
    observation = observations.get(record.get("observation_id"))
    heading = f'第 {show(record.get("turn"))} 轮 · {show(metrics["generated"])} token'
    content = [metric_line(metrics)]
    if sealed_lookahead and "<answer>" in record.get("generated_text", ""):
        content.append(
            '<p class="muted">本轮生成的最终回答保留于采集轨迹。'
            '封存的可复用前缀截至终态观察，回答 token 已排除。</p>'
        )
    if error := record.get("tool_failure"):
        content.append(f'<p class="bad">工具错误 <code>{escape(error)}</code></p>')
    if action := record.get("action"):
        content.append("<h4>模型发出的工具参数</h4>" + pre(json_text(action)))
    elif turn_limit and "<grounding>" in record.get("generated_text", ""):
        content.append(
            '<p class="muted">模型提出了新的裁剪请求。已到达采集轮次上限，工具未执行。'
            '请求参数保留在生成原文中。</p>'
        )
    if observation is not None:
        observation_id = observation["observation_id"]
        content.append(
            f'<p>工具返回观察 {show(observation_id)} · 父观察 '
            f'{show(observation.get("parent_observation_id"))} · 原图坐标 '
            f'<code>{show(observation.get("bbox_original_xyxy"))}</code></p>'
        )
    content.append("<h4>模型生成原文</h4>" + model_output(record))
    content.append(json_disclosure("完整逻辑输入 token、图像 span 与位置", record.get("input"), compact=True))
    return disclosure(heading, "".join(content), css="trace-turn")


def original_request(row):
    return "原题" if row.get("request") == "original" else f'追问 · {show(row.get("request"))}'


def render_request(row, source, output):
    label = MODES.get(row.get("mode"), row.get("mode"))
    heading = f'{show(label)} · {original_request(row)} · {status(row.get("status"))}'
    bits = [
        f'<p class="question">{show(row.get("question"))}</p>',
        f'<p>标注答案 <strong>{show(row.get("reference"))}</strong> · 模型回答 '
        f'<strong>{show(row.get("answer"))}</strong> · {verdict(row)}</p>',
    ]
    if row.get("generation"):
        metrics = token_metrics(row)
        bits.append(metric_line(metrics))
        bits.append(
            f'<p class="metric">GPU 前缀 KV {number(metrics["kv_mib"])} MiB · 本次视觉编码次数 '
            f'{show(metrics["vision"])} · 单次请求 {number(metrics["request"])} ms</p>'
        )
    if note := row.get("reason") or row.get("fallback"):
        bits.append(f'<p class="muted">{escape(note)}</p>')
    bits.append("<h4>模型生成原文</h4>" + model_output(row))
    bits.append(json_disclosure("完整逻辑输入 token、图像 span 与位置", row.get("input"), compact=True))
    if source is not None:
        bits.append(links([(source, "单次原始推理 JSON")], output))
    return disclosure(heading, "".join(bits), css="qa-request")


def render_question_table(rows):
    by_request = defaultdict(list)
    for row in rows:
        by_request[(row.get("request"), row.get("question"))].append(row)
    output = []
    for group in by_request.values():
        first = group[0]
        output.append(
            f'<h4>{original_request(first)}</h4><p class="question">{show(first.get("question"))}</p>'
            f'<p>标注答案 <strong>{show(first.get("reference"))}</strong></p>'
        )
        output.append(table(
            ["模式", "回答", "评分", "TTFT（ms）", "Prefill（ms）", "TPOT（ms）", "GPU KV（MiB）"],
            [
                [
                    show(MODES.get(row.get("mode"), row.get("mode"))), show(row.get("answer")),
                    verdict(row), *[number(metrics[key]) for key in ("ttft", "prefill", "tpot", "kv_mib")],
                ]
                for row in group
                for metrics in (token_metrics(row),)
            ],
        ))
    return "".join(output)


def render_path(path, sample_dir, rows, request_sources, output):
    directory = sample_dir / path
    trace_path = directory / "trace.json"
    state_path = directory / "state.json"
    metrics_path = directory / "path_metrics.json"
    trace = load_json(trace_path, {})
    state_data = load_json(state_path, {})
    metrics = load_json(metrics_path, {})
    title = PATHS.get(path, path)
    records = trace.get("turns", [])
    observations = {item["observation_id"]: item for item in trace.get("observations", [])}
    boundary = trace.get("terminal_boundary") or {}
    terminal_id = (state_data.get("source_observation") or {}).get("observation_id")
    if terminal_id is None:
        terminal_id = boundary.get("observation_id")
    operation = trace.get("terminal_status")
    bits = [
        f"<h3>{escape(title)}</h3>",
        links([(trace_path, "完整采集轨迹 JSON"), (state_path, "终态 z JSON"),
               (metrics_path, "缓存与时延 JSON"), (directory / "canonical_prefix.json", "规范前缀 JSON"),
               (directory / "canonical_parity.json", "C/D 数值校验 JSON"),
               (directory / "terminal_slice_approximation.json", "A/B 近似对照 JSON")], output),
    ]
    if trace:
        bits.append(
            f'<p class="metric">采集状态 {status(trace.get("status"))} · 终态 {status(operation)}'
            f' · 采集耗时 {number(trace.get("elapsed_ms"))} ms</p>'
        )
        photos = []
        for obs_id, obs in observations.items():
            label = "原图" if obs_id == 0 else f"工具裁剪 {obs_id}"
            if obs_id == terminal_id:
                label += " · 终态证据"
            photos.append(photo(obs, directory, output, label))
        if photos:
            bits.append('<div class="gallery">' + "".join(photos) + "</div>")
        bits.append(json_disclosure("初始 system 与 user 消息", trace.get("initial_messages")))
        if records:
            bits.append("<h4>采集阶段逐轮生成</h4>")
            bits.extend(
                render_turn(
                    record, observations,
                    sealed_lookahead=index == len(records) - 1 and boundary.get("answer_tokens_discarded", False),
                    turn_limit=index == len(records) - 1 and trace.get("status") == "turn_limit",
                )
                for index, record in enumerate(records)
            )
        if trace.get("actions"):
            bits.append(json_disclosure("受控回放的工具动作", trace["actions"]))
    if release := metrics.get("old_storage_release"):
        bits.append(
            f'<p class="metric">旧历史 KV {number(bytes_mib(release.get("old_storage_bytes")))} MiB'
            f' · 释放后的旧张量引用 {show(release.get("live_old_tensor_references"))}'
            f' · 进程 GPU 活跃分配减少 {number(bytes_mib(release.get("gpu_allocated_delta_bytes")))} MiB</p>'
        )
    if cost := metrics.get("cost_D_minus_C"):
        bits.append(
            f'<p class="metric">D − C 完整采集及问答 {number(cost.get("mode_episode_ms"))} ms'
            f' · 一次性转换 {number(cost.get("canonical_conversion_ms"))} ms'
            f' · 比较请求 {show(cost.get("requests_compared"))} 次</p>'
        )
    if state_data:
        bits.append(json_disclosure("终态 z 与证据来源", state_data))
    if rows:
        bits.extend(["<h4>A/B/C/D 问答结果</h4>", render_question_table(rows),
                     "<h4>展开每次请求的生成原文</h4>"])
        bits.extend(render_request(row, request_sources.get(request_key(row)), output) for row in rows)
    elif not trace:
        bits.append('<p class="muted">此路径没有已保存的采集或问答记录。</p>')
    else:
        bits.append('<p class="muted">此路径没有已保存的问答记录。</p>')
    return "".join(bits)


def render_sample(index, sample, directory, rows, request_sources, pairing, output):
    sample_error = load_json(directory / "error.json", {})
    title = f'样本 {index + 1} · {show(sample.get("difficulty"))}'
    sections = [
        f'<section id="sample-{index + 1}"><h2>{title}</h2>',
        f'<p class="muted">样本 ID <code>{show(sample.get("id"))}</code> · 按锁定清单展示</p>',
        f'<p class="question">{show(sample.get("question"))}</p>',
        f'<p>标注答案 <strong>{show(sample.get("answer"))}</strong></p>',
    ]
    if sample_error:
        sections.append(f'<p class="bad">执行错误 <code>{show(sample_error.get("reason"))}</code></p>')
    if pairing and pairing.get("status") != "paired":
        sections.append(
            f'<p class="muted">对照回放记录 <code>{show(pairing.get("reason") or pairing.get("status"))}</code></p>'
        )
    for path in PATHS:
        path_rows = [row for row in rows if row.get("path", "autonomous") == path]
        if path == "autonomous" or path_rows or (directory / path / "trace.json").is_file():
            sections.append(render_path(path, directory, path_rows, request_sources, output))
    sections.append("</section>")
    return "".join(sections)


def render(run_dir, output):
    manifest = load_json(run_dir / "selected_manifest.json", {})
    config = load_json(run_dir / "experiment_config.json", {})
    summary = load_json(run_dir / "summary.json", {})
    rows, request_sources = merge_request_traces(run_dir, load_rows(run_dir / "requests.jsonl"))
    rows_by_sample = defaultdict(list)
    for row in rows:
        rows_by_sample[row.get("sample_key")].append(row)
    pairs = {entry["sample_key"]: entry for entry in load_json(run_dir / "pairs.json", [])}
    samples = manifest.get("samples", [])
    paths = {}
    for directory in (run_dir / "samples").glob("*"):
        match = re.fullmatch(r"\d+-(.+)", directory.name)
        if match and directory.is_dir():
            paths[match[1]] = directory
    model = config.get("model", {}).get("checkpoint")
    gpu = config.get("environment", {}).get("gpu")
    traces = [load_json(folder / "autonomous" / "trace.json", {}) for folder in paths.values()]
    turn_count = sum(bool(turn.get("generation")) for trace in traces for turn in trace.get("turns", []))
    execution_count = sum(bool(row.get("generation")) for row in rows)
    dataset = manifest.get("dataset", {})
    missing = dataset.get("missing_splits", [])
    if not missing:
        missing = config.get("missing_visualprobe_splits", [])
    if not missing and dataset.get("layout") == "mini_o3_visualprobe_validation":
        available = {row.get("split_difficulty") for row in dataset.get("rows_files", [])}
        missing = sorted({"easy", "medium", "hard"} - available)
    elements = [
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Mini-o3 同图多问实验</title><style>' + CSS + '</style></head>',
        '<body data-server-no-reload><header><div class="wrap"><h1>Mini-o3 同图多问实验</h1>'
        f"<p>运行状态 {status(summary.get('status') or config.get('status'))} · 锁定样本 {len(samples)}"
        f" · 实际采集生成 {turn_count} 轮 · 实际问答生成 {execution_count} 次</p></div></header>",
        '<main class="wrap"><nav aria-label="样本目录"><a href="#overview">实验总览</a>',
    ]
    elements.extend(
        f'<a href="#sample-{index + 1}">样本 {index + 1} · {show(sample.get("difficulty"))}</a>'
        for index, sample in enumerate(samples)
    )
    elements.extend([
        '</nav><section id="overview"><h2>四种模式的实测结果</h2>',
        f'<p>数据集 <code>{show(config.get("dataset") or manifest.get("dataset", {}).get("layout"))}</code>'
        f' · GPU {show(gpu)} · 模型 <code>{show(model)}</code></p>',
        '<p class="muted">每个可展开的生成记录保留 Mini-o3 实际输出的文本。'
        '原文中的 <code>&lt;think&gt;</code> 标签由该模型生成。图像缩略图已内嵌，点击可打开对应的原尺寸 PNG。</p>',
        links(
            [(run_dir / name, label) for name, label in (
                ("selected_manifest.json", "样本锁定清单"),
                ("experiment_config.json", "运行配置"),
                ("requests.jsonl", "原始问答 JSONL"),
                ("requests.csv", "逐次问答 CSV"),
                ("summary.json", "自动汇总 JSON"),
                ("pairs.json", "对照回放记录"),
            )],
            output,
        ),
        f'<p class="bad">{show(config["reason"])}</p>' if config.get("reason") else "",
        f'<p class="muted">本次未纳入的 VisualProbe 难度 {show(", ".join(map(str, missing)))}</p>'
        if missing else "",
        json_disclosure("生成参数与运行环境", {
            "generation": config.get("generation"), "environment": config.get("environment"),
        }),
        summary_tables(rows),
        '<p class="muted">延迟、KV 行数和显存均取实际执行请求的中位数。'
        'TPOT 使用首 token 之后的耗时除以后续 token 数。'
        '标注按运行器的精确匹配评分。完整句子可能因字面差异被记为不匹配。追问缺少独立标注时显示为未评分。'
        '完整逻辑输入与实际复用的物理 KV 行数分别记录。B 实际复用的是旧 KV 的一部分。</p>',
        "</section>",
    ])
    for index, sample in enumerate(samples):
        key = sample["sample_key"]
        directory = paths.get(key, run_dir / "samples" / f"{index:02d}-{key}")
        elements.append(
            render_sample(index, sample, directory, rows_by_sample[key], request_sources, pairs.get(key), output)
        )
    elements.append("</main></body></html>")
    return "\n".join(elements) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    if not run_dir.is_dir():
        parser.error(f"Run directory does not exist: {run_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(render(run_dir, output), encoding="utf-8")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
