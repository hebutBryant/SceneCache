"""Render a portable view of masks, token footprints, and contact edges."""

import base64
import io
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image

_PALETTE = (
    "#f36d48", "#45be87", "#49a9ea", "#f4bc45",
    "#ae8be8", "#d968a6", "#61c7ca", "#e6df59",
)


def _png_url(image):
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    return "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")


def _viewer_data(memory):
    width, height = memory.geometry.original_size
    regions = []
    for index, region in enumerate(memory.regions):
        color = _PALETTE[index % len(_PALETTE)]
        mask = memory.masks[region.region_id]
        overlay = Image.new("RGB", (width, height), color)
        overlay.putalpha(Image.fromarray(mask.astype(np.uint8) * 116))
        indices = [int(i) for i in region.token_indices]
        regions.append({
            "id": region.region_id,
            "label": region.label,
            "label_source": region.label_source,
            "score": None if region.score is None else float(region.score),
            "bbox": [int(v) for v in region.bbox_xyxy],
            "area": int(region.area),
            "mask_ref": region.mask_ref,
            "mask": _png_url(overlay),
            "color": color,
            "memberships": list(zip(indices, map(float, region.token_weights))),
        })
    return {
        "width": width,
        "height": height,
        "preview": _png_url(memory.image),
        "image_hash": memory.image_hash,
        "fingerprint": memory.fingerprint,
        "contact_mode": memory.config["contact_mode"],
        "processed_size": list(memory.geometry.processed_size),
        "unassigned_count": len(memory.unassigned_token_indices),
        "regions": regions,
        "tokens": [{
            "index": int(token.visual_index),
            "source": int(token.source_index),
            "grid": [int(token.grid_y), int(token.grid_x)],
            "footprint": [float(v) for v in token.original_xyxy],
            "mrope": [int(v) for v in token.mrope_position],
        } for token in memory.tokens],
        "edges": [{
            "a": edge.region_a,
            "b": edge.region_b,
            "kind": edge.contact_kind,
        } for edge in memory.edges],
    }


def write_visualization(memory, output_path):
    """Write self-contained HTML using the saved image and numeric footprints."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_viewer_data(memory), separators=(",", ":"), ensure_ascii=False)
    temporary = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as out:
            temporary = Path(out.name)
            out.write(_HTML.replace("__SCENE_MEMORY_DATA__", encoded.replace("<", "\\u003c")))
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>区域视觉 KV Memory · VLCache</title>
<style>
:root { color-scheme: light; --ink: #172f38; --muted: #536d74; --page: #eef3f2; --paper: #fff; --line: #d1e0de; --accent: #17675d; --overlap: #984e13; --touch: #315d9e; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.65 system-ui, "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; }
.wrap { width: min(1260px, calc(100% - 48px)); margin: 0 auto; }
header { color: #eaf8f5; background: #10473f; padding: 33px 0 29px; }
h1,h2,h3,p { margin: 0; }
h1 { font-size: clamp(27px, 4vw, 42px); line-height: 1.22; letter-spacing: -.03em; }
header p { margin-top: 10px; color: #c6e7e0; }
main { padding: 24px 0 42px; }
.columns { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(315px, .85fr); gap: 18px; align-items: start; }
.card { background: var(--paper); border: 1px solid var(--line); border-radius: 15px; padding: 19px; min-width: 0; }
h2 { font-size: 21px; }
h3 { font-size: 16px; }
.muted { color: var(--muted); }
label,select,button { font: inherit; }
select,button { border: 1px solid var(--line); border-radius: 7px; padding: 5px 9px; background: #fff; color: var(--ink); }
button { cursor: pointer; }
:focus-visible { outline: 3px solid #d28a27; outline-offset: 3px; }
.tools { display: flex; gap: 10px 18px; align-items: center; flex-wrap: wrap; margin: 13px 0; }
.tools label { display: flex; gap: 6px; align-items: center; }
canvas { display: block; width: 100%; }
#imageMap { height: auto; border-radius: 10px; background: #dbe5e3; }
#tokenReadout { min-height: 46px; padding-top: 7px; overflow-wrap: anywhere; }
.aside > * + * { margin-top: 14px; }
dl { display: grid; grid-template-columns: minmax(0, 1fr) auto; margin: 0; gap: 4px 12px; }
dd { margin: 0; min-width: 0; overflow-wrap: anywhere; text-align: right; font-variant-numeric: tabular-nums; }
.edgeLegend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 13px; }
.edgeLegend i { display: inline-block; width: 23px; border-bottom: solid 3px var(--overlap); vertical-align: middle; margin-right: 5px; }
.edgeLegend .touch { border-color: var(--touch); border-bottom-style: dashed; }
#graph { position: relative; min-height: 270px; border: 1px solid var(--line); border-radius: 10px; background: #f7faf9; }
#graphCanvas { height: 100%; position: absolute; inset: 0; pointer-events: none; }
.node { position: absolute; transform: translate(-50%, -50%); color: var(--ink); border: 2px solid var(--node-color); line-height: 1.3; background: var(--paper); white-space: nowrap; font-size: 12px; }
.node[aria-pressed="true"] { border-color: var(--accent); background: var(--accent); color: #fff; }
.wide { margin-top: 18px; }
.scroller { overflow: auto; }
table { width: 100%; margin-top: 8px; border-collapse: collapse; font-size: 13px; }
th,td { text-align: left; padding: 5px 7px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); }
code { font-size: .86em; overflow-wrap: anywhere; word-break: break-all; }
.provenance > * + * { margin-top: 7px; }
[hidden] { display: none !important; }
@media (max-width: 810px) { .columns { grid-template-columns: 1fr; } }
@media (max-width: 470px) { .wrap { width: calc(100% - 24px); } .card { padding: 14px; } }
</style>
</head>
<body data-server-no-reload>
<header><div class="wrap">
  <h1>区域视觉 KV Memory</h1>
  <p id="summary"></p>
</div></header>
<main class="wrap">
  <div class="columns">
    <section class="card" aria-labelledby="imageTitle">
      <h2 id="imageTitle">原图与视觉 token 对齐</h2>
      <div class="tools">
        <label>查看区域 <select id="region"></select></label>
        <label><input type="checkbox" id="showMasks" checked> 区域掩码</label>
        <label><input type="checkbox" id="showBoxes" checked> 边框与 ID</label>
        <label><input type="checkbox" id="showGrid" checked> Token 网格</label>
      </div>
      <canvas id="imageMap" aria-label="原图上叠加区域掩码、边框和视觉 token 范围"></canvas>
      <p class="muted" id="tokenReadout">悬停查看视觉 token 的源位置、MRoPE 与区域覆盖率。</p>
    </section>
    <section class="card aside" aria-labelledby="graphTitle">
      <h2 id="graphTitle">区域接触候选图</h2>
      <div class="edgeLegend"><span><i></i>重叠 overlap</span><span><i class="touch"></i>边界接触 touch</span></div>
      <div id="graph" aria-label="区域节点与无向几何候选边"><canvas id="graphCanvas" aria-hidden="true"></canvas></div>
      <p id="noEdges" class="muted" hidden>当前掩码之间没有满足几何接触条件的边。</p>
      <div id="regionInfo" hidden>
        <h3 id="regionName"></h3>
        <dl id="regionStats"></dl>
      </div>
    </section>
  </div>
  <section class="card wide" aria-labelledby="edgeTitle">
    <h2 id="edgeTitle">几何候选边</h2>
    <p class="muted" id="contactDefinition"></p>
    <div class="scroller"><table id="edgeTable"><thead><tr><th scope="col">节点 A</th><th scope="col">节点 B</th><th scope="col">接触类型</th><th scope="col">距离</th></tr></thead><tbody id="edgeRows"></tbody></table></div>
  </section>
  <section class="card wide provenance" aria-labelledby="sourceTitle">
    <h2 id="sourceTitle">来源</h2>
    <p>归一化 RGB 图像 SHA-256 <code id="imageHash"></code></p>
    <p>Memory 指纹 <code id="memoryHash"></code></p>
    <p class="muted">网格颜色表示掩码与 footprint 的几何覆盖。选中区域后，黄色高亮显示该区域引用的源 KV 行。图中的连线为几何候选。</p>
  </section>
</main>
<script id="memoryData" type="application/json">__SCENE_MEMORY_DATA__</script>
<script>
"use strict";
const $ = id => document.getElementById(id);
const data = JSON.parse($("memoryData").textContent);
const regionSelect = $("region");
const graph = $("graph");
const mapCanvas = $("imageMap");
const mapCtx = mapCanvas.getContext("2d");
const graphCanvas = $("graphCanvas");
const graphCtx = graphCanvas.getContext("2d");
const regions = new Map(data.regions.map(region => [region.id, region]));
const memberships = new Map(data.tokens.map(token => [token.index, []]));
for (const region of data.regions) {
  for (const [index, weight] of region.memberships) memberships.get(index).push([region.id, weight]);
}
const format = n => n.toLocaleString("zh-CN");
const percent = n => (n * 100).toFixed(2) + "%";
const add = (parent, tag, text) => {
  const node = document.createElement(tag);
  node.textContent = text;
  parent.append(node);
  return node;
};
const bounds = [data.width, data.height];
const longestEdge = Math.max(...bounds);
const scale = Math.max(900, Math.min(1500, longestEdge)) / longestEdge;
mapCanvas.width = Math.max(1, Math.round(data.width * scale));
mapCanvas.height = Math.max(1, Math.round(data.height * scale));
mapCtx.setTransform(mapCanvas.width / data.width, 0, 0, mapCanvas.height / data.height, 0, 0);
$("summary").textContent = `${data.width} × ${data.height} 原图 · ${data.processed_size.join(" × ")} 预处理 · ${format(data.tokens.length)} 视觉 token · ${data.regions.length} 区域 · ${data.edges.length} 候选边`;
$("imageHash").textContent = data.image_hash;
$("memoryHash").textContent = data.fingerprint;
$("contactDefinition").textContent = data.contact_mode === "overlap_only"
  ? "本次仅记录存在重叠像素的掩码对。"
  : "闭合像素单元的最小距离为 0 时，图中记录一条重叠或接触边。";
$("noEdges").hidden = data.edges.length > 0;
$("edgeTable").hidden = data.edges.length === 0;
const allOption = add(regionSelect, "option", "全部区域");
allOption.value = "";
for (const region of data.regions) {
  const option = add(regionSelect, "option", region.label ? `${region.id} · ${region.label}` : region.id);
  option.value = region.id;
}
for (const edge of data.edges) {
  const row = add($("edgeRows"), "tr", "");
  for (const value of [edge.a, edge.b, edge.kind, "0"]) add(row, "td", value);
}

const spots = new Map();
const columns = Math.max(1, Math.ceil(Math.sqrt(data.regions.length * 1.35)));
const rows = Math.max(1, Math.ceil(data.regions.length / columns));
graph.style.height = `${Math.max(270, rows * 46 + 25)}px`;
data.regions.forEach((region, index) => {
  const x = (index % columns + 0.5) / columns;
  const y = (Math.floor(index / columns) + 0.5) / rows;
  const button = add(graph, "button", region.id);
  button.className = "node";
  button.type = "button";
  button.title = region.label || region.id;
  button.style.left = `${x * 100}%`;
  button.style.top = `${y * 100}%`;
  button.style.setProperty("--node-color", region.color);
  button.addEventListener("click", () => {
    regionSelect.value = regionSelect.value === region.id ? "" : region.id;
    redraw();
  });
  spots.set(region.id, {x, y, button});
});

function drawGraph() {
  const {width, height} = graph.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  graphCanvas.width = Math.round(width * dpr);
  graphCanvas.height = Math.round(height * dpr);
  graphCtx.scale(dpr, dpr);
  for (const edge of data.edges) {
    const a = spots.get(edge.a);
    const b = spots.get(edge.b);
    const active = !regionSelect.value || [edge.a, edge.b].includes(regionSelect.value);
    graphCtx.globalAlpha = active ? 1 : .23;
    graphCtx.strokeStyle = edge.kind === "overlap" ? "#984e13" : "#315d9e";
    graphCtx.setLineDash(edge.kind === "touch" ? [6, 4] : []);
    graphCtx.lineWidth = active ? 2.2 : 1.4;
    graphCtx.beginPath();
    graphCtx.moveTo(a.x * width, a.y * height);
    graphCtx.lineTo(b.x * width, b.y * height);
    graphCtx.stroke();
  }
  for (const [id, spot] of spots) spot.button.setAttribute("aria-pressed", String(id === regionSelect.value));
}

const images = new Map();
function decode(url) {
  const image = new Image();
  image.src = url;
  return image.decode().then(() => image);
}
let photo;
function drawMap() {
  if (!photo) return;
  const pixel = data.width / Math.max(1, mapCanvas.clientWidth);
  mapCtx.drawImage(photo, 0, 0);
  const selected = regions.get(regionSelect.value);
  const visible = selected ? [selected] : data.regions;
  if ($("showMasks").checked) {
    for (const region of visible) mapCtx.drawImage(images.get(region.id), 0, 0);
  }
  const weightByToken = new Map(selected?.memberships || []);
  for (const token of data.tokens) {
    const [left, top, right, bottom] = token.footprint;
    const weight = weightByToken.get(token.index);
    if (weight !== undefined) {
      mapCtx.fillStyle = `rgba(255,214,57,${0.18 + 0.28 * weight})`;
      mapCtx.fillRect(left, top, right - left, bottom - top);
    }
    if ($("showGrid").checked) {
      mapCtx.strokeStyle = "rgba(255,255,255,.55)";
      mapCtx.lineWidth = 0.65 * pixel;
      mapCtx.strokeRect(left, top, right - left, bottom - top);
    }
  }
  if ($("showBoxes").checked) for (const region of visible) {
    const [left, top, right, bottom] = region.bbox;
    const labelSize = 13 * pixel;
    mapCtx.lineWidth = 1.6 * pixel;
    mapCtx.strokeStyle = region.color;
    mapCtx.strokeRect(left, top, right - left, bottom - top);
    mapCtx.font = `bold ${labelSize}px system-ui`;
    mapCtx.textBaseline = "top";
    const label = region.id;
    const labelWidth = mapCtx.measureText(label).width + labelSize * 0.45;
    const labelLeft = Math.min(left, Math.max(0, data.width - labelWidth));
    const labelTop = Math.max(0, top - 1.25 * labelSize);
    mapCtx.fillStyle = region.color;
    mapCtx.fillRect(labelLeft, labelTop, labelWidth, labelSize * 1.2);
    mapCtx.fillStyle = "#172f38";
    mapCtx.fillText(label, labelLeft + labelSize * 0.23, labelTop);
  }
}

function drawRegionInfo() {
  const region = regions.get(regionSelect.value);
  $("regionInfo").hidden = !region;
  const stats = $("regionStats");
  stats.replaceChildren();
  if (!region) return;
  $("regionName").textContent = region.label ? `${region.id} · ${region.label}` : region.id;
  const neighbors = data.edges.filter(edge => edge.a === region.id || edge.b === region.id)
    .map(edge => edge.a === region.id ? edge.b : edge.a);
  const items = [
    ["掩码面积", `${format(region.area)} px²`],
    ["引用的视觉 token", format(region.memberships.length)],
    ["候选邻居", neighbors.join("、") || "无"],
    ["标签来源", region.label_source || "—"],
    ["区域分数", region.score?.toFixed(3) || "—"],
    ["掩码文件", region.mask_ref],
  ];
  for (const [label, value] of items) {
    add(stats, "dt", label);
    add(stats, "dd", value);
  }
}

function redraw() {
  drawMap();
  drawGraph();
  drawRegionInfo();
}
regionSelect.addEventListener("change", redraw);
for (const id of ["showMasks", "showBoxes", "showGrid"]) $(id).addEventListener("change", drawMap);
new ResizeObserver(drawGraph).observe(graph);
new ResizeObserver(drawMap).observe(mapCanvas);
mapCanvas.addEventListener("pointermove", event => {
  const rect = mapCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * data.width / rect.width;
  const y = (event.clientY - rect.top) * data.height / rect.height;
  const token = data.tokens.find(({footprint: [l, t, r, b]}) => l <= x && x < r && t <= y && y < b);
  if (!token) return;
  const found = memberships.get(token.index).map(([id, weight]) => `${id} ${percent(weight)}`);
  $("tokenReadout").textContent = `视觉 #${token.index} · 源序列 #${token.source} · 网格 [${token.grid.join(", ")}] · MRoPE [${token.mrope.join(", ")}] · ${found.join("、") || "未分配"}`;
});
mapCanvas.addEventListener("pointerleave", () => {
  $("tokenReadout").textContent = `未分配视觉 token ${format(data.unassigned_count)} 个。悬停查看原始序列索引与区域覆盖率。`;
});
async function loadImages() {
  const original = await decode(data.preview);
  const masks = await Promise.all(data.regions.map(region => decode(region.mask)));
  data.regions.forEach((region, index) => images.set(region.id, masks[index]));
  photo = original;
  redraw();
}
loadImages().catch(error => {
  $("tokenReadout").textContent = `图像无法显示。${error.message}`;
});
</script>
</body>
</html>
"""
