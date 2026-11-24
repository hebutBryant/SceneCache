# 单图区域图与视觉 KV Memory

给定一张图，初始化命令先生成或导入原图尺寸的区域 mask，再用仓库里的 Qwen3-VL 执行一次完整 prefill。保存的是 **语言模型全部层的视觉 K/V**。每个区域保存索引和覆盖权重，共享一份整图 K/V。加载、查询和导出在 CPU 上进行。

问题驱动的 SAM3/VLM 建图、语义检索和在线推理入口见[规划引导的 Scene Graph KV 索引](agentic_scene_graph_kv.md)。

首版支持本地非量化的 dense `Qwen3VLForConditionalGeneration`，单图、batch=1、TP=1、PP=1、单 NVIDIA GPU。真实验证选择了已在这个 fork 上运行的 Qwen3-VL-8B-Instruct。没有扩展到 Qwen2.5-VL 或 MoE。区域 K/V 仍包含整图上下文，只提供读取和物化接口。后续可由问题引导的区域选择器调用 `get_regions_kv`，本实现未改写生成、注意力或位置编号，也未评估问答准确率。

## 本地代码与资源

本地 checkout 是 `/home/lipz/SceneCache`，分支 `feature/support_kvcache_reuse_053rc2`，基准提交 `9c2935359cd53d64c53a2a8a48a0728c232b1c4e`，本地 origin 为 SceneCache fork。开始时没有适用的 AGENTS.md，已有环境、启动脚本和报告改动保留。本实现新增一个内聚包，沿用本地 SGLang 0.5.3rc2。研究背景见 [VLCache 论文](https://arxiv.org/abs/2512.12977)。

| 组件 | 验证环境 |
| --- | --- |
| Qwen 模型 | `/home/hdd4/lpz/Qwen3-VL-8B-Instruct`，本地四分片 safetensors |
| LLM 进程 | conda `scenecache`，Python 3.11.18，PyTorch 2.8.0+cu128，CUDA 12.8，transformers 4.57.0，SGLang 0.5.3rc2 |
| 设备 | Linux，RTX A6000，物理 GPU 1，NVIDIA 驱动 580.173.02。在命令里将其暴露为进程逻辑 `cuda:0` |
| SAM3 权重 | `/home/hdd4/lpz/SAM3/checkpoints` 的 Hugging Face 格式 `model.safetensors` |
| SAM3 工具 | `/home/lipz/Scene2Graph/visual_tools/sam.py` 中的 `SAMTool._detect`。在 conda `GraphAgent` 的 Python 3.10.20、PyTorch 2.12.0+cu130、transformers 5.8.1 环境运行 |

SAM3 wrapper 接受概念提示词，不提供自动穷举场景的接口。每个提示词的候选根据分数、mask 阈值和配置筛选。mask 为 EXIF 方向归正后原图的 H×W 布尔数组。重叠或嵌套的实例可同时保留，过滤项与原因写入 manifest。SAM3 子进程结束后才启动 Qwen，两个模型可以顺序共用一张 GPU。

## 调用链和数值表示

1. [qwen_vl.py](../../python/sglang/srt/multimodal/processors/qwen_vl.py) 解码图像，先执行 fork 的 `resize_image`，再调用模型的 HF processor。记录实际输入尺寸、中间尺寸和 `image_grid_thw`。按实际模型的 patch/merge 参数和网格恢复处理后的尺寸。MVP 校验 resize-only 几何，遇到不支持的 crop/pad 或多图时中止。
2. [qwen3_vl.py](../../python/sglang/srt/models/qwen3_vl.py) 按 `(t, merged_y, merged_x)` 顺序输出视觉向量。DeepStack 沿特征维拼接，与主视觉向量共享同一组 token 行。builder 校验实际嵌入行数、视觉占位符范围、逐层 K/V 行数和 `t*h*w/merge²` 相等，并用实际 MRoPE 位置校验合并网格的行优先顺序。参考路径 [qwen2_5_vl.py](../../python/sglang/srt/models/qwen2_5_vl.py) 在视觉窗口计算后使用 `argsort(window_index)` 恢复顺序。该路径暂未接入。
3. [flashinfer_backend.py](../../python/sglang/srt/layers/attention/flashinfer_backend.py) 的 `update_variable_block_wrapper` 根据图像哈希查询 [MockKVManager](../../python/sglang/srt/managers/mock_kv_manager.py)，并设置 `compute_mask`、`recompute_info` 和 `write_info`。专用 worker 只接收本次初始化请求，明确旁路独立的图像 KV 和视觉嵌入缓存。Radix、CUDA Graph 和 chunked prefill 同时关闭。只允许完整的单图 extend，捕获跨度、源序列长度不完整或超过预算时失败。
4. [qwen3.py](../../python/sglang/srt/models/qwen3.py) 原来的 `maybe_write_kv` 在 QK norm 和 MRoPE 前，新增读取点与该阶段一致。K 是 QKV projection 分出的 raw K，V 是 raw V，逐层形状 `[visual_tokens, kv_heads * head_dim]`。K 在写入普通 attention pool 前还需要 per-head RMSNorm 与原位置 MRoPE，不能将这里存储的 raw K 直接当成 pool 的 physical K。对照检查了 [qwen2.py](../../python/sglang/srt/models/qwen2.py) 的写入点，也在旋转之前，但本实现未接入该模型。
5. 固定来源输入由模型原生 chat template 渲染，用户消息只含图片，添加 assistant generation prompt。保存模板、实际源 token IDs、scheduler hash 占位符 ID、完整线性/MRoPE 位置、视觉序列索引和非视觉索引。区域 bank 不包含文本、图像边界 token 或生成 token。逐层张量通过阻塞式 CPU copy 从 GPU 分离，与请求 slot 生命周期无关。

在专用进程里没有配置捕获目录时，常规请求不进入 capture 逻辑。开启 `--verify-capture` 时额外执行同输入的正常冷 prefill 和正常图像 KV 复用，逐值比较捕获与普通冷算的首 token logits。常规 build 只执行一次 prefill。

## 区域索引、接触图与保存

每个 merger 后 token 覆盖处理图像上的一个 `patch_size × spatial_merge_size` 边长方块。将方块的连续边界逆缩放到方向归正后的原图，计算其与 mask 前景像素单位方格的面积交集。索引包含视觉局部位置、源序列位置、原图/处理图 footprint 和 MRoPE。权重为交集面积除以 token footprint 面积。默认保留任何正面积交集，避免漏掉没有覆盖网格中心的细小区域。token 可属于多个区域。未被纳入任何区域的 token 单列为 unassigned。

默认接触模式 `pixel_cells` 把前景像素当成闭合单位方格。mask 交集非空为 overlap。对无交集的 mask，仅对单侧查一次 3×3 邻域，命中为 touch，包含角点接触。有一整像素的空隙时不会连边。`overlap_only` 只连像素交集。两种模式都使用原图 mask，bbox 仅作预筛选。所有边无向、去重、规范排序，标为 `contact_candidate`，距离定义随边持久化。

输出目录含 `image.png`、`masks/rXXXX.png`、`source_kv.safetensors` 和 `manifest.json`。manifest 包含版本、图像内容与模型文件 SHA256、processor、模板和源位置指纹、映射和捕获版本、区域索引、接触边、每层 dtype/shape 与文件摘要。保存先写入同盘临时目录，重新加载、校验文件并与捕获值逐位比较，再原子发布。已有目录只有完整指纹和逐层 K/V 一致才会接受。配置变更后使用新的输出目录。加载时可传 `expected_fingerprint` 验证期望来源。

## 运行命令

从当前仓库根目录运行。若再次执行示例，已有目录会按上面的规则验证。

```bash
conda activate scenecache
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4
export MAX_JOBS=4 FLASHINFER_WORKSPACE_BASE="$PWD/.runtime/flashinfer" TRITON_CACHE_DIR="$PWD/.runtime/triton"
export PATH="$CUDA_HOME/bin:$PATH"

python -m sglang.srt.mem_cache.scene_graph_memory build \
  --image /home/lipz/test_market.jpg \
  --model-path /home/hdd4/lpz/Qwen3-VL-8B-Instruct \
  --device cuda:0 --sam3-device cuda:0 \
  --sam3-source /home/lipz/Scene2Graph \
  --sam3-python /home/lipz/miniconda3/envs/GraphAgent/bin/python \
  --sam3-checkpoint /home/hdd4/lpz/SAM3/checkpoints \
  --sam-prompts-file examples/scene_graph_memory/market_prompts.json \
  --sam3-max-candidates 4 --max-image-edge 1344 \
  --max-image-tokens 1536 --context-length 2048 --mem-fraction-static 0.97 \
  --verify-capture --output-dir .runtime/scene_graph_memory_market

CUDA_VISIBLE_DEVICES='' python -m sglang.srt.mem_cache.scene_graph_memory inspect \
  --memory-dir .runtime/scene_graph_memory_market
CUDA_VISIBLE_DEVICES='' python -m sglang.srt.mem_cache.scene_graph_memory export-region \
  --memory-dir .runtime/scene_graph_memory_market --region-id r0001 \
  --output .runtime/scene_graph_market_r0001.safetensors
CUDA_VISIBLE_DEVICES='' python -m sglang.srt.mem_cache.scene_graph_memory visualize \
  --memory-dir .runtime/scene_graph_memory_market --output reports/scene_graph_market.html
CUDA_VISIBLE_DEVICES='' python examples/scene_graph_memory/verify_bank.py \
  .runtime/scene_graph_memory_market --export .runtime/scene_graph_market_r0001.safetensors
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s test/srt/scene_graph_memory -v
```

`--max-image-edge` 在 SGLang 中间预处理前再做一次无裁剪缩放。SAM3 始终在 EXIF 归正的原图坐标系输出。可用 `--masks-file` 导入本工具的 manifest，或导入 NPZ 和 PNG 路径清单。NPZ 使用 `masks` 键和 `[N,H,W]` 的 bool 或二值整数数组。JSON 可用 `{"regions":[{"mask":"mask1.png","label":"face mask","score":0.95}]}`，路径相对于 JSON 文件。

已运行的导入示例。这里仍执行一次 Qwen prefill。

```bash
python -m sglang.srt.mem_cache.scene_graph_memory build \
  --image /home/lipz/test_market.jpg --model-path /home/hdd4/lpz/Qwen3-VL-8B-Instruct \
  --device cuda:0 --masks-file .runtime/scene_graph_memory_market/manifest.json \
  --max-image-edge 1344 --max-image-tokens 1536 --context-length 2048 --mem-fraction-static 0.97 \
  --output-dir .runtime/scene_graph_memory_market_imported
```

Python API 示例

```python
from sglang.srt.mem_cache.scene_graph_memory import load_memory

memory = load_memory(".runtime/scene_graph_memory_market")
one = memory.get_region_kv("r0001")
nearby = memory.get_regions_kv(["r0001"], expand_hops=1)
print(memory.get_neighbors("r0001"))
print(one.layers[0][0].shape, one.source_indices, one.mrope_positions)
```

`get_region_kv(..., layers=[0, 7])` 可选择层。多区域查询按源序列顺序取并集，不重新编号位置。单区域导出包含该区域每层的 K、V 及可追溯源 bank 的 safetensors 元数据。HTML 可离线打开，切换区域 mask、token 网格和 overlap/touch 关系。

## 真实运行记录

- 市场图原图 2400×1217。进程输入 1344×682，实际 processor 尺寸 1344×672，patch 网格 `[1, 42, 84]`，patch 16，merge 2。LLM 视觉 span、嵌入和 K/V 均为 882 行。DeepStack 嵌入为 `[882, 16384]`。
- SAM3 的 6 个 concepts 保留 16 个区域，69 个候选因每概念最多 4 个的设置被过滤。17 条边由 11 条 overlap 和 6 条 touch 组成。482 个视觉 token 被至少一个区域覆盖，其中 122 个有多重归属。未覆盖 400 个，与区域并集共同覆盖整图序列。
- 36 层，每层 K、V 均为 BF16 `[882, 1024]`。bank 张量总量为 130,056,192 字节（124.03 MiB），region membership JSON 索引为 13,562 字节。捕获阶段无旧图像 KV 读取或写入，实际重算 882 行。启用捕获与普通完整 prefill 的 `[1, 151936]` logits 最大绝对差 0（`atol=rtol=0`），贪心首 token ID 均为 1986。常规 VLCache 复用对照命中 36 层并重算 264 个图像位置。
- 原进程退出后，在 `CUDA_VISIBLE_DEVICES=''` 下成功执行 inspect、export 和 visualize。全部 16×36 组区域 K/V 与 bank 的对应行逐值一致，导出的 `r0001` 有 36 层、每层 K/V 各 `[4, 1024]`。另外独立复核了 14,112 项 mask–token 覆盖率与 120 个 mask 对。导入自身 manifest 后再次实际 prefill，两个 source bank 文件 SHA256 一致。11 项 CPU 单元测试通过，HTML 在桌面和手机 Chromium 离线验证通过。
- 此外使用 `env -u SGLANG_SCENE_GRAPH_CAPTURE_DIR … python .runtime/run_scene_graph_legacy.py` 在本次修改后的源码上启动普通 Engine。连续两个同图请求均成功返回 token 1986，见 `.runtime/scene_graph_legacy_smoke.json`。这项检查没有启用 capture 环境开关。
- 产物在 `.runtime/scene_graph_memory_market/manifest.json`、`.runtime/scene_graph_market_r0001.safetensors`、`reports/scene_graph_market.html`。命令输出和独立检查分别记录在 `.runtime/scene_graph_market_build.json`、`.runtime/scene_graph_market_verify.json` 和 `.runtime/scene_graph_market_cpu_verification.json`。
