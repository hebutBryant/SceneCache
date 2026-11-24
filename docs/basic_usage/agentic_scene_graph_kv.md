# 规划引导的 Scene Graph KV 索引

这个入口处理单图、单个问题的一次感知 episode。它接受已有规划，或让本地 Qwen3-VL 生成一次问题相关规划。SAM3 定位候选，VLM 基于图像和带编号的证据裁剪板验证属性与关系。之后沿用 SceneCache 的固定图像前缀捕获、物理 token 映射和区域推理。可执行入口在 [run_agentic_episode.py](../../examples/scene_graph_memory/run_agentic_episode.py)。

## 本地入口与依赖

分支为 `feature/support_kvcache_reuse_053rc2`，基础提交为 `9c2935359cd53d64c53a2a8a48a0728c232b1c4e`。没有匹配的 AGENTS.md。原有固定 [单图 Memory 构建](scene_graph_memory.md)和[区域推理](scene_graph_inference.md)仍可运行。

- 本仓库的 `builder.build_image_memory` 接受已准备的 SAM3 masks，并在新 worker 用固定图像前缀捕获整图 raw KV。原来的 `build` 入口仍接受固定 concepts 或 `--masks-file`。
- `mapping.assemble_memory` 使用实际 processor 尺寸、patch、merge、MRoPE 和源序列位置映射。几何边在 `graph.build_contact_edges` 内按原图 mask 的 overlap/touch 计算。
- `agentic_graph` 调用现有 `sam3_adapter`。它从外部 checkout 的 `visual_tools.sam.SAMTool._detect` 取得原图坐标 masks。SAM3 使用现有 GraphAgent Python、Transformers 和本地 checkpoint。
- 本地 `/home/lipz/Scene2Graph/run_agentic_sgg.py` 使用 `AgenticSGGOrchestrator`，包含 planning controller、工具 registry、在线语义字段生成和 replay 证据。适配层复用 SAM3 以及带原图像素校验的已有 planning/decision/tool request 归档导入。完整的外部 token monitor 与规划运行时继续独立，SceneCache 本次执行自己的有界验证和复用流程。
- 推理模型是 `/home/hdd4/lpz/Qwen3-VL-8B-Instruct`。构建和推理使用现有 `scenecache` Conda 环境，Python 3.11，PyTorch 2.8.0+cu128，SGLang 0.5.3rc2 和 FlashInfer 0.4.0rc3。每个 QA worker 限单并发、TP/PP=1、BF16。

## 索引与复用边界

规划由 question、concepts、attribute intents、relation intents 和有序的 perception operations 组成。已有规划直接进入分割和验证。SAM3 与 VLM 的调用上限、concepts、区域和证据图板大小都有预算。候选通过 mask 源像素 hash 或至少 0.98 的 mask IoU 去重，多个实例可保持独立。VLM 的语义验证记录具体端点、原图 mask 证据、状态和来源。置信度是可选的模型自评，检索默认跳过显式置信度低于 0.5 的关系。几何接触保持独立类型。

`semantic_graph.json` 保存语义、规划、证据和验证轨迹。稳定的 agent 区域 ID 和按 mask hash 绑定的物理区域 ID 同时保留。`semantic_kv_index.json` 记录源 bank 身份、行索引和覆盖权重、各层 safetensors 名称、原序列索引、MRoPE 与证据映射。`memory/manifest.json` 给出图像、图像预处理、模型和 processor 指纹、固定前缀、各层 dtype/shape/布局、K/V 阶段以及 payload SHA256。当前路径支持本地 dense BF16 checkpoint，运行模式使用原始权重。

关系通过已验证的端点和证据区域反查可见 token。共享源 bank 保存整图一次。区域重叠按源行做集合去重。语义边不分配额外张量。推理请求只分配自己接下来要读取的 GPU KV 槽，在该请求完成后由原 allocator 回收。持久化文件引用指向 safetensors 的行，GPU slot 仅在请求运行时存在。episode 结束后隔离 worker 退出。下一张图使用新 memory，原图或 source fingerprint 不匹配时拒绝绑定。

固定图像前缀由图像开头的原生 chat template 生成。K 保存于 QK RMSNorm 和 MRoPE 前，V 为 projection。回放时先校验像素、processor grid、图像前 token ID 和三轴 MRoPE，再按源位置恢复 K 的 norm/rotation 一次。图像前后边界 token 仍按原聊天模板处理。后续文本的逻辑位置和 decode 的 MRoPE delta 保留原图语义。区域重放复用可见集合中未重算的行，也按当前 VLCache 策略重算一部分可见视觉行。省去非连续行属于稀疏 attention 近似，必须单独核验回答质量。crop 输入有独立处理尺寸和逻辑前缀，只能命中该 crop 自己的 native cache。

检索使用实体/别名、属性匹配、语义关系的两端和引用的证据。扩展只在预算允许时加入选中邻居，并给少量同图背景 token 留预算。API 同时输出检索理由、缺证据条目和 fallback。必要实体、属性、关系无法满足，或者有证据但 token 超预算时，返回显式全图回退。区域回答报告证据不足时，回放器允许一次有界扩展及一次 native full 复核。

## 可复现命令

使用本地兼容 OpenAI Chat 的模型服务作为 planner/verifier。下面的实测使用了已有的 `127.0.0.1:30000` Qwen3-VL-8B-Instruct 服务。每次完整运行使用新输出目录和未占用的 QA 端口。

```bash
conda activate scenecache
export OMP_NUM_THREADS=4 MAX_JOBS=4
SCENE_MODEL=/home/hdd4/lpz/Qwen3-VL-8B-Instruct
python examples/scene_graph_memory/run_agentic_episode.py \
  --image /home/lipz/test_football.jpg \
  --question 'Is the goalkeeper jumping in front of the goal net? Answer only yes or no.' \
  --accepted yes \
  --model-path "$SCENE_MODEL" --vlm-url http://127.0.0.1:30000 \
  --gpu 1 --port 30020 --out .runtime/agentic_episodes/football_relation_reproduce \
  --sam3-source /home/lipz/Scene2Graph \
  --sam3-python /home/lipz/miniconda3/envs/GraphAgent/bin/python \
  --sam3-checkpoint /home/hdd4/lpz/SAM3/checkpoints \
  --max-concepts 6 --token-budget 600 --replay-rounds 2 --closed-loop
```

传入同一个问题和该图的 `--existing-plan path/to/plan.json` 可跳过 planner。Scene2Graph 的既有规划归档目录也可传入，导入器只接受原图像素一致、由实际 SAM3 tool request 支持的概念。归档中的自然语言计划保留为 provenance。它不被直接当成 SAM3 的 prompts。原图验证失败会显式报错。

构建后可离线查询并查看选中证据、token 和每层真实引用。

```bash
CUDA_VISIBLE_DEVICES='' python -m sglang.srt.mem_cache.scene_graph_memory agentic-retrieve \
  --memory-dir .runtime/agentic_episodes/football_relation/memory \
  --semantic-graph .runtime/agentic_episodes/football_relation/semantic_graph.json \
  --question 'Is the goalkeeper positioned in front of the goal net?' --operation inspect_1 \
  --token-budget 600 --background-tokens 16
```

同图、同问题可用 `--resume-build .runtime/agentic_episodes/football_relation` 重新加载并回放。保存的 mask、KV 与模型身份仍会重新校验。该模式报告新回放的实际 wall time，冷场景账单引用构建运行中已记录的规划、工具和捕获耗时。

```bash
python examples/scene_graph_memory/run_agentic_episode.py \
  --image /home/lipz/test_football.jpg \
  --question 'Is the goalkeeper jumping in front of the goal net? Answer only yes or no.' \
  --accepted yes --model-path "$SCENE_MODEL" --gpu 1 --port 30020 \
  --resume-build .runtime/agentic_episodes/football_relation \
  --out .runtime/agentic_episodes/football_relation_replay \
  --token-budget 600 --replay-rounds 2 --closed-loop
```

最终回答来自真实 SGLang 调用。推理记录包括 KV backend、活跃/重算行、K/V 和索引字节、decode KV 长度、cache hit、fallback、TTFT/TPOT 和 CUDA allocator 峰值。查看 `episode_trace.jsonl`、`inference_steps.json`、`evidence/*.png` 与可离线打开的 `mask_token_alignment.html`。

## 对照与计时

同一 VLM 实际生成的 perception 轨迹在 A–D 中回放。A 是整图 native KV 流水线。B 是普通按需 crop 缓存，每个 crop 有自己的图像前缀。C 从相同 SAM3 区域用几何关系选取 token。D 从语义图和证据索引选取 token。C/D 使用同一个固定前缀物理 bank，按各自相同的份额计入一次构建和加载成本。已有规划在所有臂共同复用，因此 A 也是同样规划下的完整图像对照。每臂开始清空该隔离 worker 的 native KV/encoder cache，后续操作保留臂内缓存。另可开启 D 闭环，用自己的已执行感知答案作为下一次调用的文本观测。

每个模式的请求 wall time 与阶段成本串行记账。规划、SAM3、VLM 验证、资产 hash、KV capture worker、持久化、索引和 QA worker 加载分别记录。QA 权重启动与健康检查属于运行环境准备，在 `qa_worker_startup_s` 单列。bank load/资产验证计入 C/D 的冷记账。`serial_cold_equivalent_s` 合并共同实测阶段和该臂的冷缓存回放，所有臂在同一 QA 进程串行执行。它是归因成本，完整程序的真实 wall time 单独保存。启用 `--verify-capture` 时捕获调试成本仍包含在实际构建总时间中。GPU 上有共享任务，跨图片不做硬件等价的延迟比较。样本是少量真实问题，用于检查可运行性和成本规模。

TTFT 从本地请求到首个非空文本。TPOT 主要看服务端每个同步 decode 步，单 token 没有 TPOT。真实流文本时间也保留，若多个 token 合成同一个片段，客户端口径可得到 0。账单不依据选中 token 数估算 H2D，读取实际推理计数。

## 真实检查

### 足球图

Qwen 规划了查守门员位置、腾空状态和球门柱的三次感知操作。SAM3 生成四个区域。VLM 验证三条关系，包括 `goalkeeper → in front of → goal net` 和 `goalkeeper → jumping → goal net`。原图可核查这些端点。源 bank 是 36 层、798 个视觉行、112.22 MiB。语义闭环实际把三条前序回答作为文本观测传给后续操作，最终回答 `yes`。最终调用选择 549 行，重算 164 行，搬运其余 385 行，共 54.14 MiB。Decode 继续读取紧凑 KV，额外 decoder KV H2D 为 0。

源构建记录在 `.runtime/agentic_episodes/football_relation`，整次程序 wall time 为 360.3 秒，包含首次空缓存建图、编译、分臂回放和 QA worker 启动。规划 21.76 秒、SAM3 17.90 秒、VLM 验证 32.28 秒。银行构建耗时 129.06 秒，主要是资产校验 21.87 秒及捕获 worker 105.69 秒。物理映射 0.10 秒、持久化 1.40 秒。修复谓词匹配后的真实同轨迹回放在 `.runtime/agentic_episodes/football_final_replay`，worker 启动 53.08 秒，其中 bank 加载和资产验证 19.97 秒。

| 组 | 答案 | 8 次实际调用 H2D | 8 次重算视觉行 | 命中 TTFT 中位 | 命中 TPOT 中位 | 单 episode 冷成本串行记账中位 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A 全图 | yes | 471.66 MiB | 3030 | 329.1 ms | 32.18 ms | 25.42 s |
| B 独立 crop 缓存 | yes | 314.44 MiB | 2792 | 318.2 ms | 31.44 ms | 42.28 s |
| C 几何索引 | yes | 440.16 MiB | 1332 | 353.5 ms | 29.83 ms | 190.42 s |
| D 语义索引 | yes | 330.75 MiB | 1004 | 344.3 ms | 29.97 ms | 223.07 s |

语义索引相对几何索引少搬运 24.9% 的 KV。八次请求中实际复用的视觉行占活跃行 70.1%，同一源 bank 有 45.36% 的行从未读取。D 的匹配回放时长是 1.80 和 1.79 秒，冷成本主要来自在线构建。TPOT 与 C 接近。此样本中 A/B 的命中 TTFT 仍更低。各组共享 worker 的 PyTorch GPU 峰值约 19.33 GB。累计调用次数和已复用比例在 summary 的 `cumulative_observed_calls` 中可查。

### 市场图与边界

市场图的已有规划重跑使用了同一份持久化计划，跳过 planner。SAM3 27.32 秒、在线 VLM 验证 27.63 秒。最终回答为 `blue`。原始运行及各类回退写在 `.runtime/agentic_episodes/market`、`market_existing_plan` 和 `market_verified`。其中实际观察到语义证据不足、token 预算和模型回答不足触发全图回退。早期 VLM 曾把两个近重复的人物 mask 当作不同实例，离线审计的 IoU 为 0.9937，最新构建器的去重检查会合并这对候选。另一轮足球图曾把红球衣绑定给绿衣守门员，证据审计发现框不相交，新增的保守检查把这类着装边降为不确定。旧图保留原判断以供复核。VLM 声明 `supported` 仍需结合原图和证据板判断。

固定图像前缀下，普通冷 prefill 与打开捕获的 logits 逐位一致。已有 VLCache 的同前缀部分重算复用命中全部 36 层，首 token 一致，logits 最大绝对差 0.265625、平均绝对差 0.04359。物理索引从市场图和足球图持久化 bank 取出的 36 层值与独立 safetensors gather 逐位一致。两张图都验证了 crop 会被原图 bank 拒绝。复核记录在 `.runtime/agentic_episodes/semantic_bank_verification.json`。完整 Scene Graph memory 测试在 CUDA 环境跑过 46 项。
