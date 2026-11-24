# Mini-o3 终态证据与 K/V 复用实验

实验用 Mini-o3 的官方工具协议获取视觉观察，每张图像围绕采集题运行一次自主搜索。采集结束后，各道题独立从同一个只读终态续写。参考答案仅用于离线评分。实验测量同图多问时各缓存策略的答题表现、首 token 延迟和构建成本，结果限于锁定样本。

## 运行环境

本机使用 conda 环境 `scenecache`，其中 Transformers 4.57.0 和 PyTorch 2.8.0 支持 Mini-o3 的 `Qwen2_5_VLForConditionalGeneration`。当前范围为单请求、batch 1、单 NVIDIA GPU，注意力后端使用 Transformers 原生 SDPA。C 保留 HF 的请求内缓存。这个原生 Transformers 实验入口无需启动 SGLang 服务，跨请求复用由本实验的 D 路径负责。Mini-o3-7B-v1 有 28 个完整注意力层，每层 4 个 K/V 头，且关闭滑动窗口。缓存采用 Transformers `DynamicCache`，K 为应用 MRoPE 后的表示，V 为线性投影结果。

配套代码使用本机 Mini-o3 官方代码目录中的裁剪实现和工具消息模板。传入 Mini-o3 根目录时，运行脚本定位其中的 `Mini-o3-7B-v1` 和 `code` 子目录。选择清单记录实际使用的样本和源图定位，运行阶段按清单复查，便于区分样本发现问题与推理结果。

## 终态与四种运行模式

一次采集保存裁剪调用、观测图像、父子观察编号和原图像素坐标。坐标采用方向归一后的 RGB 图像及整数半开区间 `xyxy`。终态 `z` 记录源图 hash、末次裁剪的像素 hash、原图 bbox、尺寸和可用证据引用。透传给模型的是固定版本的感知字段与裁剪图像，校验用的 hash 保存在清单。OCR、属性与关系字段保留为未验证状态，并记录可能缺少上下文。采集轨迹另行保存。

| 模式 | 终态保持方式 | 问答阶段 |
| --- | --- | --- |
| A `full_history` | 采集得到的原始完整 Decoder K/V | 匹配真实历史前缀，追加相同问题后缀并继续推理 |
| B `terminal_kv_slice` | 从 A 的原始 K/V 复制终态相关的精确行 | 源 K/V 可释放。物理槽位按紧凑长度追加，保留原始 K 的 MRoPE 和新 token 的逻辑坐标 |
| C `tool_result_cache` | `z` 与终态图像证据 | 在相同规范化终态提示下，每次问答重新 prefill，生成过程使用原生请求内 K/V |
| D `canonical_state_kv` | 从和 C 相同的规范化状态与图像单独 prefill，独立保存到 CPU 及磁盘 | 对每条路径载入一次终态 bank，该图的各道题共享不可变的前缀行，各自追加新的 K/V |

B 保留的是原始采集序列中所选行的状态。这些高层 K/V 已带有当时见过的历史上下文，删行也缩小了续写时的注意力范围，所以 B 与 A 可能产生不同回答。B 的物理缩短和逻辑 MRoPE 是两套坐标。Transformers 用连续的物理 `cache_position` 追加并构造因果掩码，新 token 另用完整逻辑提示中的三轴 `position_ids`。D 的 K/V 与重建的规范化前缀绑定，来源和逻辑坐标与 B 分开核查。运行时从本机 Transformers 取得完整三轴 `position_ids` 和 `rope_deltas`，保存并核验提示中的真实 token ID、位置、mask、图像网格、像素及预处理 patch 指纹。续写显式传入逻辑位置，物理 `cache_position` 依据当前持有的行数递增。

C 和 D 用相同的状态文本、图像像素、processor 和渲染模板，故可直接检查 D 缓存恢复是否保留 C 的输入与生成行为。A 与 B 的对照反映历史删行。C 或 D 与 A 的差异还包含提示及历史重建的影响。

### 改问与不同采集路径

采集题和保留的后续题共享终态证据。每个请求拥有自己追加的 KV。证据不足时记录模型发出的工具请求或 `<need_evidence>`。问答阶段的请求各自从同一终态开始，只执行一次生成。VisualProbe 的原题使用 Mini-o3 选择项格式兼容的精确匹配，没有独立参考的后续题保留人工核验规则。

路径检查用不同的父裁剪构造第二条观测链，定位同一末次原图矩形。严格比较末次裁剪模式、尺寸、像素 hash 和 RGB 字节，再检查 `z` 一致性。第二条链的工具输入按确定的坐标传入，观察历史通过已知工具响应构建。该检查验证同一末次像素证据的规范化复现性，复现条件包括可构造合规的不同父裁剪。

## 选样和运行

### GQA 同图多问

GQA 输入为 `/home/hdd3/lpz/GQA` 下的两个本地 ZIP。标注从 `questions1.2.zip` 的 `val_balanced_questions.json` 读取，用 `imageId` 分组，按题目 ID 记录原始问题和标注答案。像素从 `images.zip` 读取对应的 `images/<imageId>.jpg` 成员，只解码清单中的图像。

这组实验以图像为选样单位，用种子 42 锁定 10 张不同图像，每图 3 道有标注的问题。第一道用于采集并参加评测，后两道只在采集结束后进入推理。四种模式共用同一份清单。逐题结果可比较采集题与后续题的精确匹配，同时查看同一图像终态能否覆盖其他问题需要的证据。

采集题附有定位提示。模型从原图自行决定裁剪区域，最多执行 4 轮采集，每轮最多生成 192 token。各模式使用相同的 `<answer>` 回答前缀，采用贪心解码，最多生成 48 token。首次样本仍执行同样的 C/D 和 A/B logits 校验。

有有效裁剪时，B 复制系统消息与末次工具响应的行，C 和 D 使用末次裁剪。若模型直接回答或采集未产生有效裁剪，终态采用原图证据并记录 `whole_frame_original_evidence`。这类样本的 B 保留整段前缀，KV 行数不减少。下列运行命令仅评测真实采集路径。省略 `--skip-controlled-replay` 可对有有效裁剪的样本附加受控路径。

从仓库根目录执行。更换运行目录可以保留先前的报告与清单。

```bash
conda activate scenecache
GQA_ROOT=/home/hdd3/lpz/GQA
MODEL_PATH=/home/hdd3/yaoyf/Mini-o3
GQA_RUN=.runtime/terminal_state_reuse/gqa

PYTHONPATH=python python examples/scene_graph_memory/run_terminal_state_reuse.py \
  --dataset gqa --data-root "$GQA_ROOT" --model-path "$MODEL_PATH" \
  --num-images 10 --questions-per-image 3 --seed 42 \
  --skip-controlled-replay \
  --output-dir "$GQA_RUN" --prepare-only

PYTHONPATH=python python examples/scene_graph_memory/run_terminal_state_reuse.py \
  --dataset gqa --data-root "$GQA_ROOT" --model-path "$MODEL_PATH" \
  --num-images 10 --questions-per-image 3 --seed 42 \
  --skip-controlled-replay \
  --output-dir "$GQA_RUN" --manifest "$GQA_RUN/selected_manifest.json"
```

### VisualProbe

默认在 `--data-root` 下查找 `VisualProbe_Easy/val.json`、`VisualProbe_Medium/val.json`、`VisualProbe_Hard/val.json`。难度来自官方验证集划分。每行用 `doc_id`、`images`、`problem` 和 `solution` 定位一个样本，并要求恰好一张可解码的源图。先按样本 ID 排序，用固定种子 42 在各难度抽取 5 条。当前类别样本不足时保留实际条数，记录短缺。锁定清单保留数据根目录、标注 SHA-256、难度来源、样本 ID、源行、原始题目、参考答案，以及图像文件和方向归一 RGB 像素的 SHA-256。运行阶段重新校验来源和图像。

若已有资源保留为其他本地标注格式，可在构建阶段指定 `--prepare-only --rows-file ...` 来使用 JSON、JSONL 或 Parquet 文件。映射字段为 `id`、`difficulty`、`image`、`question` 和 `answer`，可重复传 `--field CANONICAL=SOURCE`。有真实改问标注时也可映射 `followup_question` 和 `followup_answer`。其余样本用锁定清单中的可核验视觉线索改问和人工评分细则，保留无参考答案的标记。图片应位于数据根目录内，或嵌在 Parquet 行中。

先运行发现与锁定清单。正式运行先检查首个真实样本是否形成有效终态，再按实现的门槛处理其余样本。

从仓库根目录执行。将数据目录和设备参数设为实际分配的资源。

```bash
conda activate scenecache
DATA_ROOT=/home/hdd4/lpz
MODEL_PATH=/home/hdd3/yaoyf/Mini-o3
OUTPUT_DIR=.runtime/terminal_state_reuse/visualprobe

PYTHONPATH=python python examples/scene_graph_memory/run_terminal_state_reuse.py \
  --data-root "$DATA_ROOT" --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" --prepare-only

PYTHONPATH=python python examples/scene_graph_memory/run_terminal_state_reuse.py \
  --data-root "$DATA_ROOT" --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" --manifest "$OUTPUT_DIR/selected_manifest.json"
```

本机模型配置和官方工具目录可读取。当前在 `/home/hdd4/lpz` 缺失 Easy、Medium、Hard 三个默认验证标注文件，正式样本锁定和 Mini-o3 A/B/C/D 结果因此阻塞。CPU 检查可独立执行。

## 统计与独立检查

记录分模式的终态构建成本和问答成本，并保留采集工具轨迹。前缀的 K/V 逻辑字节数由张量元素计算，实际 storage 字节按设备和存储指针去重。紧凑拷贝前后检查每层索引、形状、dtype 与值，并观察原缓存是否已释放。GPU allocator 的活动字节和保留字节需分别记录，进程峰值可能仍受首次完整采集和模型权重支配。磁盘方面保留写入、校验读取和恢复的代价，同一文件的重复读取可能受操作系统页缓存影响。

D 的目录至少包含 `terminal_kv.safetensors` 和 `manifest.json`。manifest 记录原生 cache 格式、K/V 阶段、每层张量 shape/dtype、文件 SHA-256、兼容性元数据和自身指纹。加载时严格核对传入的兼容性元数据。目录在完整写入后才发布。终态 bank 由全部 Decoder 层组成，只能在与其位置与前缀匹配的实验路径上使用。

以下检查使用 CPU 和随机初始化的小型 Qwen2.5-VL。它执行真实的视觉与语言 forward，并比较裁去原始视觉行后的续写和稠密 cache 屏蔽相同行的续写。当前检查对多 token 续写采用 `atol=rtol=1e-6`，同时覆盖源引用释放、独立 storage、K/V 持久化和数据篡改拒绝。

```bash
PYTHONPATH=python OMP_NUM_THREADS=1 python -m unittest discover \
  -s test/srt/scene_graph_memory -p test_terminal_kv.py -v
```


## 本机输出

VisualProbe 的预检报告位于 `.runtime/terminal_state_reuse/visualprobe/experiment_config.json`。预检返回 2，列出三档 `val.json` 的完整缺失路径，因此没有发布 VisualProbe 的 `selected_manifest.json`。

CPU 全部终态测试与单独的 A6000 过程验证可以这样复现。GPU 验证使用真实 Mini-o3 processor 和工具、随机初始化的两层小型 BF16 Qwen2.5-VL、预先锁定的单个合成样本以及强制工具轨迹。输出用于程序验证，准确性和时间属于该人工诊断条件。

```bash
PYTHONPATH=python OMP_NUM_THREADS=1 python -m unittest discover \
  -s test/srt/scene_graph_memory -p 'test_terminal*.py' -v

CUDA_VISIBLE_DEVICES=0 SCENECACHE_TEST_TERMINAL_DEVICE=cuda:0 \
  SCENECACHE_TERMINAL_DIAGNOSTIC_DIR=.runtime/terminal_state_reuse/mechanism_gpu \
  PYTHONPATH=python OMP_NUM_THREADS=1 python -m unittest discover \
  -s test/srt/scene_graph_memory -p test_terminal_episode.py -v
```

GPU 过程输出包括 `mechanism.json`、两条路径的图像和 `trace.json`、两组实际前缀与位置、bank 校验清单和每次请求的 JSONL/CSV。时间包括 tokenizer 边界和显式同步，`episode_ms` 汇总该模式的共同采集成本、状态转换成本与各次真实问答耗时。A/B 先处理是为了从同一个源 KV 切出 B 并验证释放。两次独立请求和 C/D 的先后顺序使用样本种子固定随机化。首次 C/D 数值校验和 A/B 近似误差探针单列，排除在模式延迟之外。

A6000 的复用阶段共执行 16 次请求，覆盖两条路径上的原题和改问。两条路径的末次裁剪逐像素一致，规范化状态 hash 相同，并命中同一个 D bank。A、B、D 分别保留 354、286、139 行，独立 K/V 存储分别为 90,624、73,216、35,584 字节。C 每次重新编码一张裁剪图。D 加载后两次问答的视觉编码次数均为零，首次规范化 prefill 的视觉编码另计。

释放检查发现旧历史 K/V 张量与已知 view 基础张量的剩余引用均为零，B 与源存储指针不重叠。CUDA allocator 活跃存储由 48,502,784 降到 47,802,368 字节，reserved 保持 69,206,016 字节。活跃差值还含旧前缀的末位 logits，K/V 单独按 storage 字节计量。C/D teacher forcing 的 logits 最大绝对误差为 0.00390625，归一化 RMSE 为 0.00250，固定的 9 个 token 位置上 top-1 一致。A/B 的 logits 最大绝对差为 0.04898，反映删历史行后的条件变化。

上述 VisualProbe 预检和人工诊断完成时，CPU Scene Graph Memory 回归套件运行 62 项，54 项通过，8 项 CUDA 专用用例在无 CUDA 配置中跳过。独立 A6000 终态集成测试通过。

### 本机 GQA 实测

输出位于 `.runtime/terminal_state_reuse/gqa_10`。两阶段命令使用 `CUDA_VISIBLE_DEVICES=1`、`--dataset gqa`、`--num-images 10`、`--questions-per-image 3` 和 `--skip-controlled-replay`。10 张图均取得模型自主选择的裁剪，共执行 120 次问答。

A/B/C/D 的全题精确匹配分别为 17/30、16/30、17/30、17/30，追问分别为 10/20、9/20、11/20、11/20。C 和 D 在 30 组配对中的回答和状态一致。D 的模型 prefill 中位 35.44 ms，C 为 98.85 ms。把首次构建和加载计入每图三题后，D−C 的配对 episode 中位差为 −81.88 ms，10 张中有 6 张更快。HF 停止条件的重复 CPU 构建占了本次 TTFT 的主要部分，单独探针测得 2.11–2.83 s。该批次用于观察机制与失败样例。

首图独立使用真实 Mini-o3-7B-v1 重建规范前缀，和落盘重载逐元素核对，28 层 K/V 均相同。相同分段的续写 logits 最大误差为零。单次完整 prefill 与重新计算的分段 prefill 之间，BF16 logits 的最大误差为 0.375，归一化 RMSE 为 0.01653，超过原先设定的 0.01。逐题答案见 `gqa_results.md`，原始请求见 `requests.csv` 和 `requests.jsonl`，逐层校验见 `real_bank_verification.json`。
