# Visual KV inference in SceneCache

The experimental modes run on this fork's Qwen3-VL decoder with one still image, one active request, TP=PP=DP=1, and BF16 weights and KV on the tested A6000. They use the existing Scene Graph memory for manual region selection. See [building a memory](scene_graph_memory.md) for the image, SAM3 and canonical prefill workflow.

## Serving

Use the existing `scenecache` conda environment and local model. For a saved memory, write the selection JSON before sending requests. Expand contact hops or include the full visual grid using the same format.

```json
{"region_ids": ["r0001"], "expand_hops": 1}
```

```bash
conda activate scenecache
export CUDA_VISIBLE_DEVICES=3
export SGLANG_VISUAL_KV_MODE=region_pipeline
export SGLANG_VISUAL_KV_MEMORY="$PWD/.runtime/scene_graph_memory_market"
export SGLANG_VISUAL_KV_SELECTION="/absolute/path/to/selection.json"
export SGLANG_VISUAL_KV_METRICS="/absolute/path/to/inference.jsonl"

python -m sglang.launch_server \
  --model-path /home/hdd4/lpz/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 --attention-backend flashinfer --mm-attention-backend triton_attn \
  --host 127.0.0.1 --port 30008 --max-running-requests 1 \
  --chunked-prefill-size -1 --disable-cuda-graph --disable-radix-cache \
  --disable-overlap-schedule --context-length 2048 --max-total-tokens 2176 \
  --mem-fraction-static 0.97
```

| Setting for `SGLANG_VISUAL_KV_MODE` | Visual KV source | Decoder prefill | Decoder decode |
| --- | --- | --- | --- |
| unset or `legacy` | Existing `MockKVManager` | Existing FlashInfer sparse path | Existing FlashInfer paged path |
| `pipelined_full_kv` | Same `MockKVManager` entry as legacy | Full visual set with bounded cross-layer prefetch, FlashInfer | Existing FlashInfer paged path |
| `region_reference` | Saved Scene Graph bank | Synchronous gather and transfer, FlashInfer | FlashInfer, selected slots |
| `region_pipeline` | Same manual selection and bank | Pinned staging and prefetch, FlashInfer | FlashInfer, selected slots |
| `region_pipeline_triton` | Same manual selection and bank | Pinned staging and prefetch, new compact Triton attention | FlashInfer, selected slots |

A full pipeline cache miss runs the original prefill and records the miss. Region modes require the bank's exact image pixels and image-first text prefix. They validate the source model files and processor when booting, validate the selected image and original MRoPE positions, and reject unsupported selection configurations. Replace the selection JSON atomically between requests. `all_visual: true` recovers the full visual set. `include_unassigned: true` adds uncovered grid rows. `active_budget` is an optional budget experiment that fills nearest grid positions around the specified region tokens. It does not implement question-based retrieval.

`SGLANG_VISUAL_KV_PREFETCH_DEPTH` defaults to 2. `SGLANG_VISUAL_KV_STAGING_BYTES` defaults to 32 MiB for the pinned ring and separately 32 MiB for the device ring. A setting smaller than the required ring produces a budget error. The ring covers staging. The regular paged KV pool retains restored KV for the duration of decode. The cache stores the encoder feature separately and keeps its full image embedding in all modes.

## Token and cache lifecycle

- **N_all** counts the source visual rows. **N_active** counts the visual rows visible to attention in the given layer. **N_recompute** counts the visual rows for which prefill computes fresh queries and K/V. This implementation uses a uniform recompute policy across layers. Nonvisual prefix and question tokens are separate.
- The tokenizer compacts physical scheduler slots for the selected image rows. It retains original MRoPE for selected rows and text, and adjusts the delta so decode starts at the original next position. The selected indices retain original causal order. The Triton prefill uses original logical positions for its causal mask. FlashInfer uses the same order in its block metadata. Decode uses the compact page table.
- The decoder fetches cached rows for (S \\ R). K is a raw projection and passes through the existing per-head RMSNorm and MRoPE before the paged write. V is a raw projection that needs neither transform. Fresh K/V overwrite their own rows. DeepStack embeddings retain the same selected image provenance.
- The transfer ring keeps host memory alive until H2D completes, and device staging until the compute consumer releases the slot. A dedicated stream copies future layers. The decoder waits on a ready event at consumption and stores the restored layer in the existing request KV pool.
- The new Triton kernel reads compact Q/K/V, handles grouped-query head mapping and masked tails, and computes QK, one online softmax over all visible blocks, and PV. The decoder uses it for selected-region prefill. Vision encoding and paged decode use the inherited backends.

Full pipelining preserves the legacy visual/query sets. Region selection changes the visible evidence and can change the answer even with numerically equivalent implementations of the selected path. The source bank still contains states produced in full image context.

## Reproducing measurements

```bash
python examples/scene_graph_memory/benchmark_inference.py \
  --gpu 3 --memory-dir .runtime/scene_graph_memory_market --fixed-output-tokens 8 \
  --out .runtime/scene_graph_inference_bench/reproduce \
  --diagnostic-cases examples/scene_graph_memory/market_quality_qa.json \
                     examples/scene_graph_memory/market_active_budgets.json
```

The runner starts local isolated services, preserves original samples and runtime counters, and verifies the image grid and runtime mode. The default QA fixture tests a fine-detail question about the market image. Separate fixtures `market_quality_qa.json` and `market_active_budgets.json` cover independent questions and four active budgets. Keep profiler and probe runs separate from performance samples.

Set `SGLANG_VISUAL_KV_VERIFY_DIR` for small real-request probe runs to capture raw K/V, restored K/V, attention, decoder outputs, and logits at representative layers. Leave this unset for latency measurements. Run `examples/scene_graph_memory/compare_inference_probes.py --help` for numerical comparisons and `examples/scene_graph_memory/analyze_cuda_trace.py --help` to measure H2D/kernel overlap from an exported Nsight Systems SQLite trace.

The measured checkout was `feature/support_kvcache_reuse_053rc2` at `9c2935359cd53d64c53a2a8a48a0728c232b1c4e` with the local SceneCache changes. The physical GPU was RTX A6000 #3 with 49,140 MiB VRAM, SM 8.6, PCIe 4.0 ×16, and NUMA affinity to node 1. Existing workloads shared this GPU during sampling. The software stack was driver 580.173.02, Python 3.11.18, PyTorch 2.8.0+cu128, Triton 3.4.0, FlashInfer 0.4.0rc3, Transformers 4.57.0 and SGLang 0.5.3rc2.

The measured source uses the local Qwen3-VL-8B-Instruct checkpoint and `/home/lipz/test_market.jpg`, with a 1344-pixel maximum submitted edge, a 1344 × 672 preprocessed image and a 21 × 42 merged grid. The model has 36 layers, 32 query heads, 8 KV heads, and head dimension 128. The saved bank occupies 124.03 MiB of raw BF16 KV and contains 16 SAM3 regions and 17 contact edges. Model loading, memory fingerprinting, SAM3, bank creation, and initial compilation are outside the warm inference table. The existing bank was built with the local SAM3 tools under `/home/lipz/Scene2Graph` and the checkpoint under `/home/hdd4/lpz/SAM3/checkpoints`.

For initialization cost, the saved build's log and output timestamps span about 92 seconds. About 48 seconds elapsed before logging the SAM3 mask count, including model file hashing and mask generation, and the rest covered Qwen initialization, full KV capture and verification, and persistence. These are estimates from the original build artifacts. In the reuse benchmark, new worker weight loading and one-time compilation happen before the formal samples. The raw warmup rows retain per-request latency including shape-specific compilation.

## Warm latency on the market image

These modes use the same image processing, model, question, BF16 dtype, seed, eight generated tokens and seven decode steps. There were five warmups and 20 measured requests per mode. The four full-image modes were interleaved in one server and the three region modes in another. The corrected cold and feature-only controls bypass image decoder bank writes. Cold also bypasses this fork's CPU vision-feature cache. Feature-only still hits the GPU encoder embedding cache. The full and legacy prefill use the same visual attention set and recompute 264 rows.

| Mode | TTFT p50 / p95 (ms) | Server TPOT p50 (ms) | Client total p50 (ms) | Server prefill p50 (ms) | N_active / N_recompute | Visual KV H2D (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cold_full` | 1045.9 / 1130.0 | 54.6 | 1443.6 | 807.6 | 882 / 882 | 0 |
| `feature_only` | 620.0 / 674.7 | 58.6 | 1033.0 | 381.4 | 882 / 882 | 0 |
| `legacy_full_kv` | 674.8 / 740.0 | 55.2 | 1071.0 | 445.7 | 882 / 264 | 124.03 |
| `pipelined_full_kv` | 451.5 / 500.2 | 60.0 | 883.4 | 195.8 | 882 / 264 | 86.91 |
| `region_reference` | 451.7 / 500.5 | 61.1 | 889.5 | 196.6 | 99 / 29 | 9.84 |
| `region_pipeline` | 418.3 / 463.0 | 62.5 | 846.6 | 153.6 | 99 / 29 | 9.84 |
| `region_pipeline_triton` | 391.4 / 481.4 | 60.0 | 825.3 | 124.0 | 99 / 29 | 9.84 |

TTFT is client time through the first nonempty streamed text. Server TPOT is scheduler decode wall time per decode step, sampled at completed CUDA work. Snapshot rows include both the client and server clocks. H2D in the table counts visual K/V only. The optimized full pipeline also copied 30 index bytes, the region FlashInfer paths 822 bytes, and the Triton path 1,876 bytes for this selection. Backend planning may issue additional metadata copies.

The full pipeline lowered median TTFT by 168.5 ms (27.2%) relative to feature-only and 223.3 ms (33.1%) relative to legacy KV. Median server TPOT was 1.4 ms and 4.8 ms higher, respectively. It preserves the native decode path. Legacy's median prefill was 64.2 ms slower than feature-only despite smaller query sets. Its repeated synchronizations and host KV copies appear on the timeline below.

All four full visual modes generated the same answer on each short QA case. They answered `白色`, `米色`, `番茄`, and `钱` in fixture order, scoring 3/4 under the supplied labels. The legacy and pipelined full modes had bitwise equal restored K/V, attention, and decoder layer outputs at sampled layers 0, 18 and 35, with bitwise equal first-step logits. The source bank and positions were shared for the system comparison.

The full pipeline used 4.83 MiB each of pinned and device staging. The 99-row region pipeline used 0.55 MiB each. The logical live paged prompt KV payload was 129.09 MiB for the full pipeline and 18.98 MiB for the region modes. The actual worker GPU memory has a preallocated pool and model weights. Median per-request PyTorch peak allocations were about 19.17–19.34 GB. The harness resets these peaks before prefill. The CPU raw bank occupies 124.03 MiB for both the whole and regional paths.

Raw samples, merged results and QA diagnostics are in `.runtime/scene_graph_inference_bench/matched_7_modes`. Exact full and region pipeline probe comparisons are `full_pipeline_strict.json` and `region_pipeline_strict.json` in the benchmark parent directory.

## Measured CUDA stream overlap

This diagnostic ran separately from latency sampling. It captured one warmed market prefill for each of the two full KV modes in a single service.

```bash
python examples/scene_graph_memory/benchmark_inference.py \
  --gpu 3 --memory-dir .runtime/scene_graph_memory_market \
  --out .runtime/scene_graph_inference_bench/full_nsys \
  --modes legacy_full_kv pipelined_full_kv --smoke --warmup 1 --rounds 1 \
  --nsys /usr/local/cuda-12.8/bin/nsys
```

| CUDA activity in the captured prefill | Legacy | Pipeline |
| --- | ---: | ---: |
| Visual K/V H2D calls | 72 | 72 |
| Visual K/V H2D payload | 124.03 MiB | 86.91 MiB |
| Summed K/V DMA duration | 8.368 ms | 3.921 ms |
| DMA intersecting other-stream kernels | 0 ms | 3.166 ms |
| DMA calls with overlap | 0 | 64 |
| CUDA device-wide synchronization calls in capture | 38 | 2 |

The pipeline trace, `full_nsys/full_group/prefill_profile.1.nsys-rep`, has an H2D event on stream 25 for layer 15 at 159.556–159.654 ms. A compute kernel on stream 7 overlaps it from 159.575 to 159.654 ms. The 80.75% overlap figure uses the union of simultaneous kernels for each copy, correlated on the same CUDA context. The legacy trace is `prefill_profile.2.nsys-rep`. `full_nsys/overlap_1.json` and `overlap_2.json` include the exact intervals and CUPTI source memory types.

Legacy uses blocking pageable copies and has one explicit device-wide barrier inside each decoder layer lookup. The pipeline waits on ready events on the consumer stream. In the traced request, its two-slot ring recorded about 11.53 ms CPU staging time, 16.64 ms of host-side wait for DMA ownership on buffer reuse, 0.108 ms of GPU wait at consumption, and 36 prefetch hits with zero demand misses. Those measurements can overlap and use different clocks from client latency.

The profiled pipeline's kernel unions attributed 80.18 ms to MLP, 10.07 ms to QKV projection, 5.88 ms to decoder attention, 5.18 ms to the new-row norm and MRoPE, and 1.46 ms to cached-key norm and MRoPE. The trace supports leaving KV restoration in the model's verified operations. Main latency results include interleaved requests on the shared GPU and exclude profiling.

For CPU packing, the measured direct `index_select` into pinned slices took about 15–21 µs per K/V pair for the three fragmented selections, versus 119–281 µs for copying each contiguous interval individually with PyTorch. The full visual suffix was one contiguous interval and both methods took about 51 µs. See `cpu_pack.json` in the benchmark output directory.

## Region behavior and equivalence

The fixed selection `{"region_ids":["r0001"],"expand_hops":1}` retained 99 of 882 visual rows. Each layer recomputed 29, and CPU packing and H2D handled the remaining 70 visual K/V pairs. All three region modes transferred 9.844 MiB of visual KV per request. The selected prompt used 135 total slots, and seven decode steps read 136 through 142 physical KV slots. The visual bank recorded zero extra H2D in decode.

The numeric comparison used separate instrumented requests. Raw K/V, restored K/V, attention outputs, decoder layer outputs and logits were bitwise equal between `region_reference` and `region_pipeline` at sampled layers 0, 18 and 35. Both produced the same forced eight-token continuation and the same short QA answers. A real streaming test closed an SSE connection after its first text, stopped after 36 of the 511 possible decode steps, and returned the allocator to idle. It then changed the selection from four tokens in `r0001` to three in `r0002`, and two subsequent eight-token requests completed with the new selection and expected transfer bytes.

`region_pipeline_triton` used the new attention kernel during the real decoder prefill. Same-input replay on captured Q/K/V using the original logical masks passed the BF16 kernel tolerance `atol=0.012, rtol=0.012` against FP32 attention at layers 0, 18 and 35. Its restored cached K/V were bitwise equal to the reference. For the market face mask question, layer 0 attention differed from FlashInfer by at most 0.00098 and the sampled end-to-end logits differed by up to 0.375. The reference first-token logits for blue and white were 44.0 and 43.75. The Triton logits tied at 43.75 and the greedy output switched to white. Its end-to-end equivalence check therefore fails for this example. Keep this backend opt-in for experiments.

An isolated GPU microbenchmark replayed the real layer 0 Q/K/V with 65 queries and 135 keys. Warm CUDA graph wrapper medians were 11.99 µs for Triton and 29.50 µs for the existing FlashInfer VariableBlockSparse path. Separate CUPTI means found 12.35 µs in the Triton kernel and 11.25 µs in FlashInfer's attention kernel, plus 13.51 µs of FlashInfer layout copies and 7.20 µs of merge work. The wrapper savings come from removing those extra operations. The microbenchmark excludes planning, staging, JIT, and the rest of the model. See `real_attention_kernel_bench.json` or re-run it with `scripts/benchmark_scene_graph_attention.py --probe-file PATH --layer 0`.

The four independent short questions produced these outputs on their manually selected regions.

| Question | Accepted answer | Sync reference | Pipeline | Pipeline with Triton |
| --- | --- | --- | --- | --- |
| Mask behind the beige headscarf | Light blue | 蓝色 | 蓝色 | 白色 |
| Headscarf on the woman handing over money | Beige | 米色 | 米色 | 米色 |
| Red vegetables in the lower-left pile | Tomatoes | 番茄 | 番茄 | 番茄 |
| What the woman hands to the man | Banknote | 纸张 | 纸张 | 纸张 |

With the three direct comparisons above, the exact-answer counts are 3/4, 3/4 and 2/4. This fixture has one image and four questions.

The budget sweep used the same mask question, with `r0001` as a seed and nearest grid rows filling the budget. This is a different selection from the 99-row contact-graph example.

| Visual budget | N_active | N_recompute | Total prefill KV slots | Causal QK pairs per head | Visual KV H2D | Answer in all three region modes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 100% | 882 | 264 | 918 | 64,926 | 86.91 MiB | 白色 |
| 50% | 441 | 132 | 477 | 24,084 | 43.45 MiB | 白色 |
| 25% | 221 | 66 | 257 | 10,213 | 21.80 MiB | 白色 |
| 10% | 89 | 26 | 125 | 3,969 | 8.86 MiB | 白色 |

The causal query-key pair count is computed from the actual prompt lengths and sets. Kernels can still execute tile padding. The final 99-token region smoke logged 4,385 causal pairs per head with seed `r0001` and expanded regions `r0001`, `r0006` and `r0014`.

See `fixed_region_v2/diagnostic_samples.jsonl`, `fixed_region_v2/cancellation.json` and `real_attention_replay.json` in the benchmark directory.
