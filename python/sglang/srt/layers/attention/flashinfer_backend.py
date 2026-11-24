from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import os
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch

if os.environ["SGLANG_ENABLE_TORCH_COMPILE"] == "1":
    import logging

    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True

from sglang.global_config import global_config
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import (
    get_int_env_var,
    is_flashinfer_available,
    is_sm100_supported,
    next_power_of_2,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
        fast_decode_plan,
        VariableBlockSparseAttentionWrapper,
    )
    from flashinfer.cascade import merge_state
    from flashinfer.decode import _get_range_buf, get_seq_lens

from sglang.srt.managers.mock_kv_manager import mock_kv_manager

class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


@dataclass
class DecodeMetadata:
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]


@dataclass
class PrefillMetadata:
    prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]
    use_ragged: bool
    extend_no_prefix: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None

# Use as a fast path to override the indptr in flashinfer's plan function
# This is used to remove some host-to-device copy overhead.
global_override_indptr_cpu = None


class FlashInferAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Parse constants
        self.decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype=model_runner.kv_cache_dtype,
            num_attention_heads=model_runner.model_config.num_attention_heads
            // get_attention_tp_size(),
            num_kv_heads=model_runner.model_config.get_num_kv_heads(
                get_attention_tp_size()
            ),
        )
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        if model_runner.sliding_window_size is not None:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW
        elif model_runner.model_config.is_encoder_decoder:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.CROSS_ATTENTION
        else:
            self.num_wrappers = 1
            self.dispatch_reason = None

        # Qwen2/Qwen3 models require higher flashinfer workspace size
        if (
            "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
        ):
            global_config.flashinfer_workspace_size = 512 * 1024 * 1024

        # When deterministic inference is enabled, tensor cores should be used for decode
        # Also set split tile sizes for prefill and decode from environment variables, and disable kv split for cuda graph
        # More information can be found here: https://github.com/flashinfer-ai/flashinfer/pull/1675
        self.enable_deterministic = (
            model_runner.server_args.enable_deterministic_inference
        )
        self.prefill_split_tile_size = None
        self.decode_split_tile_size = None
        self.disable_cuda_graph_kv_split = False
        if self.enable_deterministic:
            self.decode_use_tensor_cores = True
            self.prefill_split_tile_size = get_int_env_var(
                "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096
            )
            self.decode_split_tile_size = get_int_env_var(
                "SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE", 2048
            )
            self.disable_cuda_graph_kv_split = True
            global_config.flashinfer_workspace_size = 2048 * 1024 * 1024

        # Allocate buffers
        global global_workspace_buffer
        if global_workspace_buffer is None:
            # different from flashinfer zero_init_global_workspace_buffer
            global_workspace_size = global_config.flashinfer_workspace_size
            global_workspace_buffer = torch.empty(
                global_workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_workspace_buffer
        max_bs = model_runner.req_to_token_pool.size
        if kv_indptr_buf is None:
            self.kv_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]
        else:
            assert self.num_wrappers == 1
            self.kv_indptr = [kv_indptr_buf]

        if kv_last_page_len_buf is None:
            self.kv_last_page_len = torch.ones(
                (max_bs,), dtype=torch.int32, device=model_runner.device
            )
        else:
            assert self.num_wrappers == 1
            self.kv_last_page_len = kv_last_page_len_buf

        if not self.skip_prefill:
            self.qo_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]

        fmha_backend = "auto"
        if is_sm100_supported():
            fmha_backend = "cutlass"
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=fmha_backend
        )

        # Two wrappers: one for sliding window attention and one for full attention.
        # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        self.prefill_wrappers_paged = []
        self.prefill_wrappers_verify = []
        self.decode_wrappers = []
        self.prefill_wrappers_variable_block_dict = {}

        recompute_ratio_in_layer = getattr(model_runner.model_config.hf_config, "recompute_ratio_in_layer", 0.3)
        if isinstance(recompute_ratio_in_layer, float):
            assert recompute_ratio_in_layer >= 0 and recompute_ratio_in_layer <= 1
            recompute_ratio_in_layer = [recompute_ratio_in_layer for i in range(model_runner.model_config.num_hidden_layers)]
        else:
            assert isinstance(recompute_ratio_in_layer, list)
            assert len(recompute_ratio_in_layer) == model_runner.model_config.num_hidden_layers
            assert all(a >= b for a, b in zip(recompute_ratio_in_layer, recompute_ratio_in_layer[1:]))
            assert min(recompute_ratio_in_layer) >= 0 and max(recompute_ratio_in_layer) <= 1
        self.recompute_ratio_in_layer = recompute_ratio_in_layer
        for _ in range(self.num_wrappers):
            if not skip_prefill:
                self.prefill_wrappers_paged.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend="fa2",
                    )
                )
                self.prefill_wrappers_verify.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                    )
                )
                for i in range(model_runner.model_config.num_hidden_layers):
                    if self.recompute_ratio_in_layer[i] not in self.prefill_wrappers_variable_block_dict:
                        self.prefill_wrappers_variable_block_dict[self.recompute_ratio_in_layer[i]] = VariableBlockSparseAttentionWrapper(
                            self.workspace_buffer,
                        )
            self.decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                )
            )

        # Create indices updater
        if not skip_prefill:
            self.indices_updater_prefill = FlashInferIndicesUpdaterPrefill(
                model_runner, self
            )  # for verify
        self.indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)

        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None
        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}  # For verify
        self.draft_extend_cuda_graph_metadata = {}  # For draft extend

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                decode_wrappers=self.decode_wrappers,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
                fixed_split_size=self.decode_split_tile_size,
                disable_split_kv=False,
            )
            self.forward_metadata = DecodeMetadata(self.decode_wrappers)
            if os.getenv("SGLANG_VISUAL_KV_MODE", "").startswith("region_"):
                from sglang.srt.mem_cache.scene_graph_memory.optimized_prefill import observe_region_decode

                observe_region_decode(forward_batch)
        elif forward_batch.forward_mode.is_draft_extend():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_wrappers_paged,
                use_ragged=False,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_paged, False, False
            )
        elif forward_batch.forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_wrappers_verify,
                use_ragged=False,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_verify, False, False
            )
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            if self.is_multimodal:
                use_ragged = False
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            else:
                use_ragged = True
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            
            # fixme: use config to choose
            if True:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_cpu,
                    forward_batch.seq_lens_sum,
                    prefix_lens,
                    prefill_wrappers=self.prefill_wrappers_variable_block_dict,
                    use_ragged=use_ragged,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=None,
                    forward_batch=forward_batch,
                )
                self.forward_metadata = PrefillMetadata(
                    self.prefill_wrappers_variable_block_dict, use_ragged, extend_no_prefix
                )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_cpu,
                    forward_batch.seq_lens_sum,
                    prefix_lens,
                    prefill_wrappers=self.prefill_wrappers_paged,
                    use_ragged=use_ragged,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=None,
                )
                self.forward_metadata = PrefillMetadata(
                    self.prefill_wrappers_paged, use_ragged, extend_no_prefix
                )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        if kv_indices_buf is None:
            cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len,),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + [
            cuda_graph_kv_indices.clone() for _ in range(self.num_wrappers - 1)
        ]

        # Ensure tensors are properly allocated
        for i in range(self.num_wrappers):
            # Force allocation by performing a small operation
            if len(self.cuda_graph_kv_indices[i]) > 0:
                self.cuda_graph_kv_indices[i][0] = 0

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device="cuda",
            )
            self.cuda_graph_qk_indptr = [x.clone() for x in self.kv_indptr]
            self.cuda_graph_qo_indptr = [x.clone() for x in self.kv_indptr]

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        if forward_mode.is_decode_or_idle():
            decode_wrappers = []
            for i in range(self.num_wrappers):
                decode_wrappers.append(
                    BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=self.decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.kv_indptr[i][: num_tokens + 1],
                        paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buffer=self.kv_last_page_len[
                            :num_tokens
                        ],
                    )
                )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_decode.update(
                req_pool_indices,
                seq_lens,
                seq_lens.cpu(),  # may add a little overhead in capture stage
                seq_lens_sum,
                decode_wrappers=decode_wrappers,
                encoder_lens=encoder_lens,
                spec_info=spec_info,
                fixed_split_size=None,
                disable_split_kv=self.disable_cuda_graph_kv_split,
            )
            self.decode_cuda_graph_metadata[bs] = decode_wrappers
            self.forward_metadata = DecodeMetadata(decode_wrappers)
            for i in range(self.num_wrappers):
                decode_wrappers[i].begin_forward = partial(
                    fast_decode_plan, decode_wrappers[i]
                )
        elif forward_mode.is_target_verify():
            prefill_wrappers = []
            for i in range(self.num_wrappers):
                prefill_wrappers.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                        paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                        paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                        custom_mask_buf=self.cuda_graph_custom_mask,
                        mask_indptr_buf=self.cuda_graph_qk_indptr[i][: bs + 1],
                    )
                )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens.cpu(),  # may add a little overhead in capture stage
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=prefill_wrappers,
                use_ragged=False,
                encoder_lens=encoder_lens,
                spec_info=spec_info,
            )
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
        elif forward_mode.is_draft_extend():
            prefill_wrappers = []
            for i in range(self.num_wrappers):
                prefill_wrappers.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend="fa2",
                        use_cuda_graph=True,
                        qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                        paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                        paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                    )
                )

            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens.cpu(),  # may add a little overhead in capture stage
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=prefill_wrappers,
                use_ragged=False,
                encoder_lens=encoder_lens,
                spec_info=spec_info,
            )
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        if forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                decode_wrappers=self.decode_cuda_graph_metadata[bs],
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
                fixed_split_size=None,
                disable_split_kv=self.disable_cuda_graph_kv_split,
            )
        elif forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        elif forward_mode.is_draft_extend():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        else:
            raise ValueError("Invalid forward mode")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        visual = forward_batch.optimized_visual_kv
        if visual is not None and visual.mode == "region_pipeline_triton":
            from sglang.srt.mem_cache.scene_graph_memory.optimized_prefill import profile_range
            from sglang.srt.mem_cache.scene_graph_memory.triton_attention import triton_attention

            with profile_range("scene_kv.triton_attention"):
                output = triton_attention(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    query_positions=visual.logical_query_positions,
                    key_positions=visual.logical_key_positions,
                    key_positions_sorted=True,
                    scale=layer.scaling,
                )
            if save_kv_cache:
                with profile_range("scene_kv.paged_kv_write"):
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, forward_batch.out_cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
            return output.reshape(-1, layer.tp_q_head_num * layer.head_dim)

        prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[
            self.recompute_ratio_in_layer[layer.layer_id]
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        logits_soft_cap = layer.logit_cap

        q = q.contiguous()
        if not self.forward_metadata.use_ragged:
            # fixme: use config to choose
            if True:
                # fixme: in radix cache, here need to get former kv from token_to_kv_pool
                if self.forward_metadata.extend_no_prefix:
                    o = prefill_wrapper_paged.run(
                        q.view(-1, layer.tp_q_head_num, layer.head_dim).transpose(0,1),
                        k.view(-1, layer.tp_k_head_num, layer.head_dim).transpose(0,1),
                        v.view(-1, layer.tp_v_head_num, layer.head_dim).transpose(0,1)
                    )
                    o = o.transpose(0,1).contiguous() #head_num, token, head_dim
                else:
                    # support chunked prefill (radix cache no test)
                    o1, s1 = prefill_wrapper_paged.run(
                        q.view(-1, layer.tp_q_head_num, layer.head_dim).transpose(0,1),
                        k.view(-1, layer.tp_k_head_num, layer.head_dim).transpose(0,1),
                        v.view(-1, layer.tp_v_head_num, layer.head_dim).transpose(0,1),
                        return_lse=True
                    )
                    o1 = o1.transpose(0,1).contiguous() #head_num, token, head_dim
                    s1 = s1.transpose(0,1).contiguous() #head_num, token

                    # erxi: currently, to support chunked prefill, we need to cal attn result of current q and old k/v, but
                    # using BatchPrefillWithPagedKVCacheWrapper and VariableBlockSparseAttentionWrapper at the same time will
                    # cause illegal memory access during inference, thus here only use a torch implementation which is adapt
                    # from torch_native_backend and should be improve in future.
                    def cal_chunk(
                        query: torch.Tensor,
                        output: torch.Tensor,
                        lse: torch.Tensor,
                        k_cache: torch.Tensor,
                        v_cache: torch.Tensor,
                        req_to_token: torch.Tensor,
                        req_pool_indices: torch.Tensor,
                        seq_lens: torch.Tensor,
                        extend_seq_lens: torch.Tensor,
                        orig_seq_lens: torch.Tensor,
                        ):
                        # [num_q_tokens, num_heads, head_size] -> [num_heads, num_q_tokens, head_size]
                        query = query.movedim(0, query.dim() - 2)
                        start_q, start_kv = 0, 0
                        num_qo_heads, _ , head_dim = query.shape #fixme
                        num_kv_heads = k_cache.shape[1]
                        for seq_idx in range(seq_lens.shape[0]):
                            # TODO: this loop process a sequence per iter, this is inefficient.
                            # Need optimize the performance later.

                            extend_seq_len_q = extend_seq_lens[seq_idx]
                            seq_len_kv = seq_lens[seq_idx]
                            end_q = start_q + extend_seq_len_q
                            end_kv = start_kv + seq_len_kv
                            
                            if extend_seq_len_q == orig_seq_lens[seq_idx]:
                                start_q, start_kv = end_q, end_kv
                                continue
                            per_req_query = query[:, start_q:end_q, :]

                            # get key and value from cache. per_req_tokens contains the kv cache
                            # index for each token in the sequence.
                            req_pool_idx = req_pool_indices[seq_idx]
                            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
                            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2) # [num_heads, num_kv_tokens, head_size]
                            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2) # [num_heads, num_kv_tokens, head_size]
                            if num_qo_heads != num_kv_heads:
                                per_req_key = per_req_key.repeat_interleave(num_qo_heads // num_kv_heads, dim=0)
                                per_req_value = per_req_value.repeat_interleave(num_qo_heads // num_kv_heads, dim=0)

                            attn_scores = torch.matmul(per_req_query, per_req_key.transpose(-2, -1)) / (head_dim ** 0.5) # [num_heads, num_q_tokens, num_kv_tokens]
                            attn_lse = torch.logsumexp(attn_scores, dim=-1) # [num_heads, num_q_tokens]
                            attn_lse = attn_lse.transpose(0, 1) # [num_q_tokens, num_q_tokens]
                            attn_lse /= 0.6931 # div by ln2 because logsumexp in flashinfer_backend differs from that of torch_native_backend
                            attn_weights = torch.softmax(attn_scores, dim=-1) # [num_heads, num_q_tokens, num_kv_tokens]
                            attn_weights[torch.isnan(attn_weights)] = 0
                            per_req_out = torch.matmul(attn_weights, per_req_value)  # [num_heads, num_q_tokens, head_size]
                            per_req_out = per_req_out.movedim(query.dim() - 2, 0) #[num_q_tokens, num_heads, head_size]

                            output[start_q:end_q, :, :] = per_req_out
                            lse[start_q:end_q, :] = attn_lse
                            start_q, start_kv = end_q, end_kv

                    o2 = torch.zeros((o1.shape[0], layer.tp_q_head_num, layer.head_dim), dtype=q.dtype, device=q.device)
                    s2 = torch.zeros((o1.shape[0], layer.tp_q_head_num), dtype=q.dtype, device=q.device)

                    cal_chunk(
                        q.clone().view(-1, layer.tp_q_head_num, layer.head_dim),
                        o2,
                        s2,
                        forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                        forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                        forward_batch.req_to_token_pool.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.extend_seq_lens,
                        forward_batch.orig_seq_lens,
                        )
                    o, _ = merge_state(o1, s1, o2, s2)

                # save kvcache after this chunk is calculated
                if k is not None:
                    assert v is not None
                    if save_kv_cache:
                        if visual is not None:
                            from sglang.srt.mem_cache.scene_graph_memory.optimized_prefill import profile_range

                            with profile_range("scene_kv.paged_kv_write"):
                                forward_batch.token_to_kv_pool.set_kv_buffer(
                                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                                )
                        else:
                            forward_batch.token_to_kv_pool.set_kv_buffer(
                                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                            )
            else:
                o = prefill_wrapper_paged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=not layer.is_cross_attention,
                    sm_scale=layer.scaling,
                    window_left=layer.sliding_window_size,
                    logits_soft_cap=logits_soft_cap,
                    k_scale=layer.k_scale,
                    v_scale=layer.v_scale,
                )
        else:
            causal = True
            if layer.attn_type == AttentionType.ENCODER_ONLY:
                save_kv_cache = False
                causal = False
            if self.forward_metadata.extend_no_prefix:
                # NOTE: FlashInfer currently has limitations with head_dim = 32 or other dimensions
                # The FlashInfer head_dim limitation itself is tracked here:
                # https://github.com/flashinfer-ai/flashinfer/issues/1048
                o = self.prefill_wrapper_ragged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=causal,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )

            else:
                o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=True,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )
                o2, s2 = prefill_wrapper_paged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=False,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )

                o, _ = merge_state(o1, s1, o2, s2)

            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        decode_wrapper = self.forward_metadata.decode_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Call the wrapped function
        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _get_wrapper_idx(self, layer: RadixAttention):
        if self.num_wrappers == 1:
            return 0

        if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            return layer.sliding_window_size == -1
        if self.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            return layer.is_cross_attention

        raise ValueError(f"Unknown dispatch reason: {self.dispatch_reason}")


class FlashInferIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator

        # Dispatch the update function
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        decode_wrappers = decode_wrappers or self.decode_wrappers
        self.call_begin_forward(
            decode_wrappers[0],
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.kv_indptr[0],
            None,
            spec_info,
            seq_lens_cpu,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        assert self.sliding_window_size is not None
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Sliding window attention
                paged_kernel_lens_tmp = torch.clamp(
                    seq_lens, max=self.sliding_window_size + 1
                )
                if seq_lens_cpu is not None:
                    seq_lens_cpu_tmp = torch.clamp(
                        seq_lens_cpu, max=self.sliding_window_size + 1
                    )
                    paged_kernel_lens_sum_tmp = seq_lens_cpu_tmp.sum().item()
                else:
                    paged_kernel_lens_sum_tmp = paged_kernel_lens_tmp.sum().item()
                kv_start_idx_tmp = seq_lens - paged_kernel_lens_tmp
            else:
                # Full attention
                paged_kernel_lens_tmp = seq_lens
                paged_kernel_lens_sum_tmp = seq_lens_sum
                seq_lens_cpu_tmp = seq_lens_cpu
                kv_start_idx_tmp = None

            use_sliding_window_kv_pool = wrapper_id == 0 and isinstance(
                self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
            )

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens_tmp,
                paged_kernel_lens_sum_tmp,
                self.kv_indptr[wrapper_id],
                kv_start_idx_tmp,
                spec_info,
                seq_lens_cpu=seq_lens_cpu_tmp,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
            else:
                # Cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                seq_lens_sum = encoder_lens.sum().item()

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                seq_lens_sum,
                self.kv_indptr[wrapper_id],
                kv_start_idx,
                spec_info,
                seq_lens_cpu=seq_lens_cpu,
            )

    def call_begin_forward(
        self,
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        if spec_info is None:
            bs = len(req_pool_indices)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            if wrapper.is_cuda_graph_enabled:
                # Directly write to the cuda graph input buffer
                kv_indices = wrapper._paged_kv_indices_buf
            else:
                kv_indices = torch.empty(
                    paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
                )

            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
        else:
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
            bs = kv_indptr.shape[0] - 1

        if use_sliding_window_kv_pool:
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        global global_override_indptr_cpu
        locally_override = False
        if seq_lens_cpu is not None and global_override_indptr_cpu is None:
            locally_override = True
            global_override_indptr_cpu = torch.empty_like(kv_indptr, device="cpu")
            global_override_indptr_cpu[0] = 0
            global_override_indptr_cpu[1 : bs + 1] = torch.cumsum(seq_lens_cpu, dim=0)

        # Check if this specific wrapper's begin_forward has been replaced with fast_decode_plan
        # by checking if it's a partial function with fast_decode_plan as the func
        wrapper_uses_fast_decode_plan = (
            hasattr(wrapper.begin_forward, "func")
            and wrapper.begin_forward.func == fast_decode_plan
        )

        if wrapper_uses_fast_decode_plan:
            # When begin_forward is replaced with fast_decode_plan, pass global_override_indptr_cpu
            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
                data_type=self.data_type,
                q_data_type=self.q_data_type,
                non_blocking=True,
                fixed_split_size=fixed_split_size,
                disable_split_kv=(
                    disable_split_kv if disable_split_kv is not None else False
                ),
                global_override_indptr_cpu=global_override_indptr_cpu,
            )
        else:
            # When using original begin_forward, don't pass global_override_indptr_cpu
            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
                data_type=self.data_type,
                q_data_type=self.q_data_type,
                non_blocking=True,
                fixed_split_size=fixed_split_size,
                disable_split_kv=(
                    disable_split_kv if disable_split_kv is not None else False
                ),
            )

        if locally_override:
            global_override_indptr_cpu = None

class FlashInferIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

        # Dispatch the update function
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper
        
        # fixme: use config to choose
        if True:
            self.update = self.update_variable_block_wrapper

        self.mock_kv_manager = mock_kv_manager
        self.layer_ids = list(range(model_runner.model_config.num_hidden_layers))
        self.kv_size = self.num_kv_heads * self.head_dim
        self.recompute_ratio_in_layer = attn_backend.recompute_ratio_in_layer
        self.max_recompute_layer_id = torch.argmax(torch.tensor(self.recompute_ratio_in_layer)).item()
        if os.getenv("SGLANG_VISUAL_KV_MODE", "legacy") != "legacy":
            from pathlib import Path

            from sglang.srt.mem_cache.scene_graph_memory.optimized_prefill import validate_server

            validate_server(model_runner)
            self.visual_kv_model_path = str(Path(model_runner.server_args.model_path).resolve())

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_variable_block_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        forward_batch,
    ):
        if os.getenv("SGLANG_VISUAL_KV_MODE", "legacy") != "legacy":
            from sglang.srt.mem_cache.scene_graph_memory.optimized_prefill import start_optimized_prefill

            if start_optimized_prefill(self, forward_batch, prefill_wrappers, use_ragged):
                return

        # here consider different layer may have different sparse pattern, this can be more concise if sparse pattern is all the same
        block_compute_list = [[] for i in self.layer_ids]
        block_compute_list_valid_cumsum = [[] for i in self.layer_ids]
        block_compute_list_total_cumsum = [[] for i in self.layer_ids]
        compute_num_list = [[] for i in self.layer_ids]
        actual_extend_seq_len = []
        
        compute_mask = {}
        recompute_info = {}
        write_info = {}

        graph_capture = None
        capture_dir = os.getenv("SGLANG_SCENE_GRAPH_CAPTURE_DIR")
        if capture_dir and forward_batch.forward_mode.is_extend():
            from sglang.srt.mem_cache.scene_graph_memory.capture import PrefillCapture

            graph_capture = PrefillCapture(capture_dir, forward_batch, self.layer_ids)
            forward_batch.scene_graph_capture = graph_capture
        bypass_image_cache = graph_capture is not None and graph_capture.bypass
        decoder_cache_disabled = os.getenv("SGLANG_VISUAL_KV_DISABLE_DECODER_CACHE") == "1"
        bypass_image_cache = bypass_image_cache or decoder_cache_disabled
        
        if forward_batch.forward_mode.is_extend():
            if not decoder_cache_disabled:
                compute_mask_list = [torch.ones(forward_batch.input_ids.shape[0], dtype=torch.bool, device=forward_batch.input_ids.device) for i in self.layer_ids]
                retrived_k_list = [torch.zeros((forward_batch.input_ids.shape[0], self.kv_size), dtype=self.data_type, device=forward_batch.input_ids.device) for i in self.layer_ids]
                retrived_v_list = [torch.zeros((forward_batch.input_ids.shape[0], self.kv_size), dtype=self.data_type, device=forward_batch.input_ids.device) for i in self.layer_ids]

            for mm_inputs, prefix_len, start_loc, seq_len, orig_seq_len in zip(forward_batch.mm_inputs, 
                                                                               forward_batch.extend_prefix_lens_cpu,
                                                                               forward_batch.extend_start_loc.tolist(),
                                                                               forward_batch.extend_seq_lens.tolist(),
                                                                               forward_batch.orig_seq_lens.tolist(),
                                                                               ):
                last_idx = 0 #for each req, it should start from 0, fix me to start_loc
                #last_idx = start_loc
                current_actual_seq_len = 0
                current_actual_skip_seq_len = 0
                if mm_inputs is None:
                    continue
                for item in mm_inputs.mm_items:
                    if bypass_image_cache:
                        # The entire image span stays in the ordinary dense extend block.
                        # Do not read or write the fork's shared image cache for this request.
                        continue
                    for cur_hash, (start_idx, end_idx) in zip(item.split_hash, item.offsets):
                        start_idx -= prefix_len
                        end_idx -= prefix_len

                        if start_idx < 0:
                            continue # adapt to radix cache, should not update kvcache

                        if end_idx > seq_len - 1:
                            continue

                        reuse_happened = False
                        for layer_id in self.layer_ids:
                            recompute_ratio = self.recompute_ratio_in_layer[layer_id]
                            total_token_num =  end_idx - start_idx + 1
                            recompute_token_num = max(1, int(total_token_num*recompute_ratio))
                            reuse_start = start_idx + recompute_token_num
                            reuse_end = end_idx + 1

                            uid_k = "_".join([str(cur_hash), str(layer_id), "k"])
                            uid_v = "_".join([str(cur_hash), str(layer_id), "v"])
                            if uid_k in self.mock_kv_manager.uuid_to_pace and uid_v in self.mock_kv_manager.uuid_to_pace:
                                # this should update part of retrived_k/v,
                                # here can be imporve, because former 30% will be recompute later
                                # we can only write and read the latter 70%
                                part_k = retrived_k_list[layer_id][start_loc + start_idx: start_loc + reuse_end]
                                part_v = retrived_v_list[layer_id][start_loc + start_idx: start_loc + reuse_end]
                                torch.cuda.synchronize() #this is essential to initialize tensor in cuda to avoid get wrong data from pace
                                assert part_k.shape[0] == reuse_end - start_idx
                                if os.getenv("SGLANG_VISUAL_KV_PROFILE"):
                                    from sglang.srt.mem_cache.scene_graph_memory.optimized_prefill import profile_range

                                    with profile_range("scene_kv.sync_h2d"):
                                        self.mock_kv_manager.get_kv(part_k, uid_k, non_blocking=False)
                                        self.mock_kv_manager.get_kv(part_v, uid_v, non_blocking=False)
                                else:
                                    self.mock_kv_manager.get_kv(part_k, uid_k, non_blocking=False)
                                    self.mock_kv_manager.get_kv(part_v, uid_v, non_blocking=False)
                                
                                # fixme: here assume all req in a batch either all cached, or all no cached
                                reuse_happened=True
                                
                                if start_idx != last_idx:
                                    block_compute_list[layer_id].extend([1, 0]) #text + image_recompute, image_reuse
                                    compute_num_list[layer_id].extend([reuse_start - last_idx, reuse_end - reuse_start])
                                    if layer_id == self.layer_ids[0]:
                                        current_actual_seq_len += reuse_start - last_idx
                                        current_actual_skip_seq_len += reuse_end - reuse_start
                                else:
                                    block_compute_list[layer_id].extend([1, 0]) #image_recompute, image_reuse
                                    compute_num_list[layer_id].extend([recompute_token_num, reuse_end - reuse_start])
                                    if layer_id == self.layer_ids[0]:
                                        current_actual_seq_len += recompute_token_num
                                        current_actual_skip_seq_len += reuse_end - reuse_start

                                # fixme: here assume all req in a batch either all cached, or all no cached
                                compute_mask_list[layer_id][start_loc + reuse_start: start_loc + reuse_end] = False
                                if layer_id not in recompute_info:
                                    recompute_info[layer_id] = [retrived_k_list[layer_id], retrived_v_list[layer_id]]
                                    
                                if layer_id not in compute_mask:
                                    compute_mask[layer_id] = compute_mask_list[layer_id]
                            else:
                                # when somewhat image is reused, q tensor will be compressed in future,
                                # which mean we need to revise write info here
                                real_start_idx = start_idx - current_actual_skip_seq_len
                                real_end_idx = end_idx - current_actual_skip_seq_len

                                if layer_id not in write_info:  
                                    write_info[layer_id] = [[real_start_idx, real_end_idx, uid_k, uid_v]]
                                else:
                                    write_info[layer_id].append([real_start_idx, real_end_idx, uid_k, uid_v])

                                # fixme: even kvcache not in pace, we can skip attn(img, k, v)

                        if reuse_happened:
                            last_idx = reuse_end
                for layer_id in self.layer_ids:
                    if last_idx != seq_len:
                        block_compute_list[layer_id].append(1)
                        compute_num_list[layer_id].append(seq_len-last_idx)
                        if layer_id == self.layer_ids[0]:
                            current_actual_seq_len += seq_len-last_idx
                    block_compute_list_valid_cumsum[layer_id].append(sum(block_compute_list[layer_id]))
                    block_compute_list_total_cumsum[layer_id].append(len(block_compute_list[layer_id]))

                # when one req in a batch ends, update actual_extend_seq_len, here just use layer[-1] info
                if current_actual_seq_len != 0:
                    actual_extend_seq_len.append(current_actual_seq_len)
                else:
                    actual_extend_seq_len.append(seq_len)
        
        if actual_extend_seq_len != [] and block_compute_list[0] != [1] and 0 in compute_mask:
            assert compute_mask[0].sum() == sum(actual_extend_seq_len)

        forward_batch.compute_mask = compute_mask
        forward_batch.recompute_info = recompute_info
        forward_batch.write_info = write_info
        if graph_capture is not None:
            graph_capture.observe_plan(compute_mask, recompute_info, write_info)
        if actual_extend_seq_len != []:
            forward_batch.actual_extend_seq_len = torch.tensor(actual_extend_seq_len, dtype=torch.int32).cuda()
        
        #fixme: here assume block_compute_list and compute_num_list are all the same for every layer
        processed_ratio = {}
        for layer_id in self.layer_ids:
            recompute_ratio = self.recompute_ratio_in_layer[layer_id]
            if recompute_ratio in processed_ratio:
                continue
            else:
                processed_ratio[recompute_ratio] = True
            cur_block_compute_list = block_compute_list[layer_id]
            cur_compute_num_list = compute_num_list[layer_id]
            if cur_block_compute_list == []: # all req do not have image modality
                cur_block_compute_list = [1]
                cur_compute_num_list = [seq_lens_sum]

            # fixme: here should consider conbine successive 0,0 or 1,1? e.g. [1,1,1,1,0]->[1,0]
            # but currently 0,0 situation should not happend?

            # e.g. block_compute_list=[1,1,0,1], compute_num_list = [2,3,7,2]
            # final mask map should be 3*4, [[1,0,0,0],[1,1,0,0],[1,1,1,1]]
            # in this cause, block_row_sz should be [2,3,2]; block_col_sz should be [2,3,7,2]

            block_num = len(cur_block_compute_list)
            row_mask = torch.tensor(cur_block_compute_list, dtype=torch.bool)
            block_mask_map = torch.tril(torch.ones(block_num, block_num, dtype=torch.bool), diagonal=0)  # [n, n]
            block_mask_map = block_mask_map[row_mask].to("cuda:0")# fixme: device

            # mask out cross attn between different reqs in a batch
            for si, ei in zip(block_compute_list_valid_cumsum[layer_id][:-1], block_compute_list_total_cumsum[layer_id][:-1]):
                block_mask_map[si:, :ei] = False

            block_mask_map = block_mask_map.repeat(self.num_kv_heads, 1, 1)
            block_row_sz = (torch.tensor(cur_compute_num_list, dtype=torch.int32)).to("cuda:0")# fixme: device
            block_row_sz = block_row_sz[row_mask]
            block_row_sz = block_row_sz.repeat(self.num_kv_heads,1)
            block_col_sz = (torch.tensor([cur_compute_num_list], dtype=torch.int32)).to("cuda:0")# fixme: device
            block_col_sz = block_col_sz.repeat(self.num_kv_heads,1)

            assert not use_ragged
            prefill_wrappers[recompute_ratio].plan(
                block_mask_map,
                block_row_sz,
                block_col_sz,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                causal=True,
                # fixme: some kwargs maybe need to pass in future
                non_blocking=True,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )

    
    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
    ):
        if use_ragged:
            # TODO: remove this device sync, we can use forward_batch.extend_prefix_lens_cpu
            # and forward_batch.extend_seq_lens_cpu
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self.call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrappers[0],
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            None,
            self.kv_indptr[0],
            self.qo_indptr[0],
            use_ragged,
            spec_info,
            fixed_split_size=fixed_split_size,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # window attention use paged only
                paged_kernel_lens = torch.minimum(
                    seq_lens,
                    torch.tensor(self.sliding_window_size) + seq_lens - prefix_lens,
                )
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()
            else:
                # full attention
                paged_kernel_lens = seq_lens
                paged_kernel_lens_sum = seq_lens_sum

            kv_start_idx = seq_lens - paged_kernel_lens
            use_sliding_window_kv_pool = wrapper_id == 0 and isinstance(
                self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
            )

            self.call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                paged_kernel_lens_sum = seq_lens_sum
            else:
                # cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()

            self.call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
            )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchPrefillWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[SpecInput],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
    ):
        bs = len(seq_lens)
        if spec_info is None:
            assert len(seq_lens) == len(req_pool_indices)
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
        else:
            assert isinstance(spec_info, SpecInput)
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    self.req_to_token,
                )
            )

        # extend part
        if use_ragged:
            wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

        if use_sliding_window_kv_pool:
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        # cached part
        wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            custom_mask=custom_mask,
            non_blocking=True,
            fixed_split_size=fixed_split_size,
        )


class FlashInferMultiStepDraftBackend:
    """
    Wrap multiple flashinfer attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices

        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        self.page_size = model_runner.page_size

        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.attn_backends: List[FlashInferAttnBackend] = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                FlashInferAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    kv_last_page_len_buf=self.kv_last_page_len,
                )
            )

        self.max_context_len = self.attn_backends[0].max_context_len

        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn: Callable,
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
        )

        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        # Copy the kv_indptr once to avoid multiple device-to-host copies in flashinfer's plan.
        indptr_cpu_whole = self.kv_indptr[:, : bs + 1].cpu()
        global global_override_indptr_cpu

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            global_override_indptr_cpu = indptr_cpu_whole[i]
            call_fn(i, forward_batch)

        global_override_indptr_cpu = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices = torch.empty(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int32,
            device="cuda",
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_bs * self.max_context_len),
            dtype=torch.int32,
            device="cuda",
        )

        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)


def should_use_tensor_core(
    kv_cache_dtype: torch.dtype,
    num_attention_heads: int,
    num_kv_heads: int,
) -> bool:
    """
    Determine whether to use tensor cores for attention computation.

    Args:
        kv_cache_dtype: Data type of the KV cache
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key/value heads

    Returns:
        bool: Whether to use tensor cores
    """
    # Try to use environment variable first
    env_override = os.environ.get("SGLANG_FLASHINFER_USE_TENSOR_CORE")
    if env_override is not None:
        return env_override.lower() == "true"

    # Try to use _grouped_size_compiled_for_decode_kernels if available
    # This is for flashinfer <=0.1.6. Otherwise, there is an accuracy bug
    try:
        from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

        if not _grouped_size_compiled_for_decode_kernels(
            num_attention_heads,
            num_kv_heads,
        ):
            return True
        else:
            return False
    except (ImportError, AttributeError):
        pass

    # Calculate GQA group size
    gqa_group_size = num_attention_heads // num_kv_heads

    # For Flashinfer, a GQA group size of at least 4 is needed to efficiently
    # use Tensor Cores, as it fuses the head group with the token dimension in MMA.
    if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    elif kv_cache_dtype in (torch.float16, torch.half, torch.bfloat16):
        return gqa_group_size >= 4
    else:
        return False
