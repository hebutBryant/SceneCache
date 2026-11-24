"""Causal GQA over compact K/V rows on BF16 NVIDIA GPUs.

K and Q must already use their model's attention representation (including
any QK normalization and RoPE). Logical positions apply only the causal mask;
they do not change the coordinates or contents of Q/K/V.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["Q_LEN", "K_LEN", "SPLIT_LEN"])
def _compact_gqa_forward(
    Q, K, V, Q_POS, K_POS, OUT, PART_MAX, PART_SUM,
    Q_LEN, K_LEN, Q_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr, STRIDE_Q_T: tl.constexpr, STRIDE_Q_H: tl.constexpr,
    STRIDE_K_T: tl.constexpr, STRIDE_K_H: tl.constexpr,
    STRIDE_V_T: tl.constexpr, STRIDE_V_H: tl.constexpr,
    STRIDE_Q_POS: tl.constexpr, STRIDE_K_POS: tl.constexpr,
    SCALE_LOG2: tl.constexpr, EXPLICIT_POS: tl.constexpr, CAUSAL: tl.constexpr,
    KEY_POSITIONS_SORTED: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    SPLIT_K: tl.constexpr, SPLIT_LEN,
):
    group_row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    query_row = group_row // GROUP_SIZE
    query_head = kv_head * GROUP_SIZE + group_row % GROUP_SIZE
    query_valid = query_row < Q_LEN
    d = tl.arange(0, HEAD_DIM)
    q = tl.load(
        Q + query_row[:, None] * STRIDE_Q_T + query_head[:, None] * STRIDE_Q_H + d[None, :],
        query_valid[:, None], other=0,
    )
    if CAUSAL:
        if EXPLICIT_POS:
            q_pos = tl.load(Q_POS + query_row * STRIDE_Q_POS, query_valid, other=0)
        else:
            q_pos = query_row + (K_LEN - Q_LEN)

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.full((BLOCK_M,), 0, tl.float32)
    acc = tl.full((BLOCK_M, HEAD_DIM), 0, tl.float32)
    first_key = split * SPLIT_LEN
    last_key = tl.minimum(first_key + SPLIT_LEN, K_LEN)
    if CAUSAL:
        if not EXPLICIT_POS:
            # A block may end inside a query's GQA head group.
            next_query = tl.minimum(tl.cdiv((tl.program_id(0) + 1) * BLOCK_M, GROUP_SIZE), Q_LEN)
            last_key = tl.minimum(last_key, next_query + K_LEN - Q_LEN)
        elif KEY_POSITIONS_SORTED:
            if q_pos.dtype == tl.int32:
                floor = -2147483648
            else:
                floor = -9223372036854775807
            largest_query = tl.max(tl.where(query_valid, q_pos, floor), axis=0)
            if largest_query < tl.load(K_POS + (last_key - 1) * STRIDE_K_POS):
                # Binary search tile starts. Retain the boundary tile and mask its
                # individual positions below; never visit fully future tiles.
                lo = 0
                hi = tl.cdiv(last_key - first_key, BLOCK_N)
                while lo < hi:
                    midpoint = (lo + hi) // 2
                    tile_key = first_key + midpoint * BLOCK_N
                    if tl.load(K_POS + tile_key * STRIDE_K_POS) <= largest_query:
                        lo = midpoint + 1
                    else:
                        hi = midpoint
                last_key = tl.minimum(last_key, first_key + lo * BLOCK_N)
    for start in range(first_key, last_key, BLOCK_N):
        key_row = start + tl.arange(0, BLOCK_N)
        key_valid = key_row < last_key
        k = tl.load(
            K + key_row[None, :] * STRIDE_K_T + kv_head * STRIDE_K_H + d[:, None],
            key_valid[None, :], other=0,
        )
        v = tl.load(
            V + key_row[:, None] * STRIDE_V_T + kv_head * STRIDE_V_H + d[None, :],
            key_valid[:, None], other=0,
        )
        visible = query_valid[:, None] & key_valid[None, :]
        if CAUSAL:
            if EXPLICIT_POS:
                k_pos = tl.load(K_POS + key_row * STRIDE_K_POS, key_valid, other=0)
            else:
                k_pos = key_row
            visible = visible & (k_pos[None, :] <= q_pos[:, None])

        logits = tl.dot(q, k) * SCALE_LOG2
        logits = tl.where(visible, logits, -float("inf"))
        next_max = tl.maximum(row_max, tl.max(logits, axis=1))
        # Mapping the empty maximum to zero leaves both exponentials zero.
        # This covers masked blocks and query rows without any visible keys.
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        previous_weight = tl.exp2(row_max - safe_max)
        current_weight = tl.exp2(logits - safe_max[:, None])
        row_sum = row_sum * previous_weight + tl.sum(current_weight, axis=1)
        acc = tl.dot(current_weight.to(tl.bfloat16), v, acc * previous_weight[:, None])
        row_max = next_max

    output_row = query_row * Q_HEADS + query_head
    if SPLIT_K:
        output_row = output_row + split * Q_LEN * Q_HEADS
        tl.store(PART_MAX + output_row, row_max, query_valid)
        tl.store(PART_SUM + output_row, row_sum, query_valid)
    else:
        acc = acc / tl.where(row_sum > 0, row_sum, 1.0)[:, None]
    tl.store(OUT + output_row[:, None] * HEAD_DIM + d[None, :], acc, query_valid[:, None])


@triton.jit
def _merge_decode_splits(
    PART_OUTPUT, PART_MAX, PART_SUM, OUT, Q_HEADS: tl.constexpr,
    SPLITS: tl.constexpr, BLOCK_SPLITS: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    head = tl.program_id(0)
    split = tl.arange(0, BLOCK_SPLITS)
    d = tl.arange(0, HEAD_DIM)
    row_index = split * Q_HEADS + head
    split_max = tl.load(PART_MAX + row_index, split < SPLITS, other=-float("inf"))
    split_sum = tl.load(PART_SUM + row_index, split < SPLITS, other=0.0)
    maximum = tl.max(split_max, axis=0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weight = tl.exp2(split_max - maximum)
    denom = tl.sum(split_sum * weight, axis=0)
    accumulators = tl.load(
        PART_OUTPUT + row_index[:, None] * HEAD_DIM + d[None, :],
        (split < SPLITS)[:, None], other=0.0,
    )
    result = tl.sum(accumulators * weight[:, None], axis=0) / tl.where(denom > 0, denom, 1.0)
    tl.store(OUT + head * HEAD_DIM + d, result)


def _check_positions(query, key, query_positions, key_positions, causal):
    if (query_positions is None) != (key_positions is None):
        raise ValueError("Provide both position vectors when using explicit logical positions")
    if query_positions is None:
        return
    if not causal:
        raise ValueError("Logical positions are used only for causal attention")
    for label, tensor, input_rows in (
        ("query_positions", query_positions, query), ("key_positions", key_positions, key)
    ):
        if (
            tensor.shape != (input_rows.shape[0],)
            or tensor.device != input_rows.device
            or tensor.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(f"{label} must have one int32 or int64 position per row on the input device")


def torch_attention_reference(
    query, key, value, *, query_positions=None, key_positions=None, scale=None, causal=True
):
    """Torch FP32 attention oracle, with stable all-masked-row handling."""
    _check_positions(query, key, query_positions, key_positions, causal)
    q_len, query_heads, head_dim = query.shape
    k_len, kv_heads, _ = key.shape
    if k_len == 0:
        return torch.zeros_like(query)
    scale = head_dim ** -0.5 if scale is None else scale
    queries = query.transpose(0, 1).float()
    keys = key.repeat_interleave(query_heads // kv_heads, dim=1).permute(1, 2, 0).float()
    values = value.repeat_interleave(query_heads // kv_heads, dim=1).transpose(0, 1).float()
    logits = torch.matmul(queries, keys) * scale
    if causal:
        if query_positions is None:
            query_positions = torch.arange(k_len - q_len, k_len, device=query.device)
            key_positions = torch.arange(k_len, device=key.device)
        visible = key_positions[None, :] <= query_positions[:, None]
        logits = logits.masked_fill(~visible, float("-inf"))
        logits = torch.where(visible.any(dim=-1)[None, :, None], logits, 0.0)
        probabilities = logits.softmax(dim=-1).masked_fill(~visible, 0.0)
    else:
        probabilities = logits.softmax(dim=-1)
    return torch.matmul(probabilities, values).transpose(0, 1).to(query.dtype).contiguous()


def triton_attention(
    query, key, value, *, query_positions=None, key_positions=None, scale=None, causal=True,
    key_positions_sorted=False,
):
    """Run inference attention for Q[Q,Hq,128], K/V[K,Hkv,128].

    Inputs are BF16 CUDA tensors with a contiguous last dimension. Consecutive
    groups of query heads share a K/V head. Explicit logical positions keep
    gaps in the causal mask when the K/V bank has been gathered or reordered.
    The implicit mask aligns query rows with the end of the compact K/V rows.
    A query with no visible keys returns zeros.
    Set key_positions_sorted only when explicit key positions are known to be
    nondecreasing. That allows skipping full tiles past each query block.
    """
    if (
        any(tensor.ndim != 3 or tensor.shape[-1] != 128 for tensor in (query, key, value))
        or key.shape != value.shape
    ):
        raise ValueError("Expected Q[Q,Hq,128] and identically shaped K/V[K,Hkv,128]")
    if (
        not query.is_cuda
        or any(tensor.device != query.device or tensor.dtype != torch.bfloat16 for tensor in (query, key, value))
        or any(tensor.stride(-1) != 1 for tensor in (query, key, value))
    ):
        raise ValueError("Inputs must be BF16 on one CUDA device with contiguous head dimensions")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (query, key, value)):
        raise ValueError("This Triton kernel supports inference only")
    q_len, q_heads, head_dim = query.shape
    k_len, kv_heads, _ = key.shape
    if not q_heads or not kv_heads or q_heads % kv_heads:
        raise ValueError("Query head count must be a positive multiple of the K/V head count")
    _check_positions(query, key, query_positions, key_positions, causal)
    scale = head_dim ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Attention scale must be finite and positive")
    if q_len == 0 or k_len == 0:
        return torch.zeros_like(query, memory_format=torch.contiguous_format)

    group_size = q_heads // kv_heads
    block_m, block_n = 16, 64
    split_len = triton.cdiv(k_len, block_n) * block_n
    # A short decode has too few query tiles to occupy an A6000. Split long K/V
    # banks into disjoint partitions and merge their online softmax statistics.
    if q_len == 1 and k_len >= 512:
        split_len = max(128, triton.cdiv(k_len, 16 * block_n) * block_n)
    splits = triton.cdiv(k_len, split_len)
    with torch.cuda.device(query.device):
        output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
        part_output, part_max, part_sum = output, output, output
        if splits > 1:
            part_output = torch.empty((splits, q_heads, head_dim), device=query.device, dtype=torch.float32)
            part_max = torch.empty((splits, q_heads), device=query.device, dtype=torch.float32)
            part_sum = torch.empty_like(part_max)
        _compact_gqa_forward[(triton.cdiv(q_len * group_size, block_m), kv_heads, splits)](
            query, key, value,
            query if query_positions is None else query_positions,
            key if key_positions is None else key_positions,
            part_output, part_max, part_sum,
            q_len, k_len, q_heads, group_size,
            query.stride(0), query.stride(1), key.stride(0), key.stride(1),
            value.stride(0), value.stride(1),
            1 if query_positions is None else query_positions.stride(0),
            1 if key_positions is None else key_positions.stride(0),
            scale * math.log2(math.e), query_positions is not None, causal, key_positions_sorted,
            block_m, block_n, head_dim, splits > 1, split_len, num_warps=4,
        )
        if splits > 1:
            _merge_decode_splits[(q_heads,)](
                part_output, part_max, part_sum, output, q_heads,
                splits, triton.next_power_of_2(splits), head_dim, num_warps=4,
            )
    return output
