import unittest
import torch

from sglang.srt.mem_cache.scene_graph_memory.triton_attention import (
    torch_attention_reference,
    triton_attention,
)


def _inputs(q_len, k_len, q_heads=32, kv_heads=8):
    generator = torch.Generator(device="cuda").manual_seed(42)
    query = torch.randn(q_len, q_heads, 128, dtype=torch.bfloat16, device="cuda", generator=generator)
    key = torch.randn(k_len, kv_heads, 128, dtype=query.dtype, device=query.device, generator=generator)
    value = torch.randn(key.shape, dtype=key.dtype, device=key.device, generator=generator)
    return query, key, value


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    "CUDA BF16 is required for Triton GQA",
)
class CompactTritonAttentionTest(unittest.TestCase):
    def test_causal_tails(self):
        for q_len, k_len in [(1, 7), (31, 257), (67, 131), (128, 882), (1, 882)]:
            with self.subTest(q_len=q_len, k_len=k_len):
                q, k, v = _inputs(q_len, k_len)
                result = triton_attention(q, k, v)
                expected = torch_attention_reference(q, k, v)
                self.assertEqual(result.shape, q.shape)
                self.assertEqual(result.dtype, q.dtype)
                torch.testing.assert_close(result, expected, atol=0.012, rtol=0.012)

    def test_sparse_logical_positions_and_strides(self):
        # Heads and rows retain a padded stride from their source allocation.
        q, k, v = _inputs(29, 302, q_heads=12, kv_heads=5)
        q, k, v = q[::2], k[::2, :4], v[::2, 1:]
        source_positions = torch.arange(k.shape[0], device=k.device, dtype=torch.int64) * 4 + 25
        query_positions = torch.arange(q.shape[0], device=q.device, dtype=torch.int32) * 45
        # Unsorted keys exercise the logical mask even when physical ordering differs.
        args = {"query_positions": query_positions, "key_positions": source_positions.flip(0)}
        result = triton_attention(q, k, v, **args)
        reference = torch_attention_reference(q, k, v, **args)
        torch.testing.assert_close(result, reference, atol=0.012, rtol=0.012)
        self.assertEqual(torch.count_nonzero(result[0]).item(), 0)

    def test_region_prefill_mask_matches_compact_causal_order(self):
        # Real market probe topology: 4 prefix, 99 selected visual rows, 32
        # suffix. Queries cover the prefix, first 29 visual rows, and suffix.
        q, k, v = _inputs(65, 135)
        dtype = torch.int32
        device = q.device
        positions = torch.cat((
            torch.arange(4, device=device, dtype=dtype),
            22 + 7 * torch.arange(99, device=device, dtype=dtype),
            torch.arange(886, 918, device=device, dtype=dtype),
        ))
        active_queries = torch.cat((torch.arange(33, device=device), torch.arange(103, 135, device=device)))
        kwargs = {"query_positions": positions[active_queries], "key_positions": positions}
        out = triton_attention(q, k, v, key_positions_sorted=True, **kwargs)
        self.assertTrue(torch.equal(out, triton_attention(q, k, v, **kwargs)))
        reference = torch_attention_reference(q, k, v, **kwargs)
        torch.testing.assert_close(out, reference, atol=0.012, rtol=0.012)

        poisoned = v.clone()
        poisoned[33:] = 256
        changed = triton_attention(q, k, poisoned, key_positions_sorted=True, **kwargs)
        self.assertTrue(torch.equal(out[:33], changed[:33]))
        self.assertFalse(torch.equal(out[33:], changed[33:]))

    def test_split_decode_with_partially_or_fully_masked_partitions(self):
        q, k, v = _inputs(1, 647)
        key_positions = torch.arange(647, device=k.device, dtype=torch.int32) * 2 + 3
        for position in (0, 115, 685, 1535):
            with self.subTest(position=position):
                args = {
                    "query_positions": torch.tensor([position], device=q.device, dtype=torch.int32),
                    "key_positions": key_positions,
                }
                result = triton_attention(q, k, v, key_positions_sorted=True, **args)
                reference = torch_attention_reference(q, k, v, **args)
                torch.testing.assert_close(result, reference, atol=0.012, rtol=0.012)
                if position == 0:
                    self.assertEqual(result.count_nonzero().item(), 0)

    def test_noncausal_and_empty_keys(self):
        q, k, v = _inputs(19, 11)
        torch.testing.assert_close(
            triton_attention(q, k, v, causal=False, scale=0.2),
            torch_attention_reference(q, k, v, causal=False, scale=0.2),
            atol=0.016,
            rtol=0.012,
        )
        self.assertTrue(torch.equal(triton_attention(q, k[:0], v[:0]), torch.zeros_like(q)))
