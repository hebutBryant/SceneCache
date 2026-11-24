"""Focused checks for reusable pinned-buffer and CUDA stream ownership."""

import unittest

import torch

from sglang.srt.mem_cache.scene_graph_memory.kv_transfer import KVLayerTransfer


def sample_layers():
    # Noncontiguous CPU source views test that staging respects strides.
    return {
        layer: (
            (torch.arange(96).reshape(12, 8) + 400 * layer).to(torch.bfloat16)[::2],
            (torch.arange(72).reshape(12, 6) - 400 * layer).to(torch.float32)[::2],
        )
        for layer in range(4)
    }


class TestKVLayerValidation(unittest.TestCase):
    def test_budget_checked_before_cuda_allocation(self):
        source = sample_layers()
        # Each compact layer is 3 * (8 * 2 + 6 * 4) = 120 bytes.
        with self.assertRaisesRegex(MemoryError, "max_pinned_bytes=239"):
            KVLayerTransfer(
                source, "cuda:0", source_indices=[4, 1, 0], max_pinned_bytes=239
            )
        with self.assertRaisesRegex(MemoryError, "max_gpu_bytes=239"):
            KVLayerTransfer(
                source, "cuda:0", source_indices=[4, 1, 0], max_gpu_bytes=239
            )

    def test_rejects_invalid_sources_before_cuda_allocation(self):
        source = sample_layers()
        with self.assertRaisesRegex(ValueError, "positive integer"):
            KVLayerTransfer(source, "cuda:0", prefetch_depth=0)
        with self.assertRaisesRegex(ValueError, "at least one layer"):
            KVLayerTransfer({}, "cuda:0")
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            KVLayerTransfer(source, "cpu")
        with self.assertRaisesRegex(IndexError, "out-of-range"):
            KVLayerTransfer(source, "cuda:0", source_indices=[6])
        with self.assertRaisesRegex(ValueError, "1D CPU int64"):
            KVLayerTransfer(source, "cuda:0", source_indices=torch.tensor([1], dtype=torch.float32))
        source[2] = source[2][0][1:], source[2][1]
        with self.assertRaisesRegex(ValueError, "Layer 2"):
            KVLayerTransfer(source, "cuda:0")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for H2D ring")
class TestKVLayerTransfer(unittest.TestCase):
    def test_prefetched_rows_and_buffer_recycling(self):
        source = sample_layers()
        indices = [4, 1, 0]
        consumer = torch.cuda.Stream()
        outputs = {}
        with KVLayerTransfer(
            source, "cuda", source_indices=indices, prefetch_depth=2, telemetry=True,
            max_pinned_bytes=240, max_gpu_bytes=240,
        ) as transfer:
            self.assertTrue(transfer.schedule(0))
            self.assertTrue(transfer.schedule(1))
            self.assertTrue(transfer.schedule(1))
            self.assertFalse(transfer.schedule(2))
            with torch.cuda.stream(consumer):
                for layer in range(4):
                    key, value = transfer.consume(layer)
                    if layer == 0:
                        with self.assertRaisesRegex(RuntimeError, "already has a consumer"):
                            transfer.consume(0)
                        # Queue unfinished reads on the consumer when release
                        # makes the slot eligible for a later transfer.
                        torch.cuda._sleep(10_000_000)
                    outputs[layer] = key.clone(), value.clone()
                    transfer.release(layer)
                    if layer + 2 < len(source):
                        self.assertTrue(transfer.schedule(layer + 2))

        for layer, pair in outputs.items():
            for copied, original in zip(pair, source[layer]):
                self.assertTrue(torch.equal(copied.cpu(), original[indices]))
        metrics = transfer.metrics()
        self.assertEqual(metrics["key_shape"], [3, 8])
        self.assertEqual(metrics["value_shape"], [3, 6])
        self.assertEqual(metrics["pinned_bytes"], 240)
        self.assertEqual(metrics["device_buffer_bytes"], 240)
        self.assertEqual(metrics["h2d_bytes"], 480)
        self.assertEqual(metrics["demand_misses"], 0)
        self.assertEqual(metrics["prefetch_hits"], 4)
        self.assertEqual(metrics["max_slots_occupied"], 2)
        self.assertEqual(len(metrics["h2d_event_samples"]), 4)
        self.assertEqual(len(metrics["consumer_wait_event_samples"]), 4)
        self.assertGreaterEqual(metrics["h2d_event_ms"], 0)
        self.assertGreaterEqual(metrics["consumer_wait_event_ms"], 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            transfer.schedule(0)

    def test_demand_copy_and_stream_ownership(self):
        source = sample_layers()
        consumer = torch.cuda.Stream()
        transfer = KVLayerTransfer(source, "cuda", prefetch_depth=1)
        with torch.cuda.stream(consumer):
            staged = transfer.consume(1)
            copied = tuple(tensor.clone() for tensor in staged)
        with self.assertRaisesRegex(RuntimeError, "stream that consumed"):
            transfer.release(1)
        # Close must also drain an active consumer created on another stream.
        transfer.close()
        transfer.close()
        for got, want in zip(copied, source[1]):
            self.assertTrue(torch.equal(got.cpu(), want))
        self.assertEqual(transfer.metrics()["demand_misses"], 1)
        self.assertEqual(transfer.metrics()["prefetch_hits"], 0)
        self.assertEqual(transfer.metrics()["slot_count"], 1)
        self.assertEqual(transfer.metrics()["host_wait_count"], 0)
        self.assertEqual(transfer.metrics()["close_wait_count"], 2)

    def test_explicit_empty_selection(self):
        source = sample_layers()
        with KVLayerTransfer(
            source, "cuda", source_indices=[], prefetch_depth=4,
            max_pinned_bytes=0, max_gpu_bytes=0,
        ) as transfer:
            for layer in source:
                key, value = transfer.consume(layer)
                self.assertEqual(key.shape, (0, 8))
                self.assertEqual(value.shape, (0, 6))
                transfer.release(layer)
        self.assertEqual(transfer.metrics()["h2d_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
