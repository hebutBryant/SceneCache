"""Check Nsight overlap grouping and duplicate-interval accounting offline."""

from pathlib import Path
import sqlite3
import tempfile
import unittest

from examples.scene_graph_memory.analyze_cuda_trace import _partition_kernel_intervals, analyze


class TestCudaOverlap(unittest.TestCase):
    def test_kernel_phase_union_partitions_concurrent_time_once(self):
        only, including, concurrent = _partition_kernel_intervals({
            "qkv": [(100, 160), (130, 180)],
            "attention": [(160, 220)],
            "mlp": [(240, 260)],
        })
        self.assertEqual(only, {"qkv": 60, "attention": 40, "mlp": 20})
        self.assertEqual(including, {"qkv": 80, "attention": 60, "mlp": 20})
        self.assertEqual(concurrent, 20)

    def test_correlated_copies_intersect_only_distinct_streams_and_union_kernels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.sqlite"
            pid = 5 << 24
            competitor_pid = 6 << 24
            with sqlite3.connect(path) as database:
                database.executescript("""
                    CREATE TABLE StringIds(id, value);
                    INSERT INTO StringIds VALUES
                          (1, 'cudaMemcpyAsync_v3020'), (2, 'decoder_kernel'),
                          (3, 'scene_kv.h2d_enqueue.layer=0');
                    CREATE TABLE NVTX_EVENTS(start, end, text, textId, globalTid);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start, end, correlationId,
                                                             globalTid, nameId);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start, end, globalPid, deviceId,
                        contextId, streamId, correlationId, bytes, copyKind);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start, end, globalPid, deviceId,
                        contextId, streamId, correlationId, shortName);
                """)
                database.execute("INSERT INTO NVTX_EVENTS VALUES (50, 85, NULL, 3, ?)", (pid + 3,))
                database.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, 1)", [
                    (55, 56, 1, pid + 3), (80, 81, 2, pid + 3),
                    (60, 61, 1, competitor_pid + 2),
                ])
                database.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, 0, 1, 7, ?, 16, 1)", [
                        (100, 200, pid, 1), (210, 230, pid, 2), (100, 200, competitor_pid, 1),
                    ]
                )
                database.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, 0, ?, ?, 0, 2)", [
                        (95, 205, pid, 1, 7),  # same stream: never eligible
                        (95, 140, pid, 1, 9), (130, 165, pid, 1, 10), (180, 210, pid, 1, 10),
                        (95, 240, pid, 2, 9),  # other context: never eligible
                        (95, 240, competitor_pid, 1, 9),
                    ],
                )
            summary = analyze(path)
        self.assertEqual(summary["selection"]["h2d_events_in_export"], 3)
        self.assertEqual(summary["totals"]["h2d_copy_count"], 2)
        self.assertEqual(summary["totals"]["copy_duration_sum_ns"], 120)
        self.assertEqual(summary["totals"]["copy_overlap_sum_ns"], 85)
        self.assertEqual(summary["totals"]["copies_with_cross_stream_overlap"], 1)
        self.assertEqual(summary["examples"][0]["overlap_intervals_ns"], [(100, 165), (180, 200)])
        self.assertEqual(summary["copies"][1]["overlap_intervals_ns"], [])
        activities = summary["competing_gpu_activity"]["other_contexts_on_selected_devices"]
        self.assertEqual(sum(item["kernel_count"] for item in activities), 2)


if __name__ == "__main__":
    unittest.main()
