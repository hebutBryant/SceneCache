"""Bounded CPU-to-CUDA transfer for fixed visual K/V rows.

A slot owns a pinned host and CUDA buffer for K and V. The transfer stream
waits for the prior consumer before overwriting a CUDA slot. The host only
rewrites pinned memory after its prior DMA completes.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from operator import index as integer_index

import torch


KVPair = tuple[torch.Tensor, torch.Tensor]


@dataclass
class _Slot:
    host: KVPair
    device: KVPair
    ready: torch.cuda.Event
    compute_done: torch.cuda.Event
    pending_transfer: bool = False
    pending_compute: bool = False
    layer: int | None = None
    consumer_stream: torch.cuda.Stream | None = None


@dataclass(frozen=True)
class _Timing:
    layer: int
    start: torch.cuda.Event
    end: torch.cuda.Event


def _copy_shapes(
    source_layers: Mapping[int, KVPair],
    source_indices: Iterable[int] | torch.Tensor | None,
) -> tuple[tuple[torch.Size, torch.Size], tuple[torch.dtype, torch.dtype], torch.Tensor | None]:
    first = next(iter(source_layers.values()))
    if len(first) != 2 or any(tensor.ndim < 2 for tensor in first):
        raise ValueError("Expected token-major K/V tensors with at least two axes")
    shapes = tuple(tensor.shape for tensor in first)
    dtypes = tuple(tensor.dtype for tensor in first)
    rows = shapes[0][0]
    if shapes[1][0] != rows:
        raise ValueError("K and V must have the same number of source rows")

    for layer, pair in source_layers.items():
        if (
            len(pair) != 2
            or tuple(tensor.shape for tensor in pair) != shapes
            or tuple(tensor.dtype for tensor in pair) != dtypes
        ):
            raise ValueError(f"Layer {layer} does not match the source K/V shape or dtype")
        if any(
            tensor.device.type != "cpu"
            or tensor.requires_grad
            or tensor.layout != torch.strided
            for tensor in pair
        ):
            raise ValueError(f"Layer {layer} must provide detached strided CPU K/V tensors")

    indices = None
    if source_indices is not None:
        if isinstance(source_indices, torch.Tensor):
            if (
                source_indices.device.type != "cpu"
                or source_indices.ndim != 1
                or source_indices.dtype != torch.long
            ):
                raise ValueError("Tensor source_indices must be a 1D CPU int64 tensor")
            indices = source_indices.clone().contiguous()
        else:
            values = [integer_index(value) for value in source_indices]
            indices = torch.tensor(values, dtype=torch.long)
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= rows):
            raise IndexError("source_indices contains an out-of-range visual row")
        shapes = tuple(torch.Size((len(indices), *shape[1:])) for shape in shapes)
    return shapes, dtypes, indices


class KVLayerTransfer:
    """Stage visual K/V by layer through a fixed-size pinned host/device ring.

    The source maps layer IDs to token-major CPU tensors. If source_indices is
    provided, stage just those rows in the supplied order. Otherwise the pair
    already represents the desired compact bank.

    Call schedule for upcoming layers. Consume a pair on the CUDA stream that
    will read it, enqueue all operations that use the returned views, then
    call release on that same stream. The views belong to the ring and must
    not be used after release. This class is request-local and single-owner.
    """

    def __init__(
        self,
        source_layers: Mapping[int, KVPair],
        device: torch.device | str,
        *,
        source_indices: Iterable[int] | torch.Tensor | None = None,
        prefetch_depth: int = 2,
        max_pinned_bytes: int | None = None,
        max_gpu_bytes: int | None = None,
        telemetry: bool = False,
    ) -> None:
        if not source_layers:
            raise ValueError("The transfer source must contain at least one layer")
        if isinstance(prefetch_depth, bool) or integer_index(prefetch_depth) < 1:
            raise ValueError("prefetch_depth must be a positive integer")
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("The pinned transfer destination must be a CUDA device")

        self._source = dict(source_layers)
        self._shapes, self._dtypes, self._indices = _copy_shapes(
            self._source, source_indices
        )
        self.prefetch_depth = int(prefetch_depth)
        self.slot_count = min(self.prefetch_depth, len(self._source))
        first = next(iter(self._source.values()))
        self.bytes_per_layer = sum(
            math.prod(shape) * tensor.element_size()
            for shape, tensor in zip(self._shapes, first)
        )
        self.pinned_bytes = self.device_buffer_bytes = self.slot_count * self.bytes_per_layer
        for name, budget, required in (
            ("max_pinned_bytes", max_pinned_bytes, self.pinned_bytes),
            ("max_gpu_bytes", max_gpu_bytes, self.device_buffer_bytes),
        ):
            if budget is not None:
                budget = integer_index(budget)
                if budget < 0:
                    raise ValueError(f"{name} must be nonnegative")
                if required > budget:
                    raise MemoryError(
                        f"Transfer ring requires {required} bytes, above {name}={budget}. "
                        "Reduce prefetch_depth or use a smaller compact source."
                    )

        self.device = torch.device(
            "cuda", target.index if target.index is not None else torch.cuda.current_device()
        )
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._slots: list[_Slot] = []
        with torch.cuda.stream(self._copy_stream):
            for _ in range(self.slot_count):
                self._slots.append(
                    _Slot(
                        host=tuple(
                            torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
                            for shape, dtype in zip(self._shapes, self._dtypes)
                        ),
                        device=tuple(
                            torch.empty(shape, dtype=dtype, device=self.device)
                            for shape, dtype in zip(self._shapes, self._dtypes)
                        ),
                        ready=torch.cuda.Event(),
                        compute_done=torch.cuda.Event(),
                    )
                )

        self._scheduled: dict[int, _Slot] = {}
        self._active: dict[int, _Slot] = {}
        self._copies = 0
        self._consumes = 0
        self._demand_misses = 0
        self._max_slots_occupied = 0
        self._closed = False
        self._telemetry = telemetry
        self._staging_ms = 0.0
        self._host_reuse_wait_ms = 0.0
        self._host_wait_count = 0
        self._close_wait_count = 0
        self._close_wait_ms = 0.0
        self._copy_timings: list[_Timing] = []
        self._consumer_timings: list[_Timing] = []

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("The K/V transfer ring is closed")

    def _profile_range(self, name: str):
        return torch.cuda.nvtx.range(name) if self._telemetry else nullcontext()

    def _free_slot(self) -> _Slot | None:
        free_slots = [slot for slot in self._slots if slot.layer is None]
        if not free_slots:
            return None
        for slot in free_slots:
            if not slot.pending_transfer or slot.ready.query():
                return slot
        return free_slots[0]

    def schedule(self, layer: int) -> bool:
        """Stage a layer now, returning False when all slots are owned."""
        self._require_open()
        source = self._source[layer]
        if layer in self._scheduled or layer in self._active:
            return True
        slot = self._free_slot()
        if slot is None:
            return False

        if slot.pending_transfer:
            # The prior nonblocking copy still owns this host memory until its
            # ready event. This wait is event-local and does not drain the GPU.
            if not slot.ready.query():
                started = time.perf_counter() if self._telemetry else 0
                self._host_wait_count += 1
                with self._profile_range(f"scene_kv.host_slot_wait.layer={layer}"):
                    slot.ready.synchronize()
                if self._telemetry:
                    self._host_reuse_wait_ms += (time.perf_counter() - started) * 1000
            slot.pending_transfer = False
        started = time.perf_counter() if self._telemetry else 0
        with self._profile_range(f"scene_kv.cpu_stage.layer={layer}"):
            for destination, tensor in zip(slot.host, source):
                if self._indices is None:
                    destination.copy_(tensor)
                else:
                    torch.index_select(tensor, 0, self._indices, out=destination)
        if self._telemetry:
            self._staging_ms += (time.perf_counter() - started) * 1000

        with torch.cuda.stream(self._copy_stream):
            if slot.pending_compute:
                self._copy_stream.wait_event(slot.compute_done)
                slot.pending_compute = False
            if self._telemetry:
                copy_start = torch.cuda.Event(enable_timing=True)
                copy_end = torch.cuda.Event(enable_timing=True)
                copy_start.record(self._copy_stream)
            with self._profile_range(f"scene_kv.h2d_enqueue.layer={layer}"):
                for destination, tensor in zip(slot.device, slot.host):
                    destination.copy_(tensor, non_blocking=True)
            if self._telemetry:
                copy_end.record(self._copy_stream)
                self._copy_timings.append(_Timing(layer, copy_start, copy_end))
            slot.ready.record(self._copy_stream)

        slot.pending_transfer = True
        slot.layer = layer
        self._scheduled[layer] = slot
        self._copies += 1
        self._max_slots_occupied = max(
            self._max_slots_occupied, len(self._scheduled) + len(self._active)
        )
        return True

    def consume(self, layer: int) -> KVPair:
        """Await the transfer on the caller's compute stream and return ring views."""
        self._require_open()
        if layer in self._active:
            raise RuntimeError(f"Layer {layer} already has a consumer")
        if layer not in self._scheduled:
            if not self.schedule(layer):
                raise RuntimeError(
                    f"No transfer slot for layer {layer}. Release a consumed layer "
                    "or avoid scheduling more future layers than the ring can hold."
                )
            self._demand_misses += 1

        slot = self._scheduled[layer]
        compute_stream = torch.cuda.current_stream(self.device)
        if self._telemetry:
            wait_start = torch.cuda.Event(enable_timing=True)
            wait_end = torch.cuda.Event(enable_timing=True)
            wait_start.record(compute_stream)
        compute_stream.wait_event(slot.ready)
        if self._telemetry:
            wait_end.record(compute_stream)
            self._consumer_timings.append(_Timing(layer, wait_start, wait_end))

        slot.consumer_stream = compute_stream
        self._active[layer] = self._scheduled.pop(layer)
        self._consumes += 1
        return slot.device

    def release(self, layer: int) -> None:
        """Record the final queued CUDA read before making its slot reusable."""
        self._require_open()
        if layer not in self._active:
            raise RuntimeError(f"Layer {layer} has no active consumer")
        slot = self._active[layer]
        compute_stream = torch.cuda.current_stream(self.device)
        if compute_stream != slot.consumer_stream:
            raise RuntimeError("Release must run on the stream that consumed the layer")
        slot.compute_done.record(compute_stream)
        slot.pending_compute = True
        slot.consumer_stream = None
        slot.layer = None
        del self._active[layer]

    def close(self) -> None:
        """Drain this ring's copies and reads before giving up its buffer references."""
        if self._closed:
            return
        try:
            started = time.perf_counter() if self._telemetry else 0
            with self._profile_range("scene_kv.close_wait"):
                for slot in self._active.values():
                    slot.compute_done.record(slot.consumer_stream)
                    slot.pending_compute = True
                self._close_wait_count += 1
                self._copy_stream.synchronize()
                for slot in self._slots:
                    if slot.pending_compute:
                        self._close_wait_count += 1
                        slot.compute_done.synchronize()
            if self._telemetry:
                self._close_wait_ms = (time.perf_counter() - started) * 1000
        finally:
            self._closed = True
            self._scheduled.clear()
            self._active.clear()
            self._source.clear()
            self._slots.clear()

    def __enter__(self) -> KVLayerTransfer:
        self._require_open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @staticmethod
    def _completed_timings(timings: list[_Timing]) -> list[dict]:
        return [
            {"layer": timing.layer, "ms": timing.start.elapsed_time(timing.end)}
            for timing in timings
            if timing.end.query()
        ]

    def metrics(self) -> dict:
        """Report logical buffer/transfer bytes and completed timing events.

        GPU milliseconds measure stream-local intervals. Individual phases may
        overlap and should not be summed as a request latency. Call after
        close for complete timing samples. This method itself never waits.
        """
        measurements = {
            "prefetch_depth": self.prefetch_depth,
            "slot_count": self.slot_count,
            "key_shape": list(self._shapes[0]),
            "value_shape": list(self._shapes[1]),
            "key_dtype": str(self._dtypes[0]),
            "value_dtype": str(self._dtypes[1]),
            "bytes_per_layer": self.bytes_per_layer,
            "pinned_bytes": self.pinned_bytes,
            "device_buffer_bytes": self.device_buffer_bytes,
            "h2d_bytes": self._copies * self.bytes_per_layer,
            "scheduled_layers": self._copies,
            "consumed_layers": self._consumes,
            "prefetch_hits": self._consumes - self._demand_misses,
            "demand_misses": self._demand_misses,
            "max_slots_occupied": self._max_slots_occupied,
            "host_wait_count": self._host_wait_count,
            "close_wait_count": self._close_wait_count,
        }
        if self._telemetry:
            copies = self._completed_timings(self._copy_timings)
            waits = self._completed_timings(self._consumer_timings)
            measurements.update(
                staging_cpu_ms=self._staging_ms,
                host_buffer_reuse_wait_ms=self._host_reuse_wait_ms,
                close_wait_cpu_ms=self._close_wait_ms,
                h2d_event_ms=sum(sample["ms"] for sample in copies),
                consumer_wait_event_ms=sum(sample["ms"] for sample in waits),
                h2d_event_samples=copies,
                consumer_wait_event_samples=waits,
            )
        return measurements
