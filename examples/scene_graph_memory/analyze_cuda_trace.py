#!/usr/bin/env python3
"""Measure visual-KV H2D and kernel overlap from an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable
from contextlib import closing
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3


_RUNTIME = "CUPTI_ACTIVITY_KIND_RUNTIME"
_KERNEL = "CUPTI_ACTIVITY_KIND_KERNEL"
_MEMCPY = "CUPTI_ACTIVITY_KIND_MEMCPY"
_NVTX = "NVTX_EVENTS"
_DEFAULT_PREFIXES = ("scene_kv.h2d_enqueue.layer=", "scene_kv.sync_h2d")
_PREFILL_SCOPE = "scenecache_post_warmup_prefill"
Interval = tuple[int, int]
Context = tuple[int | None, int, int]


@dataclass(frozen=True)
class Range:
    start_ns: int
    end_ns: int
    text: str
    global_tid: int | None


@dataclass(frozen=True)
class Api:
    start_ns: int
    end_ns: int
    correlation_id: int
    global_tid: int
    name: str


@dataclass(frozen=True)
class Activity:
    start_ns: int
    end_ns: int
    global_pid: int | None
    device_id: int
    context_id: int
    stream_id: int
    correlation_id: int

    @property
    def context(self) -> Context:
        return self.global_pid, self.device_id, self.context_id

    @property
    def interval(self) -> Interval:
        return self.start_ns, self.end_ns


@dataclass(frozen=True)
class Kernel(Activity):
    name: str


@dataclass(frozen=True)
class Copy(Activity):
    bytes: int
    source_memory_kind: str


def _union(intervals: Iterable[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            prior_start, prior_end = merged[-1]
            merged[-1] = prior_start, max(prior_end, end)
        else:
            merged.append((start, end))
    return merged


def _duration(intervals: Iterable[Interval]) -> int:
    return sum(end - start for start, end in intervals)


class _IntervalUnion:
    def __init__(self, intervals: Iterable[Interval]):
        self.intervals = _union(intervals)
        self._starts = [start for start, _ in self.intervals]
        self._ends = [end for _, end in self.intervals]

    def intersection(self, interval: Interval) -> list[Interval]:
        start, end = interval
        first = bisect_right(self._ends, start)
        stop = bisect_left(self._starts, end)
        return [
            (max(start, left), min(end, right)) for left, right in self.intervals[first:stop]
        ]


def _table_names(db: sqlite3.Connection) -> set[str]:
    return {
        name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _table_columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}


def _string_ids(db: sqlite3.Connection, tables: set[str]) -> dict[int, str]:
    if "StringIds" not in tables:
        return {}
    return dict(db.execute("SELECT id, value FROM StringIds"))


def _read_ranges(db, tables, strings) -> list[Range]:
    if _NVTX not in tables:
        return []
    columns = _table_columns(db, _NVTX)
    text_id = "textId" if "textId" in columns else "NULL"
    text = "text" if "text" in columns else "NULL"
    ranges = []
    for start, end, message, message_id, tid in db.execute(
        f"SELECT start, end, {text}, {text_id}, globalTid FROM NVTX_EVENTS "
        "WHERE end > start ORDER BY start"
    ):
        message = message or strings.get(message_id)
        if message:
            ranges.append(Range(start, end, message, tid))
    return ranges


def _read_runtime(db, tables, strings) -> dict[tuple[int, int], list[Api]]:
    apis = defaultdict(list)
    if _RUNTIME not in tables:
        return apis
    for start, end, corr, tid, name_id in db.execute(
        f"SELECT start, end, correlationId, globalTid, nameId FROM {_RUNTIME} "
        "WHERE correlationId IS NOT NULL AND globalTid IS NOT NULL AND end >= start"
    ):
        # The Nsight export encodes the local TID in the last 24 bits. The
        # remaining process bits are the same as the activity globalPid.
        apis[tid >> 24, corr].append(Api(start, end, corr, tid, strings.get(name_id, str(name_id))))
    return apis


def _read_kernels(db, tables, strings) -> list[Kernel]:
    table = _KERNEL if _KERNEL in tables else "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"
    if table not in tables:
        raise ValueError("Trace has no CUPTI kernel table. Profile with --trace=cuda,nvtx.")
    columns = _table_columns(db, table)
    name = "shortName" if "shortName" in columns else "demangledName"
    return [
        Kernel(*row[:-1], strings.get(row[-1], str(row[-1])))
        for row in db.execute(
            "SELECT start, end, globalPid, deviceId, contextId, streamId, "
            f"correlationId, {name} FROM {table} WHERE end > start ORDER BY start"
        )
    ]


def _read_copies(db, tables) -> list[Copy]:
    if _MEMCPY not in tables:
        raise ValueError("Trace has no CUPTI memcpy table. Profile with --trace=cuda,nvtx.")
    # CUDA_MEMCPY_KIND_HTOD is 1 in the CUPTI enumeration.
    kind = 1
    if "ENUM_CUDA_MEMCPY_OPER" in tables:
        row = db.execute(
            "SELECT id FROM ENUM_CUDA_MEMCPY_OPER WHERE name='CUDA_MEMCPY_KIND_HTOD'"
        ).fetchone()
        if row:
            kind = row[0]
    columns = _table_columns(db, _MEMCPY)
    source = "srcKind" if "srcKind" in columns else "NULL"
    memory_kinds = (
        dict(db.execute("SELECT id, label FROM ENUM_CUDA_MEM_KIND"))
        if "ENUM_CUDA_MEM_KIND" in tables else {}
    )
    return [
        Copy(*row[:-1], memory_kinds.get(row[-1], str(row[-1] or "unknown")))
        for row in db.execute(
            "SELECT start, end, globalPid, deviceId, contextId, streamId, "
            f"correlationId, bytes, {source} FROM {_MEMCPY} "
            "WHERE copyKind=? AND end > start ORDER BY start",
            (kind,),
        )
    ]


def _copy_attribution(
    copy: Copy,
    apis: dict[tuple[int, int], list[Api]],
    ranges_by_thread: dict[int | None, list[Range]],
) -> tuple[Api | None, Range | None]:
    if copy.global_pid is None:
        return None, None
    candidates = [
        api for api in apis.get((copy.global_pid >> 24, copy.correlation_id), ())
        if api.start_ns <= copy.start_ns and "memcpy" in api.name.lower()
    ]
    tagged = [
        (api, marker)
        for api in candidates
        for marker in ranges_by_thread[api.global_tid]
        if marker.start_ns <= api.start_ns <= api.end_ns <= marker.end_ns
    ]
    if tagged:
        return min(
            tagged,
            key=lambda pair: (pair[1].end_ns - pair[1].start_ns, pair[0].end_ns - pair[0].start_ns),
        )
    return (min(candidates, key=lambda api: api.end_ns - api.start_ns) if candidates else None, None)


def _competing_activity(
    db: sqlite3.Connection, tables: set[str], contexts: set[Context], kernels: list[Kernel]
) -> dict:
    devices = {context[1] for context in contexts}
    others = [
        kernel for kernel in kernels
        if kernel.device_id in devices and kernel.context not in contexts
    ]
    groups = Counter(kernel.context for kernel in others)
    other_rows = [
        {
            "global_pid": pid,
            "device_id": device,
            "context_id": context,
            "kernel_count": count,
        }
        for (pid, device, context), count in groups.items()
    ]
    switch_tables = sorted(
        table for table in tables
        if table != "ENUM_GPU_CTX_SWITCH"
        and ("GPU_CONTEXT_SWITCH" in table or "GPU_CTX_SWITCH" in table)
    )
    switch_counts = {
        table: db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        for table in switch_tables
        if table.replace("_", "").isalnum()
    }
    if other_rows:
        note = (
            "The export records kernels from other CUDA processes or contexts with the same "
            "reported device ordinal. Their intervals are excluded from the overlap calculation. "
            "Per-process device ordinals can be remapped. Confirm physical device identity "
            "before attributing contention to these workloads."
        )
    else:
        note = "No other CUDA process/context kernels on selected devices are present in this export."
    if not any(switch_counts.values()):
        note += (
            " GPU context switches were not recorded. This process trace cannot rule out contention "
            "from other GPU workloads."
        )
    else:
        note += " GPU context switch rows are inventoried separately and excluded from overlap."
    return {
        "other_contexts_on_selected_devices": other_rows,
        "context_switch_table_rows": switch_counts,
        "interpretation": note,
    }


def _kernel_category(kernel, apis, markers_by_thread):
    if kernel.global_pid is None:
        return "unattributed"
    associated = apis.get((kernel.global_pid >> 24, kernel.correlation_id), ())
    markers = [
        marker
        for api in associated
        for marker in markers_by_thread[api.global_tid]
        if marker.start_ns <= api.start_ns <= api.end_ns <= marker.end_ns
    ]
    if not markers:
        return "unattributed"
    enclosing = min(markers, key=lambda marker: marker.end_ns - marker.start_ns)
    return enclosing.text.split(".layer=", 1)[0]


def _partition_kernel_intervals(categories):
    """Count each time slice once, separating simultaneous categories."""
    events = defaultdict(Counter)
    for category, intervals in categories.items():
        for start, end in _union(intervals):
            events[start][category] += 1
            events[end][category] -= 1
    exclusive = Counter()
    inclusive = Counter()
    active = Counter()
    previous = None
    concurrent = 0
    for stamp, updates in sorted(events.items()):
        if previous is not None and active:
            elapsed = stamp - previous
            for category in active:
                inclusive[category] += elapsed
            if len(active) == 1:
                exclusive[next(iter(active))] += elapsed
            else:
                concurrent += elapsed
        active.update(updates)
        active += Counter()  # Remove categories whose intervals just ended.
        previous = stamp
    return exclusive, inclusive, concurrent


def _kernel_phases(kernels, apis, ranges):
    markers_by_thread = defaultdict(list)
    prefill_intervals = defaultdict(list)
    has_prefill_scope = False
    for marker in ranges:
        if marker.text.startswith("scene_kv."):
            markers_by_thread[marker.global_tid].append(marker)
        elif marker.text == _PREFILL_SCOPE and marker.global_tid is not None:
            prefill_intervals[marker.global_tid >> 24].append((marker.start_ns, marker.end_ns))
            has_prefill_scope = True
    scopes = {
        pid: _IntervalUnion(intervals) for pid, intervals in prefill_intervals.items()
    }
    by_context = defaultdict(lambda: defaultdict(list))
    kernel_counts = Counter()
    for kernel in kernels:
        if has_prefill_scope:
            scope = scopes.get(kernel.global_pid >> 24) if kernel.global_pid is not None else None
            intervals = scope.intersection(kernel.interval) if scope else []
            if not intervals:
                continue
        else:
            intervals = [kernel.interval]
        category = _kernel_category(kernel, apis, markers_by_thread)
        by_context[kernel.context][category].extend(intervals)
        kernel_counts[category] += 1
    exclusive = Counter()
    inclusive = Counter()
    concurrent = 0
    for categories in by_context.values():
        only, any_category, simultaneous = _partition_kernel_intervals(categories)
        exclusive.update(only)
        inclusive.update(any_category)
        concurrent += simultaneous
    return {
        "scope": _PREFILL_SCOPE if has_prefill_scope else "entire CUDA export",
        "kernel_count": sum(kernel_counts.values()),
        "kernel_union_ns": sum(exclusive.values()) + concurrent,
        "multi_category_concurrent_ns": concurrent,
        "categories": [
            {
                "name": category,
                "kernel_count": kernel_counts[category],
                "exclusive_ns": exclusive[category],
                "including_concurrent_ns": inclusive[category],
            }
            for category in sorted(kernel_counts)
        ],
        "interpretation": (
            "Attribute each GPU kernel to the innermost scene_kv NVTX range enclosing its "
            "correlated launch API. Total kernel time sums the interval union per CUDA context. "
            "Exclusive categories plus the multi-category time partition that union. "
            "Including-concurrent values count overlapping categories individually. "
            "Unattributed kernels still contribute to the total."
        ),
    }


def analyze(
    path: str | Path,
    *,
    prefixes: tuple[str, ...] = _DEFAULT_PREFIXES,
    all_h2d: bool = False,
    examples: int = 5,
) -> dict:
    root = Path(path).resolve()
    with closing(sqlite3.connect(root.as_uri() + "?mode=ro", uri=True)) as db:
        tables = _table_names(db)
        strings = _string_ids(db, tables)
        ranges = _read_ranges(db, tables, strings)
        apis = _read_runtime(db, tables, strings)
        kernels = _read_kernels(db, tables, strings)
        copies = _read_copies(db, tables)
        tagged_ranges = defaultdict(list)
        for marker in ranges:
            if marker.text.startswith(prefixes):
                tagged_ranges[marker.global_tid].append(marker)

        selected = []
        copy_api_matches = 0
        for copy in copies:
            api, marker = _copy_attribution(copy, apis, tagged_ranges)
            copy_api_matches += api is not None
            if all_h2d or marker is not None:
                selected.append((copy, api, marker))

        kernels_by_context = defaultdict(list)
        for kernel in kernels:
            kernels_by_context[kernel.context].append(kernel)
        copy_streams = defaultdict(set)
        for copy, _, _ in selected:
            copy_streams[copy.context].add(copy.stream_id)
        distinct_stream_unions = {
            (context, stream): _IntervalUnion(
                kernel.interval for kernel in kernels_by_context[context] if kernel.stream_id != stream
            )
            for context, streams in copy_streams.items()
            for stream in streams
        }
        copy_rows: list[dict] = []
        rows_by_context = defaultdict(list)
        for copy, api, marker in selected:
            intervals = distinct_stream_unions[copy.context, copy.stream_id].intersection(copy.interval)
            row = {
                **asdict(copy),
                "duration_ns": copy.end_ns - copy.start_ns,
                "overlap_ns": _duration(intervals),
                "overlap_intervals_ns": intervals,
                "api": asdict(api) if api is not None else None,
                "nvtx": asdict(marker) if marker is not None else None,
            }
            copy_rows.append(row)
            rows_by_context[copy.context].append(row)

        groups = []
        for context, rows in rows_by_context.items():
            pid, device, context_id = context
            compute = kernels_by_context[context]
            groups.append({
                "global_pid": pid,
                "process_id": (pid >> 24) & 0xFFFFFF if pid is not None else None,
                "device_id": device,
                "context_id": context_id,
                "copy_streams": sorted(copy_streams[context]),
                "kernel_streams": sorted({kernel.stream_id for kernel in compute}),
                "kernel_count": len(compute),
                "copy_union_ns": _duration(_union((row["start_ns"], row["end_ns"]) for row in rows)),
                "cross_stream_overlap_union_ns": _duration(_union(
                    interval for row in rows for interval in row["overlap_intervals_ns"]
                )),
            })

        witnesses = []
        order = sorted(
            range(len(selected)),
            key=lambda index: (-copy_rows[index]["overlap_ns"], copy_rows[index]["start_ns"]),
        )
        for index in order[:examples]:
            copy = selected[index][0]
            row = copy_rows[index]
            intersecting = [
                (
                    min(copy.end_ns, kernel.end_ns) - max(copy.start_ns, kernel.start_ns),
                    kernel,
                )
                for kernel in kernels_by_context[copy.context]
                if kernel.stream_id != copy.stream_id
                and kernel.start_ns < copy.end_ns and kernel.end_ns > copy.start_ns
            ]
            intersecting.sort(key=lambda pair: (-pair[0], pair[1].start_ns))
            witnesses.append({
                **row,
                "witness_kernels": [
                    {
                        "name": kernel.name,
                        "stream_id": kernel.stream_id,
                        "correlation_id": kernel.correlation_id,
                        "start_ns": kernel.start_ns,
                        "end_ns": kernel.end_ns,
                        "intersection_ns": overlap,
                    }
                    for overlap, kernel in intersecting[:3]
                ],
                "intersecting_kernel_count": len(intersecting),
            })

        scene_ranges = defaultdict(list)
        for marker in ranges:
            if marker.text.startswith("scene_kv."):
                scene_ranges[marker.text.split(".layer=", 1)[0]].append(marker)
        nvtx_summary = [
            {
                "name": family,
                "count": len(markers),
                "cpu_duration_sum_ns": sum(m.end_ns - m.start_ns for m in markers),
            }
            for family, markers in sorted(scene_ranges.items())
        ]
        competing = _competing_activity(db, tables, set(copy_streams), kernels)
        kernel_phases = _kernel_phases(kernels, apis, ranges)

    total_copy = sum(row["duration_ns"] for row in copy_rows)
    total_overlap = sum(row["overlap_ns"] for row in copy_rows)
    return {
        "sqlite": str(root),
        "timestamp_units": "nanoseconds relative to the Nsight capture",
        "method": (
            "Intersect each H2D device interval with the union of different-stream kernel "
            "intervals in the same process, device, and CUDA context. The summed copy "
            "ratio is copy-duration-weighted. Group union durations avoid double-counting "
            "concurrently active copies. NVTX durations refer to host API/staging ranges."
        ),
        "selection": {
            "all_h2d": all_h2d,
            "nvtx_prefixes": list(prefixes),
            "h2d_events_in_export": len(copies),
            "h2d_events_with_cuda_api_correlation": copy_api_matches,
            "kv_nvtx_ranges": sum(len(values) for values in tagged_ranges.values()),
            "selected_h2d_events_attributed_to_nvtx": sum(marker is not None for _, _, marker in selected),
            "unmatched_note": (
                "No H2D transfers matched the selected NVTX ranges. This does not constitute "
                "a measurement of zero overlap."
                if not selected and not all_h2d else None
            ),
        },
        "totals": {
            "h2d_copy_count": len(copy_rows),
            "h2d_bytes": sum(row["bytes"] for row in copy_rows),
            "copy_duration_sum_ns": total_copy,
            "copy_overlap_sum_ns": total_overlap,
            "copy_overlap_fraction": total_overlap / total_copy if total_copy else None,
            "copies_with_cross_stream_overlap": sum(row["overlap_ns"] > 0 for row in copy_rows),
            "h2d_source_memory_kinds": dict(Counter(row["source_memory_kind"] for row in copy_rows)),
        },
        "groups": groups,
        "nvtx_range_summary": nvtx_summary,
        "kernel_phase_summary": kernel_phases,
        "competing_gpu_activity": competing,
        "examples": witnesses,
        "copies": copy_rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Export first with nsys export --type sqlite --output TRACE.sqlite TRACE.nsys-rep.",
    )
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, help="Write JSON to this path, or print to stdout")
    parser.add_argument("--nvtx-prefix", action="append", help="Include copies enqueued in this NVTX range prefix")
    parser.add_argument("--all-h2d", action="store_true", help="Also include untagged H2D copies")
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args(argv)
    if args.examples < 0:
        parser.error("--examples must be nonnegative")
    try:
        summary = analyze(
            args.sqlite,
            prefixes=tuple(args.nvtx_prefix or _DEFAULT_PREFIXES),
            all_h2d=args.all_h2d,
            examples=args.examples,
        )
    except (ValueError, sqlite3.Error) as exception:
        parser.error(str(exception))
    serialized = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
