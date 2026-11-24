"""Optional completion-timed scheduler stages for the isolated benchmark server."""

import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from contextlib import nullcontext
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

PROFILE_RANGE = "scenecache_post_warmup_prefill"
MODE_ENV = "SGLANG_VISUAL_KV_MODE"
DISABLE_DECODER_CACHE_ENV = "SGLANG_VISUAL_KV_DISABLE_DECODER_CACHE"
DISABLE_VISION_CACHE_ENV = "SGLANG_VISUAL_KV_DISABLE_VISION_CACHE"
PROFILE_ENV = "SGLANG_VISUAL_KV_PROFILE"
VERIFY_ENV = "SGLANG_VISUAL_KV_VERIFY_DIR"


def install():
    from scenecache_report_trace import install as install_cache_trace

    install_cache_trace()
    root = Path(os.environ["SCENECACHE_REPORT_DIR"])
    scheduler_module = "sglang.srt.managers.scheduler"
    processor_module = "sglang.srt.multimodal.processors.qwen_vl"
    mixed = os.getenv("SCENECACHE_BENCHMARK_MIXED") == "1"

    class ProcessorLoader(importlib.abc.Loader):
        def __init__(self, delegate):
            self.delegate = delegate

        def create_module(self, spec):
            return self.delegate.create_module(spec)

        def exec_module(self, module):
            self.delegate.exec_module(module)
            mode_override = ContextVar("benchmark_mode", default=None)

            class ProcessorEnvironment:
                def __getattr__(self, name):
                    return getattr(os, name)

                def getenv(self, name, default=None):
                    override = mode_override.get() if name == MODE_ENV else None
                    return override if override is not None else os.getenv(name, default)

            module.os = ProcessorEnvironment()
            processor = module.Qwen2_5VLImageProcessor
            original = processor.process_mm_data_async

            @wraps(original)
            async def process(self, image_data, input_text, request_obj, *args, **kwargs):
                control = json.loads((root / "control.json").read_text())
                if request_obj.rid != control["request_id"]:
                    raise ValueError("Interleaved benchmark only accepts its current image request")
                token = mode_override.set(control["bench_mode"])
                try:
                    return await original(self, image_data, input_text, request_obj, *args, **kwargs)
                finally:
                    mode_override.reset(token)

            processor.process_mm_data_async = process

    class SchedulerLoader(importlib.abc.Loader):
        def __init__(self, delegate):
            self.delegate = delegate

        def create_module(self, spec):
            return self.delegate.create_module(spec)

        def exec_module(self, module):
            self.delegate.exec_module(module)
            original = module.Scheduler.run_batch

            @wraps(original)
            def run_batch(scheduler, batch):
                if batch.forward_mode.is_extend():
                    stage = "prefill"
                elif batch.forward_mode.is_decode():
                    stage = "decode"
                else:
                    return original(scheduler, batch)
                import torch

                request_id = None
                try:
                    control = json.loads((root / "control.json").read_text())
                    request_id = control["request_id"]
                except FileNotFoundError:
                    return original(scheduler, batch)
                if mixed and (len(batch.reqs) != 1 or batch.reqs[0].rid != request_id):
                    return original(scheduler, batch)
                capture = stage == "prefill" and control.get("capture_profile", False)
                previous_env = {
                    name: os.environ.get(name)
                    for name in (
                        MODE_ENV,
                        DISABLE_DECODER_CACHE_ENV,
                        DISABLE_VISION_CACHE_ENV,
                        PROFILE_ENV,
                        VERIFY_ENV,
                    )
                }
                bench_mode = control.get("bench_mode")
                if stage == "prefill":
                    torch.cuda.reset_peak_memory_stats()
                start_ns = time.monotonic_ns()
                try:
                    if bench_mode == "legacy":
                        os.environ.pop(MODE_ENV, None)
                    elif bench_mode:
                        os.environ[MODE_ENV] = bench_mode
                    if control.get("disable_decoder_cache", False):
                        os.environ[DISABLE_DECODER_CACHE_ENV] = "1"
                    if control.get("disable_vision_cache", False):
                        os.environ[DISABLE_VISION_CACHE_ENV] = "1"
                    if verify_dir := control.get("verify_dir"):
                        os.environ[VERIFY_ENV] = verify_dir
                        os.environ.pop(PROFILE_ENV, None)
                    with torch.cuda.nvtx.range(
                        PROFILE_RANGE
                    ) if capture else nullcontext():
                        condition = control.get("condition")
                        label = (
                            torch.cuda.nvtx.range(
                                f"scenecache_benchmark_mode={condition}"
                            )
                            if capture
                            else nullcontext()
                        )
                        with label:
                            result = original(scheduler, batch)
                            # Outside model execution, once per scheduler step. Use the
                            # same synchronization policy for every benchmark mode.
                            torch.cuda.synchronize()
                finally:
                    for name, value in previous_env.items():
                        if value is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = value
                finish_ns = time.monotonic_ns()
                row = dict(
                    request_id=request_id,
                    stage=stage,
                    worker_pid=os.getpid(),
                    kv_length=int(batch.seq_lens_cpu[0]) if mixed else None,
                    start_ns=start_ns,
                    finish_ns=finish_ns,
                    process_gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    process_gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
                with (root / "scheduler_timing.jsonl").open("a") as output:
                    output.write(json.dumps(row) + "\n")
                return result

            module.Scheduler.run_batch = run_batch

    class SchedulerFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != scheduler_module and not (mixed and fullname == processor_module):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.loader:
                loader = SchedulerLoader if fullname == scheduler_module else ProcessorLoader
                spec.loader = loader(spec.loader)
            return spec

    sys.meta_path.insert(0, SchedulerFinder())
