"""Optional, per-worker boundary telemetry. No CUDA initialization or synchronization.

Enable with WAN_GOAL_RESOURCE_DIR pointing to the new run's resource directory.
One JSONL per PID avoids cross-process writes. Recording errors never mask training.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time


def record_resource_boundary(worker, stage: str, *, reset_peak: bool = False) -> None:
    target = os.environ.get("WAN_GOAL_RESOURCE_DIR")
    if not target:
        return
    try:
        import torch

        row = {
            "time": time.time(), "pid": os.getpid(), "stage": stage,
            "role": getattr(worker, "_group_name", type(worker).__name__),
            "rank": getattr(worker, "_rank", None),
            "step": getattr(worker, "global_step", getattr(worker, "version", None)),
            "trajectory_step": getattr(worker, "_trajectory_step", None),
            "cuda_initialized": torch.cuda.is_initialized(),
        }
        for filename, fields in (("/proc/self/status", ("VmRSS", "VmHWM", "VmSwap")),
                                 ("/proc/meminfo", ("MemAvailable", "SwapFree"))):
            for line in Path(filename).read_text().splitlines():
                key, _, value = line.partition(":")
                if key in fields:
                    row[key + "_bytes"] = int(value.split()[0]) * 1024
        if row["cuda_initialized"]:
            stats = torch.cuda.memory_stats()
            row.update(
                cuda_device=torch.cuda.current_device(),
                cuda_allocated=torch.cuda.memory_allocated(),
                cuda_reserved=torch.cuda.memory_reserved(),
                cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved=torch.cuda.max_memory_reserved(),
                cuda_inactive_split_bytes=stats.get("inactive_split_bytes.all.current"),
                cuda_allocation_retries=stats.get("num_alloc_retries"),
                cuda_ooms=stats.get("num_ooms"),
            )
        directory = Path(target)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"boundary-{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        if reset_peak and row["cuda_initialized"]:
            torch.cuda.reset_peak_memory_stats()
    except Exception as error:
        # Best-effort diagnostic only: it must not affect an optimizer step or cleanup.
        try:
            print(f"RESOURCE_TELEMETRY_ERROR {stage}: {type(error).__name__}: {error}", flush=True)
        except Exception:
            pass
