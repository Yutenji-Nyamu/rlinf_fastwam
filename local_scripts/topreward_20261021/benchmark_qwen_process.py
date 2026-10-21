# Copyright 2026 The RLinf Authors and task-reward contributors.
# Licensed under the Apache License, Version 2.0.
"""Measure an actual 16-row scorer request and verify child offload/exit."""

import argparse
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from qwen_process import QwenProcess


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--physical-gpu", type=int, choices=(4, 5), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(a.physical_gpu):
        p.error("Bind the benchmark to the requested physical GPU")
    inventory = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"]))
    gpu = inventory.findall("gpu")[a.physical_gpu]
    if gpu.findall("processes/process_info"):
        raise RuntimeError("Benchmark card already has compute/graphics contexts")
    episodes = json.loads((a.dataset / "manifest.json").read_text())["episodes"]
    future, history, instructions, seeds = [], [], [], []
    for e in episodes:
        with np.load(a.dataset / e["file"], allow_pickle=False) as z:
            lengths = z["lengths"]
            pair = next(
                (
                    i
                    for i in range(1, len(lengths))
                    if lengths[i - 1] == lengths[i] == 8
                ),
                None,
            )
            if pair is None:
                continue
            frames = z["frames"]
            history.append(frames[pair - 1])
            future.append(frames[pair])
            instructions.append(e["instruction"])
            seeds.append(e["actual_seed"])
        if len(future) == 16:
            break
    if len(future) != 16:
        raise ValueError("Need 16 distinct native windows with two complete chunks")
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    reward = QwenProcess(a.python, a.model, a.physical_gpu, a.output)
    result = {
        "physical_gpu": a.physical_gpu,
        "gpu_uuid": gpu.findtext("uuid"),
        "cpu_load_seconds": time.monotonic() - started,
        "actual_seeds": seeds,
        "execution": "one actual 16-row pipe request; 16 serialized video forwards; no WM loaded",
    }
    try:
        started = time.monotonic()
        reward.to("cuda:0")
        result["onload_seconds"] = time.monotonic() - started
        started = time.monotonic()
        result["probabilities"] = reward.score_windows(
            np.stack(future), np.stack(history), np.full(16, 8, np.int64), instructions
        )
        result["request_seconds"] = time.monotonic() - started
        result["details"] = reward.last_details
        result["memory"] = reward.memory_usage()
        started = time.monotonic()
        reward.to("cpu")
        result["offload_seconds"] = time.monotonic() - started
        result["offload_ack"] = reward.last_residency
    finally:
        reward.close()
    result["child_exit_code"] = reward.child.returncode
    if reward.child.returncode != 0:
        raise RuntimeError("Scorer child did not exit cleanly")
    (a.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
