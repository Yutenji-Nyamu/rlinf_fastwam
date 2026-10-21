# Copyright 2026 The RLinf Authors and task-reward contributors.
# Licensed under the Apache License, Version 2.0.
"""One owned Qwen child process isolates Transformers from the OpenDW runtime.

The existing WM service owns this child, and forwards its synchronous onload /
offload operations. Scoring has no hidden episode history or public endpoint.
"""

from __future__ import annotations

import io
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


def read_packet(stream) -> bytes | None:
    """Read one length-prefixed packet, rejecting truncated transfers."""
    header = stream.read(8)
    if not header:
        return None
    if len(header) != 8:
        raise EOFError("Truncated scorer header")
    length = struct.unpack("!Q", header)[0]
    if not 0 < length <= 256 * 1024**2:
        raise ValueError("Invalid scorer packet size")
    body = stream.read(length)
    if len(body) != length:
        raise EOFError("Truncated scorer body")
    return body


def write_packet(stream, body: bytes) -> None:
    """Write a packet atomically with respect to the caller's service lock."""
    stream.write(struct.pack("!Q", len(body)))
    stream.write(body)
    stream.flush()


class QwenProcess(nn.Module):
    """Module-shaped lifecycle adapter for a separately installed frozen VLM."""

    def __init__(self, python: Path, model_path: Path, physical_gpu: int, output: Path):
        super().__init__()
        self.model_path = str(model_path)
        self.physical_gpu = int(physical_gpu)
        self.last_details: list[dict] = []
        script = Path(__file__).resolve()
        env = dict(
            os.environ,
            PYTHONPATH=str(script.parent),
            CUDA_VISIBLE_DEVICES=str(physical_gpu),
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            USE_TF="0",
            USE_FLAX="0",
        )
        self.log = (output / "qwen-worker.log").open("x")
        self.child = subprocess.Popen(
            [
                str(python),
                "-u",
                "-B",
                str(script),
                "--worker",
                str(model_path),
                str(physical_gpu),
            ],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
        )
        try:
            self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self) -> dict:
        body = read_packet(self.child.stdout)
        if body is None:
            raise RuntimeError("Qwen worker exited; inspect qwen-worker.log")
        result = json.loads(body)
        if result.get("ok") is not True:
            raise RuntimeError("Qwen worker failed: " + str(result.get("error")))
        return result

    def _call(self, op: str, **arrays) -> dict:
        packet = io.BytesIO()
        np.savez(packet, op=np.asarray(op), **arrays)
        write_packet(self.child.stdin, packet.getvalue())
        return self._receive()

    def to(self, device):
        """Synchronize the child's model residency before acknowledging a move."""
        target = str(device)
        if target not in {"cpu", "cuda:0"}:
            raise ValueError("Qwen worker is bound to one physical GPU")
        r = self._call(target)
        if r["device"] != target or (target == "cpu" and r["non_cpu_tensors"] != 0):
            raise RuntimeError("Qwen worker residency acknowledgement is inconsistent")
        self.last_residency = r
        return self

    def memory_usage(self) -> dict:
        """Read peak memory only when an explicit benchmark requests it."""
        return self._call("memory")

    def score_windows(self, future, history, lengths, instructions) -> list[float]:
        """Score every row once, with its own caller-supplied episode history."""
        result = self._call(
            "score",
            future=future,
            history=history,
            lengths=lengths,
            instructions=np.asarray(instructions),
        )
        details = result["details"]
        values = [r["probability"] for r in details]
        if len(values) != len(future) or not all(
            np.isfinite(v) and 0 <= v <= 1 for v in values
        ):
            raise RuntimeError("Invalid scorer batch result")
        self.last_details = details
        return values

    def deployment_metadata(self) -> dict:
        """Return the frozen scorer contract for the existing service manifest."""
        return {
            "reward_architecture": "Qwen3-VL-8B-Instruct",
            "reward_window_frames": 16,
            "reward_threshold_owner": "RLinf environment",
            "reward_output": "raw P(next token = ' True')",
            "reward_worker_pid": self.child.pid,
            "reward_worker_physical_gpu": self.physical_gpu,
            "reward_history_owner": "explicit caller state; reset and get_state/set_state",
            "reward_instructions": "PR1554 exact assertion prompt",
        }

    def close(self) -> None:
        """Close only this Popen child; inherited owner tags also cover recovery."""
        if self.child.stdin:
            try:
                self.child.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.child.terminate()
            try:
                self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait(timeout=10)
        if self.child.stdout:
            self.child.stdout.close()
        self.log.close()


def worker(model_path: str, physical_gpu: int) -> None:
    """Serve the parent pipe; no text or logging is written to protocol stdout."""
    from qwen_reward import QwenReward, window_frames

    torch.set_num_threads(4)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise ValueError("Scorer GPU mask mismatch")
    model = QwenReward(model_path)

    def reply(**data):
        write_packet(sys.stdout.buffer, json.dumps(dict(ok=True, **data)).encode())

    reply(device="cpu")
    while (body := read_packet(sys.stdin.buffer)) is not None:
        with np.load(io.BytesIO(body), allow_pickle=False) as z:
            op = str(z["op"])
            if op in {"cpu", "cuda:0"}:
                if op == "cuda:0":
                    inventory = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,uuid",
                            "--format=csv,noheader",
                        ],
                        text=True,
                    )
                    expected = [
                        s.split(",")[1].strip()
                        for s in inventory.splitlines()
                        if int(s.split(",")[0]) == physical_gpu
                    ][0]
                    actual = str(torch.cuda.get_device_properties(0).uuid)
                    if actual.lower().removeprefix(
                        "gpu-"
                    ) != expected.lower().removeprefix("gpu-"):
                        raise RuntimeError(
                            "Scorer CUDA UUID differs from authorized GPU"
                        )
                model.to(op)
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                    if op == "cpu":
                        torch.cuda.empty_cache()
                residual = sum(
                    t.device.type != "cpu"
                    for t in list(model.parameters()) + list(model.buffers())
                )
                reply(device=op, non_cpu_tensors=residual)
            elif op == "memory":
                reply(
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                )
            elif op == "score":
                future, history, lengths, instructions = (
                    z["future"],
                    z["history"],
                    z["lengths"],
                    z["instructions"],
                )
                if model.model.device.type != "cuda":
                    raise RuntimeError("Scorer must be onloaded before inference")
                details = [
                    model.score(
                        window_frames(history[i, : int(lengths[i])], frames),
                        str(instructions[i]),
                    )
                    for i, frames in enumerate(future)
                ]
                reply(details=details)
            else:
                raise ValueError("Unknown scorer operation")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("Usage: qwen_process.py --worker MODEL PHYSICAL_GPU")
    try:
        worker(sys.argv[2], int(sys.argv[3]))
    except BaseException as error:
        # Model-load errors occur before the initial ready packet.
        try:
            write_packet(
                sys.stdout.buffer,
                json.dumps({"ok": False, "error": repr(error)}).encode(),
            )
        except BrokenPipeError:
            pass
        raise
