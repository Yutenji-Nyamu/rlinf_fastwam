"""Small data contracts shared by the appendix collectors."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")

    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(f"Unsupported JSON value: {type(x)}")

    tmp.write_text(
        json.dumps(
            value, indent=2, ensure_ascii=False, allow_nan=False, default=convert
        )
    )
    tmp.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    with np.load(tmp, allow_pickle=False) as z:
        assert set(z.files) == set(arrays)
        for key, value in arrays.items():
            np.testing.assert_array_equal(z[key], value)
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def entropy(actions, epsilon=1e-6):
    """[B,K,H,D] diagonal-Gaussian plug-in entropy, actual normalized D only.

    Same ddof/epsilon as tools/pi05_policy_eval/entropy_analysis.py. The result
    is action disagreement, not a calibrated probability of task failure.
    """
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 4 or a.shape[1] < 2 or not np.isfinite(a).all():
        raise ValueError("Expected finite [B,K,H,D] with K >= 2")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return 0.5 * np.log(2 * np.pi * np.e * (a.var(axis=1, ddof=1) + epsilon)).sum(-1)


def uniform_indices(length, count):
    if length <= 0 or count <= 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(
        np.rint(np.linspace(0, length - 1, min(length, count))).astype(int)
    )


def causal_prefix(anchor, count=8):
    """Eight real past frames, repeating only when the prefix is shorter."""
    if anchor < 0 or count < 1:
        raise ValueError("Invalid causal prefix")
    return np.rint(np.linspace(0, anchor, count)).astype(np.int64)


def seed_for(bank, observation, sample):
    if not (0 <= bank < 10 and 0 <= observation < 10000 and 0 <= sample < 32):
        raise ValueError("Seed index outside the declared experiment")
    return 111100000 + bank * 1000000 + observation * 32 + sample


def cuda_guard(gpu=5):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("Collector must inherit physical GPU5 isolation")
    import torch

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one visible GPU")
    return torch


class Timer:
    def __enter__(self):
        import torch

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        import torch

        torch.cuda.synchronize()
        self.result = dict(
            seconds=time.perf_counter() - self.start,
            allocated_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            reserved_peak_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
