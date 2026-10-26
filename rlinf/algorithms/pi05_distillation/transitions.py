# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Atomic, lossless decision-boundary transitions; no videos or signal collection."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import balanced_indices

OBS_KEYS = ("main_images", "wrist_images", "states")


def array(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def append(h, key, value):
    value = np.asarray(value)
    if key not in h:
        h.create_dataset(
            key,
            shape=(0, *value.shape),
            maxshape=(None, *value.shape),
            dtype=value.dtype,
            chunks=(1, *value.shape),
            compression="lzf",
        )
    ds = h[key]
    ds.resize(len(ds) + 1, axis=0)
    ds[-1] = value


class TransitionWriter:
    """Only complete native success/limit episodes enter the training index."""

    def __init__(self, out, config, seeds, obs):
        self.directory = Path(out) / "transitions"
        self.directory.mkdir()
        self.config, self.files, self.paths = config, [], []
        self.limit = config["step_limit"]
        actual = [seed["actual"] for seed in seeds]
        if len(set(actual)) != len(actual):
            raise ValueError("Duplicate native actual seed")
        if set(actual) & set(config.get("forbidden_actual_seeds", [])):
            raise ValueError("Collection seed overlaps held-out evaluation")
        for seed in seeds:
            i = seed["slot"]
            path = self.directory / f"episode-{seed['actual']}.partial.h5"
            h = h5py.File(path, "x")
            meta = {
                "schema": 1,
                "task": config["task"],
                "task_id": config["global_task_id"],
                "instruction": obs["task_descriptions"][i],
                "seed": seed,
                "config": config,
                "observation_format": "raw_evaluator_rgb_uint8_native_resize_on_load",
                "reward_semantics": "first_success_macro_decision_finite_official_budget",
            }
            h.attrs["metadata"] = json.dumps(meta)
            h.attrs["complete"] = False
            for key in OBS_KEYS:
                append(h, "obs/" + key, array(obs[key])[i])
            self.files.append(h)
            self.paths.append(path)
        self.rows = []

    def step(
        self,
        obs,
        actions,
        normalized,
        active,
        before,
        after,
        success_before,
        success_after,
        native,
        query,
        noise_seed,
    ):
        raw_reward, terminated, truncated = native
        for i in np.flatnonzero(active):
            h = self.files[i]
            if h is None:
                raise ValueError("Completed slot submitted again")
            length = int(after[i] - before[i])
            offered = min(10, self.limit - int(before[i]))
            if length != offered or not 0 < length <= 10:
                raise ValueError(
                    f"Native submitted count mismatch {length} != {offered}"
                )
            done = bool(success_after[i]) or after[i] >= self.limit
            if (bool(terminated[i]) or bool(truncated[i])) and not done:
                raise ValueError("Unexpected native end before success/official budget")
            reward = float(success_after[i] and not success_before[i])
            row = {
                "command": actions[i, :10],
                "predicted_h50": normalized[i, :50, :14],
                "action": normalized[i, :10, :14],
                "mask": (np.arange(10) < length).astype(np.float32),
                "submitted_length": length,
                "reward": reward,
                "reward_raw": float(np.asarray(raw_reward[i]).reshape(-1)[0]),
                "terminated_native": bool(terminated[i]),
                "truncated_native": bool(truncated[i]),
                "success_before": bool(success_before[i]),
                "success_after": bool(success_after[i]),
                "bootstrap_mask": float(not done),
                "remaining_before": self.limit - int(before[i]),
                "remaining_after": self.limit - int(after[i]),
                "before": int(before[i]),
                "after": int(after[i]),
                "query": query,
                "noise_seed": noise_seed,
            }
            for key, value in row.items():
                append(h, key, value)
            for key in OBS_KEYS:
                append(h, "obs/" + key, array(obs[key])[i])
            if done:
                h.attrs["complete"] = True
                h.attrs["success"] = bool(success_after[i])
                h.attrs["termination_reason"] = (
                    "success" if success_after[i] else "official_step_limit"
                )
                # Exact readback on the final real observation, once per episode.
                for key in OBS_KEYS:
                    np.testing.assert_array_equal(
                        h["obs/" + key][-1], array(obs[key])[i]
                    )
                n = len(h["reward"])
                total_reward = float(h["reward"][:].sum())
                assert (
                    total_reward == float(success_after[i])
                    and h["bootstrap_mask"][-1] == 0
                )
                h.close()
                target = self.paths[i].with_name(
                    self.paths[i].name.replace(".partial", "")
                )
                self.paths[i].replace(target)
                self.files[i] = None
                self.rows.append(
                    {
                        "path": str(target),
                        "transitions": n,
                        "success": bool(success_after[i]),
                        "final_observation_exact": True,
                        "bytes": target.stat().st_size,
                    }
                )

    def close(self):
        for h in self.files:
            if h is not None:
                h.close()
        (self.directory / "receipt.json").write_text(json.dumps(self.rows, indent=2))


def build_manifest(roots, output):
    records, used = [], set()
    for root in roots:
        for path in sorted(Path(root).rglob("episode-*.h5")):
            if ".partial." in path.name:
                continue
            with h5py.File(path, "r") as h:
                if not h.attrs.get("complete", False):
                    continue
                meta = json.loads(h.attrs["metadata"])
                n = len(h["reward"])
                identity = (meta["task"], meta["seed"]["actual"])
                if identity in used:
                    raise ValueError(f"Repeated task/actual seed: {identity}")
                used.add(identity)
                assert len(h["obs/states"]) == n + 1 and n > 0
                assert all(len(h["obs/" + k]) == n + 1 for k in OBS_KEYS)
                assert h["bootstrap_mask"][-1] == 0 and h["bootstrap_mask"][:-1].all()
                assert h["reward"][:].sum() == float(h.attrs["success"])
                assert np.all(h["after"][:-1] == h["before"][1:])
                assert np.all(
                    h["submitted_length"][:] == h["after"][:] - h["before"][:]
                )
                assert np.array_equal(
                    h["mask"][:],
                    np.arange(10)[None] < h["submitted_length"][:][:, None],
                )
                records.append(
                    {
                        "path": str(path.resolve()),
                        "task": meta["task"],
                        "task_id": meta["task_id"],
                        "actual_seed": meta["seed"]["actual"],
                        "transitions": n,
                        "success": bool(h.attrs["success"]),
                        "bytes": path.stat().st_size,
                    }
                )
    if not records:
        raise ValueError("No valid complete transitions")
    data = {"schema": 1, "episodes": records}
    Path(output).write_text(json.dumps(data, indent=2))
    return data


class Transitions:
    """Balanced tasks, natural within-task transition mixture; stateless sampling."""

    def __init__(self, manifest):
        self.manifest = json.loads(Path(manifest).read_text())
        self.names = sorted({r["task"] for r in self.manifest["episodes"]})
        self.rows = []
        sizes = []
        for name in self.names:
            before = len(self.rows)
            for ep in self.manifest["episodes"]:
                if ep["task"] == name:
                    self.rows.extend((ep["path"], i) for i in range(ep["transitions"]))
            sizes.append(len(self.rows) - before)
        self.sizes = np.asarray(sizes)
        self.cache = {}

    def __len__(self):
        return len(self.rows)

    def indices(self, step, batch, seed):
        return balanced_indices(self.sizes, step, batch, seed, self.cache)

    def batch(self, indices):
        samples = [None] * len(indices)
        groups = {}
        for pos, idx in enumerate(indices):
            path, index = self.rows[int(idx)]
            groups.setdefault(path, []).append((pos, index))
        for path, selection in groups.items():
            with h5py.File(path, "r") as h:
                meta = json.loads(h.attrs["metadata"])
                for pos, i in selection:
                    samples[pos] = dict(
                        obs={k: h["obs/" + k][i] for k in OBS_KEYS},
                        next_obs={k: h["obs/" + k][i + 1] for k in OBS_KEYS},
                        prompt=meta["instruction"],
                        limit=meta["config"]["step_limit"],
                        **{
                            k: h[k][i]
                            for k in (
                                "action",
                                "mask",
                                "reward",
                                "bootstrap_mask",
                                "remaining_before",
                                "remaining_after",
                            )
                        },
                    )
        result = {}
        for which in ("obs", "next_obs"):
            result[which] = {
                k: torch.from_numpy(np.stack([s[which][k] for s in samples]))
                for k in OBS_KEYS
            }
            result[which]["task_descriptions"] = [s["prompt"] for s in samples]
        for k in (
            "action",
            "mask",
            "reward",
            "bootstrap_mask",
            "remaining_before",
            "remaining_after",
            "limit",
        ):
            result[k] = torch.as_tensor(np.stack([s[k] for s in samples])).float()
        return result
