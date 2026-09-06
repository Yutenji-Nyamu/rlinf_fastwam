"""Observation-only Fast-WAM action denoising telemetry.

This module is intentionally independent of RLinf.  It writes tensors already
computed by the official standalone inference loop and never samples RNG or
changes the action queue.
"""

from __future__ import annotations

import atexit
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from PIL import Image


SCHEMA_VERSION = "fastwam-dvac-observation-v1"


def _as_numpy(value: Any, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def compute_z_endpoint(
    x_chain: Any,
    v_chain: Any,
    timesteps: Any,
    *,
    num_train_timesteps: int = 1000,
) -> np.ndarray:
    """Reconstruct ``z_i = x_i - tau_i * v_i`` from a raw Fast-WAM trace."""
    x = _as_numpy(x_chain, dtype=np.float32)
    v = _as_numpy(v_chain, dtype=np.float32)
    t = _as_numpy(timesteps, dtype=np.float32).reshape(-1)
    if x.ndim != 3 or v.ndim != 3:
        raise ValueError(f"Expected x/v rank 3, got {x.shape} and {v.shape}")
    if x.shape[0] != v.shape[0] + 1 or x.shape[1:] != v.shape[1:]:
        raise ValueError(f"Expected x=[M+1,H,D], v=[M,H,D], got {x.shape} and {v.shape}")
    if t.shape != (v.shape[0],):
        raise ValueError(f"Expected {v.shape[0]} timesteps, got {t.shape}")
    if num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be positive")
    tau = t.reshape(-1, 1, 1) / float(num_train_timesteps)
    return x[:-1] - tau * v


def _git_head(project_root: Path) -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class FastWAMDvacTelemetryWriter:
    """Write one standalone process's query and episode telemetry."""

    QUERY_FIELDS = (
        "run_id",
        "policy",
        "task",
        "instruction",
        "episode_id",
        "reset_id",
        "query_idx",
        "query_start_action_slot",
        "query_end_action_slot_exclusive",
        "planned_exec_length",
        "executed_length",
        "success_before",
        "success_after",
        "trace_path",
        "head_image_path",
        "left_image_path",
        "right_image_path",
        "video_query_index",
        "video_frame_start",
        "video_frame_end_exclusive",
        "terminal_success_video_frame",
    )
    EPISODE_FIELDS = (
        "run_id",
        "policy",
        "task",
        "episode_id",
        "reset_id",
        "source_seed",
        "success",
        "total_action_slots",
        "queries",
        "first_success_action_slot",
        "first_success_query",
        "video_path",
    )

    def __init__(
        self,
        output_dir: str | Path,
        *,
        manifest: Mapping[str, Any],
        project_root: str | Path,
        skip_get_obs_within_replan: bool,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        if self.output_dir.exists():
            raise FileExistsError(f"DVAC telemetry output already exists: {self.output_dir}")
        self.traces_dir = self.output_dir / "traces"
        self.images_dir = self.output_dir / "query_images"
        self.traces_dir.mkdir(parents=True)
        self.images_dir.mkdir(parents=True)

        self.run_id = str(manifest["run_id"])
        self.policy = str(manifest.get("policy", "fastwam"))
        self.skip_get_obs_within_replan = bool(skip_get_obs_within_replan)
        self.queries_path = self.output_dir / "queries.csv"
        self.episodes_path = self.output_dir / "episodes.csv"
        self.manifest_path = self.output_dir / "run_manifest.json"
        self._manifest = dict(manifest)
        self._manifest.update(
            {
                "telemetry_schema_version": SCHEMA_VERSION,
                "instrumented_source_head": _git_head(Path(project_root)),
                "output_dir": str(self.output_dir),
                "argv": list(sys.argv),
                "video_alignment": (
                    "fresh_pre_action_frame_per_executed_action"
                    if not self.skip_get_obs_within_replan
                    else "query_observation_repeated_within_replan"
                ),
                "action_slot_indexing": "zero_based_end_exclusive",
                "source_seed_contract": "not_exposed_by_policy_api; join accepted seed from official evaluator log",
            }
        )
        self._write_manifest()

        self._episode: Optional[dict[str, Any]] = None
        self._query: Optional[dict[str, Any]] = None
        self._closed = False
        atexit.register(self.close)

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _append_csv(path: Path, row: Mapping[str, Any], fieldnames: tuple[str, ...]) -> None:
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    def begin_episode(self, episode_id: int) -> None:
        if self._episode is not None:
            self.finalize_episode(success=False)
        self._episode = {
            "run_id": self.run_id,
            "policy": self.policy,
            "task": "",
            "episode_id": int(episode_id),
            "reset_id": "",
            "source_seed": "",
            "success": False,
            "total_action_slots": 0,
            "queries": 0,
            "first_success_action_slot": "",
            "first_success_query": "",
            "video_path": "",
        }

    def _save_query_images(
        self, observation: Mapping[str, Any], episode_id: int, query_idx: int
    ) -> dict[str, str]:
        obs = observation["observation"]
        result: dict[str, str] = {}
        for name, key in (("head", "head_camera"), ("left", "left_camera"), ("right", "right_camera")):
            relative = Path("query_images") / f"episode{episode_id:04d}_query{query_idx:04d}_{name}.png"
            rgb = _as_numpy(obs[key]["rgb"], dtype=np.uint8)
            Image.fromarray(rgb, mode="RGB").save(self.output_dir / relative)
            result[f"{name}_image_path"] = relative.as_posix()
        return result

    def record_query(
        self,
        *,
        task: str,
        instruction: str,
        reset_id: Any,
        query_start_action_slot: int,
        planned_exec_length: int,
        observation: Mapping[str, Any],
        robot_state: Any,
        final_model_action: Any,
        env_action: Any,
        trace: Mapping[str, Any],
    ) -> None:
        if self._episode is None:
            self.begin_episode(episode_id=0)
        self._finish_query()
        assert self._episode is not None

        x_chain = _as_numpy(trace["x_chain"], dtype=np.float32)
        v_chain = _as_numpy(trace["v_chain"], dtype=np.float32)
        timesteps = _as_numpy(trace["timesteps"], dtype=np.float32).reshape(-1)
        deltas = _as_numpy(trace["deltas"], dtype=np.float32).reshape(-1)
        x_next = _as_numpy(trace["x_next"], dtype=np.float32)
        if x_chain.shape[0] != v_chain.shape[0] + 1 or x_chain.shape[1:] != v_chain.shape[1:]:
            raise ValueError(f"Invalid x/v chain shapes: {x_chain.shape}, {v_chain.shape}")
        if timesteps.shape != (v_chain.shape[0],) or deltas.shape != timesteps.shape:
            raise ValueError("Timestep/delta count does not match denoising steps")
        if x_next.shape != v_chain.shape or not np.array_equal(x_chain[1:], x_next):
            raise ValueError("x_next must exactly equal x_chain[1:]")

        query_idx = int(self._episode["queries"])
        episode_id = int(self._episode["episode_id"])
        trace_relative = Path("traces") / f"episode{episode_id:04d}_query{query_idx:04d}.npz"
        np.savez_compressed(
            self.output_dir / trace_relative,
            x_chain=x_chain,
            v_chain=v_chain,
            timesteps=timesteps,
            deltas=deltas,
            x_next=x_next,
            final_model_action=_as_numpy(final_model_action, dtype=np.float32),
            env_action=_as_numpy(env_action, dtype=np.float32),
            robot_state=_as_numpy(robot_state, dtype=np.float32),
        )
        image_paths = self._save_query_images(observation, episode_id, query_idx)

        self._episode["task"] = str(task)
        self._episode["reset_id"] = str(reset_id)
        self._episode["queries"] += 1
        self._query = {
            "run_id": self.run_id,
            "policy": self.policy,
            "task": str(task),
            "instruction": str(instruction),
            "episode_id": episode_id,
            "reset_id": str(reset_id),
            "query_idx": query_idx,
            "query_start_action_slot": int(query_start_action_slot),
            "query_end_action_slot_exclusive": int(query_start_action_slot),
            "planned_exec_length": int(planned_exec_length),
            "executed_length": 0,
            "success_before": False,
            "success_after": False,
            "trace_path": trace_relative.as_posix(),
            "video_query_index": query_idx,
            "video_frame_start": "" if self.skip_get_obs_within_replan else int(query_start_action_slot),
            "video_frame_end_exclusive": "",
            "terminal_success_video_frame": "",
            **image_paths,
        }

        if "resolved_action_timesteps" not in self._manifest:
            self._manifest.update(
                {
                    "resolved_action_timesteps": timesteps.tolist(),
                    "resolved_action_deltas": deltas.tolist(),
                    "raw_shapes_per_query": {
                        "x_chain": list(x_chain.shape),
                        "v_chain": list(v_chain.shape),
                        "x_next": list(x_next.shape),
                    },
                }
            )
            self._write_manifest()

    def record_action(self, *, success_after: bool) -> None:
        if self._episode is None or self._query is None:
            raise RuntimeError("No active DVAC query for executed action")
        action_slot = int(self._episode["total_action_slots"])
        self._episode["total_action_slots"] = action_slot + 1
        self._query["executed_length"] += 1
        self._query["query_end_action_slot_exclusive"] = action_slot + 1
        self._query["success_after"] = bool(success_after)
        if not self.skip_get_obs_within_replan:
            self._query["video_frame_end_exclusive"] = action_slot + 1
        if success_after:
            self._episode["first_success_action_slot"] = action_slot
            self._episode["first_success_query"] = self._query["query_idx"]
            if not self.skip_get_obs_within_replan:
                self._query["terminal_success_video_frame"] = action_slot + 1
            self.finalize_episode(success=True)

    def _finish_query(self) -> None:
        if self._query is None:
            return
        self._append_csv(self.queries_path, self._query, self.QUERY_FIELDS)
        self._query = None

    def finalize_episode(self, *, success: bool) -> None:
        if self._episode is None:
            return
        self._finish_query()
        self._episode["success"] = bool(success)
        video_dir = self._manifest.get("official_video_dir")
        if video_dir and self._episode["reset_id"] != "":
            randomized = "randomized" in str(self._manifest.get("task_config", "")).lower()
            self._episode["video_path"] = str(
                Path(str(video_dir))
                / (
                    f"episode{self._episode['reset_id']}_randomized-{str(randomized).lower()}_"
                    f"success-{str(bool(success)).lower()}.mp4"
                )
            )
        if int(self._episode["queries"]) > 0:
            self._append_csv(self.episodes_path, self._episode, self.EPISODE_FIELDS)
        self._episode = None

    def close(self) -> None:
        if self._closed:
            return
        if self._episode is not None:
            self.finalize_episode(success=False)
        self._closed = True
