# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rank-local writers for opt-in DVAC evaluation telemetry."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

QUERY_COLUMNS = (
    "trace_row",
    "query_uid",
    "episode_idx",
    "eval_epoch",
    "query_idx",
    "action_slot_start",
    "source_env_rank",
    "stage_id",
    "local_env_slot",
    "reset_id",
    "rollout_rank",
    "video_worker_seed",
    "video_index",
    "video_tile_index",
    "video_pre_frame",
    "video_post_frame",
    "video_relpath",
    "success_before",
    "head_image_relpath",
    "left_wrist_image_relpath",
    "right_wrist_image_relpath",
)

EPISODE_COLUMNS = (
    "episode_uid",
    "episode_idx",
    "eval_epoch",
    "source_env_rank",
    "stage_id",
    "local_env_slot",
    "reset_id",
    "success",
    "success_at_end",
    "return",
    "reward",
    "action_slots",
    "final_query_idx",
    "final_action_slot",
    "termination_reason",
)


def _to_numpy(value: Any, *, float32: bool = False) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        array = value.numpy()
    else:
        array = np.asarray(value)
    if float32:
        array = array.astype(np.float32, copy=False)
    return array


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return value


def _metadata_value(metadata: dict[str, Any], key: str, index: int) -> Any:
    value = metadata[key]
    if isinstance(value, torch.Tensor):
        return _scalar(value[index])
    if isinstance(value, np.ndarray):
        return _scalar(value[index])
    if isinstance(value, (list, tuple)):
        return _scalar(value[index])
    return _scalar(value)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class DVACTelemetryWriter:
    """Collect one standalone evaluation and write one shard per rollout rank."""

    def __init__(
        self,
        output_dir: str,
        rank: int,
        *,
        save_query_inputs: bool,
        run_metadata: dict[str, Any],
        resolved_config: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.rank = int(rank)
        self.save_query_inputs = bool(save_query_inputs)
        self.run_metadata = dict(run_metadata)
        self.resolved_config = resolved_config
        self._arrays: dict[str, list[np.ndarray]] = {
            "x_chain": [],
            "z_endpoint": [],
            "final_model_action": [],
            "env_action": [],
            "robot_state": [],
        }
        self._timesteps: np.ndarray | None = None
        self._main_images: list[np.ndarray] = []
        self._wrist_images: list[np.ndarray] = []
        self._rows: list[dict[str, Any]] = []
        self._query_uids: set[str] = set()
        self._finalized = False

    @property
    def query_count(self) -> int:
        return len(self._rows)

    def append(
        self,
        *,
        telemetry: dict[str, Any],
        env_action: Any,
        env_obs: dict[str, Any],
        query_metadata: dict[str, Any],
    ) -> None:
        if self._finalized:
            raise RuntimeError("Cannot append after DVAC telemetry has been finalized.")

        x_chain = _to_numpy(telemetry["x_chain"], float32=True)
        z_endpoint = _to_numpy(telemetry["z_endpoint"], float32=True)
        timesteps = _to_numpy(telemetry["timesteps"], float32=True)
        final_model_action = _to_numpy(telemetry["final_model_action"], float32=True)
        env_action = _to_numpy(env_action, float32=True)
        robot_state = _to_numpy(env_obs["states"], float32=True)

        batch_size = x_chain.shape[0]
        batch_arrays = {
            "z_endpoint": z_endpoint,
            "final_model_action": final_model_action,
            "env_action": env_action,
            "robot_state": robot_state,
        }
        for name, array in batch_arrays.items():
            if array.shape[0] != batch_size:
                raise ValueError(
                    f"{name} batch {array.shape[0]} does not match x_chain batch {batch_size}."
                )
        if timesteps.ndim != 1:
            raise ValueError("DVAC timesteps must be the shared [M] schedule.")
        if self._timesteps is None:
            self._timesteps = np.array(timesteps, copy=True)
        elif not np.array_equal(self._timesteps, timesteps):
            raise ValueError("DVAC timestep schedule changed within one run.")

        self._arrays["x_chain"].append(x_chain)
        self._arrays["z_endpoint"].append(z_endpoint)
        self._arrays["final_model_action"].append(final_model_action)
        self._arrays["env_action"].append(env_action)
        self._arrays["robot_state"].append(robot_state)

        if self.save_query_inputs:
            main_images = _to_numpy(env_obs["main_images"])
            wrist_images = _to_numpy(env_obs["wrist_images"])
            if (
                main_images.shape[0] != batch_size
                or wrist_images.shape[0] != batch_size
            ):
                raise ValueError(
                    "Query images must be aligned with the telemetry batch."
                )
            if wrist_images.ndim != 5 or wrist_images.shape[1] < 2:
                raise ValueError("RoboTwin DVAC expects left and right wrist images.")
            self._main_images.append(main_images.astype(np.uint8, copy=False))
            self._wrist_images.append(wrist_images.astype(np.uint8, copy=False))

        first_trace_row = len(self._rows)
        for batch_index in range(batch_size):
            row = {
                key: _metadata_value(query_metadata, key, batch_index)
                for key in QUERY_COLUMNS
                if key
                not in {
                    "trace_row",
                    "query_uid",
                    "rollout_rank",
                    "head_image_relpath",
                    "left_wrist_image_relpath",
                    "right_wrist_image_relpath",
                }
            }
            row["trace_row"] = first_trace_row + batch_index
            row["rollout_rank"] = self.rank
            row["query_uid"] = (
                f"ep{int(row['episode_idx']):06d}_q{int(row['query_idx']):03d}_"
                f"reset{int(row['reset_id'])}"
            )
            if row["query_uid"] in self._query_uids:
                raise ValueError(f"Duplicate DVAC query_uid: {row['query_uid']}")
            self._query_uids.add(row["query_uid"])
            row["head_image_relpath"] = ""
            row["left_wrist_image_relpath"] = ""
            row["right_wrist_image_relpath"] = ""
            self._rows.append(row)

    def finalize(self) -> dict[str, str]:
        if self._finalized:
            raise RuntimeError("DVAC telemetry has already been finalized.")
        self._finalized = True
        self.output_dir.mkdir(parents=True, exist_ok=True)

        trace_path = self.output_dir / f"trace_rollout_rank{self.rank:02d}.npz"
        index_path = self.output_dir / f"query_index_rollout_rank{self.rank:02d}.csv"
        manifest_path = self.output_dir / f"manifest_rollout_rank{self.rank:02d}.json"
        reserved_paths = [trace_path, index_path, manifest_path]
        if self.rank == 0:
            reserved_paths.extend(
                [
                    self.output_dir / "resolved_config.yaml",
                    self.output_dir / "run_manifest.json",
                ]
            )
        existing_paths = [path for path in reserved_paths if path.exists()]
        image_dir = self.output_dir / "query_images" / f"rollout_rank{self.rank:02d}"
        if image_dir.exists() and any(image_dir.iterdir()):
            existing_paths.append(image_dir)
        if existing_paths:
            joined = ", ".join(str(path) for path in existing_paths)
            raise FileExistsError(
                f"Refusing to overwrite DVAC shard artifacts: {joined}"
            )

        arrays = {
            name: np.concatenate(parts, axis=0)
            for name, parts in self._arrays.items()
            if parts
        }
        if self._timesteps is not None:
            arrays["timesteps"] = self._timesteps
        np.savez_compressed(trace_path, **arrays)

        if self.save_query_inputs and self._rows:
            main_images = np.concatenate(self._main_images, axis=0)
            wrist_images = np.concatenate(self._wrist_images, axis=0)
            image_dir.mkdir(parents=True, exist_ok=True)
            for trace_row, row in enumerate(self._rows):
                image_paths = {
                    "head_image_relpath": image_dir / f"{row['query_uid']}__head.png",
                    "left_wrist_image_relpath": image_dir
                    / f"{row['query_uid']}__left_wrist.png",
                    "right_wrist_image_relpath": image_dir
                    / f"{row['query_uid']}__right_wrist.png",
                }
                Image.fromarray(main_images[trace_row]).save(
                    image_paths["head_image_relpath"]
                )
                Image.fromarray(wrist_images[trace_row, 0]).save(
                    image_paths["left_wrist_image_relpath"]
                )
                Image.fromarray(wrist_images[trace_row, 1]).save(
                    image_paths["right_wrist_image_relpath"]
                )
                for field, path in image_paths.items():
                    row[field] = path.relative_to(self.output_dir).as_posix()

        _write_csv(index_path, QUERY_COLUMNS, self._rows)

        manifest = {
            "schema_version": 1,
            "rollout_rank": self.rank,
            "query_count": self.query_count,
            "trace_file": trace_path.name,
            "query_index_file": index_path.name,
            "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
            "array_dtypes": {name: str(array.dtype) for name, array in arrays.items()},
            **self.run_metadata,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        if self.rank == 0 and self.resolved_config is not None:
            (self.output_dir / "resolved_config.yaml").write_text(
                self.resolved_config, encoding="utf-8"
            )
            rollout_world_size = int(self.run_metadata["rollout_world_size"])
            env_world_size = int(self.run_metadata["env_world_size"])
            run_manifest = {
                "schema_version": 1,
                "expected_rollout_shards": [
                    {
                        "rank": rank,
                        "trace": f"trace_rollout_rank{rank:02d}.npz",
                        "query_index": f"query_index_rollout_rank{rank:02d}.csv",
                        "manifest": f"manifest_rollout_rank{rank:02d}.json",
                    }
                    for rank in range(rollout_world_size)
                ],
                "expected_episode_shards": [
                    f"episode_index_env_rank{rank:02d}.csv"
                    for rank in range(env_world_size)
                ],
                **self.run_metadata,
            }
            (self.output_dir / "run_manifest.json").write_text(
                json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8"
            )

        return {
            "trace": str(trace_path),
            "query_index": str(index_path),
            "manifest": str(manifest_path),
        }


class DVACEpisodeWriter:
    """Write true reset-ID to episode-outcome joins from an EnvWorker rank."""

    def __init__(self, output_dir: str, rank: int):
        self.output_dir = Path(output_dir)
        self.rank = int(rank)
        self._rows: list[dict[str, Any]] = []
        self._finalized = False

    def append(
        self,
        *,
        query_metadata: dict[str, Any],
        env_info: dict[str, Any],
        newly_done: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        done_slots = np.flatnonzero(_to_numpy(newly_done).astype(bool))
        terminated = _to_numpy(terminated).astype(bool)
        truncated = _to_numpy(truncated).astype(bool)
        for filtered_index, local_env_slot in enumerate(done_slots):
            meta = {
                key: _metadata_value(query_metadata, key, int(local_env_slot))
                for key in (
                    "episode_idx",
                    "eval_epoch",
                    "source_env_rank",
                    "stage_id",
                    "local_env_slot",
                    "reset_id",
                    "query_idx",
                    "action_slot_start",
                    "action_chunk",
                )
            }

            def episode_value(key: str, default: Any = "") -> Any:
                if key not in env_info:
                    return default
                value = _to_numpy(env_info[key]).reshape(-1)
                return _scalar(value[filtered_index])

            if terminated[local_env_slot]:
                termination_reason = "termination"
            elif truncated[local_env_slot]:
                termination_reason = "truncation"
            else:
                termination_reason = "done"
            episode_uid = (
                f"ep{int(meta['episode_idx']):06d}_reset{int(meta['reset_id'])}"
            )
            self._rows.append(
                {
                    "episode_uid": episode_uid,
                    "episode_idx": meta["episode_idx"],
                    "eval_epoch": meta["eval_epoch"],
                    "source_env_rank": meta["source_env_rank"],
                    "stage_id": meta["stage_id"],
                    "local_env_slot": meta["local_env_slot"],
                    "reset_id": meta["reset_id"],
                    "success": episode_value("success_once", False),
                    "success_at_end": episode_value("success_at_end", False),
                    "return": episode_value("return"),
                    "reward": episode_value("reward"),
                    "action_slots": episode_value("episode_len"),
                    "final_query_idx": meta["query_idx"],
                    "final_action_slot": int(meta["action_slot_start"])
                    + int(meta["action_chunk"]),
                    "termination_reason": termination_reason,
                }
            )

    def finalize(self) -> str:
        if self._finalized:
            raise RuntimeError("DVAC episode telemetry has already been finalized.")
        self._finalized = True
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"episode_index_env_rank{self.rank:02d}.csv"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite DVAC episode shard: {path}")
        _write_csv(path, EPISODE_COLUMNS, self._rows)
        return str(path)
