# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any

import torch

from rlinf.data.schema.embodied_types import Trajectory


def _align_bootstrap_field(
    tensor: torch.Tensor,
    trajectory_length: int,
    field_name: str,
) -> torch.Tensor:
    """Drop the leading bootstrap slot from each rollout epoch."""
    if tensor.shape[0] == trajectory_length:
        return tensor
    extra = int(tensor.shape[0] - trajectory_length)
    if extra <= 0 or trajectory_length % extra != 0:
        raise ValueError(
            f"Cannot align {field_name}: length={tensor.shape[0]}, "
            f"trajectory_length={trajectory_length}"
        )
    epoch_length = trajectory_length // extra
    expected = extra * (epoch_length + 1)
    if tensor.shape[0] != expected:
        raise ValueError(
            f"Invalid bootstrap layout for {field_name}: "
            f"length={tensor.shape[0]}, expected={expected}"
        )
    aligned = tensor.reshape(extra, epoch_length + 1, *tensor.shape[1:])[:, 1:]
    return aligned.reshape(trajectory_length, *tensor.shape[1:])


def project_dsrl_trajectory(
    trajectory: Trajectory,
    *,
    action_horizon: int,
    latent_dim: int,
    state_dim: int,
    num_action_chunks: int,
    gamma: float,
) -> dict[str, Any]:
    """Project a rollout trajectory into compact DSRL macro transitions."""
    if trajectory.rewards is None or trajectory.actions is None:
        raise ValueError("DSRL projection requires rewards and latent actions")
    trajectory_length, batch_size = trajectory.rewards.shape[:2]
    if trajectory.actions.shape[:2] != (trajectory_length, batch_size):
        raise ValueError(
            "DSRL action/reward prefix mismatch: "
            f"actions={tuple(trajectory.actions.shape)}, "
            f"rewards={tuple(trajectory.rewards.shape)}"
        )
    for obs_name, obs in (
        ("curr_obs", trajectory.curr_obs),
        ("next_obs", trajectory.next_obs),
    ):
        if "main_images" not in obs or "states" not in obs:
            raise ValueError(
                f"{obs_name} must contain main_images and states, got {tuple(obs)}"
            )
        for key in ("main_images", "states"):
            if obs[key].shape[:2] != (trajectory_length, batch_size):
                raise ValueError(
                    f"{obs_name}.{key} prefix mismatch: {tuple(obs[key].shape)}"
                )

    if trajectory.terminations is None or trajectory.truncations is None:
        raise ValueError("DSRL projection requires termination and truncation fields")
    terminations = _align_bootstrap_field(
        trajectory.terminations, trajectory_length, "terminations"
    )
    truncations = _align_bootstrap_field(
        trajectory.truncations, trajectory_length, "truncations"
    )
    success = (
        terminations.to(torch.bool)
        .reshape(trajectory_length, batch_size, -1)
        .any(dim=-1)
    )
    truncated = (
        truncations.to(torch.bool)
        .reshape(trajectory_length, batch_size, -1)
        .any(dim=-1)
    )
    done = success | truncated

    keep = torch.ones_like(done, dtype=torch.bool)
    for env_idx in range(batch_size):
        done_indices = torch.nonzero(done[:, env_idx], as_tuple=False).flatten()
        if done_indices.numel() > 0:
            first_done = int(done_indices[0])
            keep[first_done + 1 :, env_idx] = False
    if not keep.any():
        raise ValueError("DSRL projection produced no valid transition")
    num_valid = int(keep.sum().item())

    selected_actions = trajectory.actions[keep]
    if selected_actions.ndim == 2:
        if selected_actions.shape[-1] != action_horizon * latent_dim:
            raise ValueError(
                f"Flattened DSRL latent has wrong size: {tuple(selected_actions.shape)}"
            )
        selected_actions = selected_actions.reshape(-1, action_horizon, latent_dim)
    elif selected_actions.ndim != 3 or selected_actions.shape[1:] != (
        action_horizon,
        latent_dim,
    ):
        raise ValueError(
            "DSRL latent must be [N,H,L] or [N,H*L], "
            f"got {tuple(selected_actions.shape)}"
        )
    repeated_latent = selected_actions[:, :1].expand_as(selected_actions)
    if not torch.equal(selected_actions, repeated_latent):
        max_error = (
            (selected_actions.float() - repeated_latent.float()).abs().max().item()
        )
        raise ValueError(
            "DSRL latent is not exactly repeated across action_horizon; "
            f"max_error={max_error}"
        )
    latent = selected_actions[:, 0].to(torch.bfloat16).cpu().contiguous()

    curr_states = trajectory.curr_obs["states"][keep].reshape(num_valid, -1)
    next_states = trajectory.next_obs["states"][keep].reshape(num_valid, -1)
    if curr_states.shape[-1] != state_dim or next_states.shape[-1] != state_dim:
        raise ValueError(
            "DSRL state dimension mismatch: "
            f"curr={curr_states.shape[-1]}, next={next_states.shape[-1]}, "
            f"expected={state_dim}"
        )

    # Lazy import keeps the generic replay package usable without OpenPI.
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        preprocess_dsrl_images,
    )

    curr_images = preprocess_dsrl_images(trajectory.curr_obs["main_images"][keep])[:, 0]
    next_images = preprocess_dsrl_images(trajectory.next_obs["main_images"][keep])[:, 0]
    selected_success = success[keep]
    selected_truncated = truncated[keep]
    num_transitions = int(selected_success.numel())
    discount = float(gamma) ** int(num_action_chunks)

    return {
        "curr_obs": {
            "main_images": curr_images.to(torch.bfloat16).cpu().contiguous(),
            "states": curr_states.float().cpu().contiguous(),
        },
        "next_obs": {
            "main_images": next_images.to(torch.bfloat16).cpu().contiguous(),
            "states": next_states.float().cpu().contiguous(),
        },
        "actions": latent,
        "rewards": torch.where(
            selected_success,
            torch.zeros_like(selected_success, dtype=torch.float32),
            -torch.ones_like(selected_success, dtype=torch.float32),
        )
        .unsqueeze(-1)
        .cpu()
        .contiguous(),
        "continuations": (~selected_success).unsqueeze(-1).cpu().contiguous(),
        "terminations": selected_success.unsqueeze(-1).cpu().contiguous(),
        "truncations": selected_truncated.unsqueeze(-1).cpu().contiguous(),
        "discounts": torch.full((num_transitions, 1), discount, dtype=torch.float32),
    }


def _tree_map(fn, tree):
    if isinstance(tree, torch.Tensor):
        return fn(tree)
    if isinstance(tree, dict):
        return {key: _tree_map(fn, value) for key, value in tree.items()}
    raise TypeError(f"DSRL replay only supports tensors and dicts, got {type(tree)}")


def _tree_leaves(tree):
    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _tree_leaves(value)
    else:
        raise TypeError(
            f"DSRL replay only supports tensors and dicts, got {type(tree)}"
        )


def _assert_same_structure(left, right, path="batch"):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape[1:] != right.shape[1:] or left.dtype != right.dtype:
            raise ValueError(
                f"{path} schema mismatch: "
                f"{tuple(left.shape[1:])}/{left.dtype} != "
                f"{tuple(right.shape[1:])}/{right.dtype}"
            )
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError(f"{path} keys mismatch: {sorted(left)} != {sorted(right)}")
        for key in left:
            _assert_same_structure(left[key], right[key], f"{path}.{key}")
        return
    raise ValueError(f"{path} structure mismatch")


class DSRLTransitionReplayBuffer:
    """Fixed-capacity CPU ring for compact DSRL macro transitions."""

    CHECKPOINT_FILE = "dsrl_transition_replay.pt"

    def __init__(
        self,
        *,
        capacity: int,
        seed: int,
        rank: int,
        world_size: int,
        schema_version: int = 1,
    ):
        if capacity <= 0 or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(
                f"Invalid DSRL replay layout: {capacity=}, {rank=}, {world_size=}"
            )
        self.global_capacity = int(capacity)
        self.rank = int(rank)
        self.world_size = int(world_size)
        base, remainder = divmod(self.global_capacity, self.world_size)
        self.capacity = base + int(self.rank < remainder)
        if self.capacity <= 0:
            raise ValueError("Global replay capacity is smaller than actor world size")
        self.schema_version = int(schema_version)
        self.seed = int(seed)
        self.random_generator = torch.Generator(device="cpu")
        self.random_generator.manual_seed(self.seed + self.rank)
        self._storage: dict[str, Any] | None = None
        self.write_cursor = 0
        self.resident_size = 0
        self.total_inserted = 0

    def _prepare_batch(self, batch: dict[str, Any]) -> tuple[dict[str, Any], int]:
        required = {
            "curr_obs",
            "next_obs",
            "actions",
            "rewards",
            "continuations",
            "terminations",
            "truncations",
            "discounts",
        }
        if set(batch) != required:
            raise ValueError(
                f"DSRL replay keys mismatch: {sorted(batch)} != {sorted(required)}"
            )
        prepared = _tree_map(lambda x: x.detach().cpu().contiguous(), batch)
        leaves = list(_tree_leaves(prepared))
        if not leaves:
            raise ValueError("Cannot add an empty DSRL transition tree")
        batch_size = int(leaves[0].shape[0])
        if batch_size <= 0 or any(int(leaf.shape[0]) != batch_size for leaf in leaves):
            raise ValueError("All DSRL replay leaves must share a non-empty batch dim")
        return prepared, batch_size

    def add_batch(self, batch: dict[str, Any]) -> int:
        prepared, original_batch_size = self._prepare_batch(batch)
        if original_batch_size > self.capacity:
            prepared = _tree_map(lambda x: x[-self.capacity :], prepared)
        batch_size = min(original_batch_size, self.capacity)

        if self._storage is None:
            self._storage = _tree_map(
                lambda x: torch.zeros(
                    (self.capacity, *x.shape[1:]), dtype=x.dtype, device="cpu"
                ),
                prepared,
            )
        else:
            _assert_same_structure(self._storage, prepared)

        indices = (
            torch.arange(batch_size, dtype=torch.long) + self.write_cursor
        ) % self.capacity

        def copy_into(storage, values):
            if isinstance(storage, torch.Tensor):
                storage.index_copy_(0, indices, values)
                return
            for key in storage:
                copy_into(storage[key], values[key])

        copy_into(self._storage, prepared)
        self.write_cursor = (self.write_cursor + batch_size) % self.capacity
        self.resident_size = min(self.capacity, self.resident_size + batch_size)
        self.total_inserted += original_batch_size
        return original_batch_size

    def sample(self, num_chunks: int) -> dict[str, Any]:
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks}")
        if self._storage is None or self.resident_size == 0:
            raise RuntimeError("Cannot sample from an empty DSRL replay")
        indices = torch.randint(
            0,
            self.resident_size,
            (num_chunks,),
            generator=self.random_generator,
            device="cpu",
        )
        return _tree_map(lambda x: x.index_select(0, indices), self._storage)

    def __len__(self) -> int:
        return self.resident_size

    @property
    def total_samples(self) -> int:
        return self.resident_size

    def is_ready(self, min_size: int) -> bool:
        return self.resident_size >= int(min_size)

    async def is_ready_async(self, min_size: int) -> bool:
        return self.is_ready(min_size)

    def get_stats(self) -> dict[str, float]:
        return {
            "resident_transitions": self.resident_size,
            "total_inserted": self.total_inserted,
            "local_capacity": self.capacity,
            "write_cursor": self.write_cursor,
        }

    def save_checkpoint(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        checkpoint = {
            "schema_version": self.schema_version,
            "global_capacity": self.global_capacity,
            "local_capacity": self.capacity,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "write_cursor": self.write_cursor,
            "resident_size": self.resident_size,
            "total_inserted": self.total_inserted,
            "rng_state": self.random_generator.get_state(),
            "storage": self._storage,
        }
        target_path = os.path.join(save_path, self.CHECKPOINT_FILE)
        temp_path = f"{target_path}.tmp"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, target_path)

    def load_checkpoint(self, load_path: str):
        checkpoint_path = os.path.join(load_path, self.CHECKPOINT_FILE)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"DSRL replay checkpoint not found: {checkpoint_path}"
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        expected = {
            "schema_version": self.schema_version,
            "global_capacity": self.global_capacity,
            "local_capacity": self.capacity,
            "rank": self.rank,
            "world_size": self.world_size,
        }
        mismatches = {
            key: (checkpoint.get(key), value)
            for key, value in expected.items()
            if checkpoint.get(key) != value
        }
        if mismatches:
            raise ValueError(f"DSRL replay checkpoint layout mismatch: {mismatches}")
        storage = checkpoint.get("storage")
        if storage is not None:
            for leaf in _tree_leaves(storage):
                if int(leaf.shape[0]) != self.capacity or leaf.device.type != "cpu":
                    raise ValueError(
                        "Invalid DSRL replay storage leaf: "
                        f"shape={tuple(leaf.shape)}, device={leaf.device}"
                    )
            storage = _tree_map(lambda x: x.contiguous(), storage)
        resident_size = int(checkpoint["resident_size"])
        write_cursor = int(checkpoint["write_cursor"])
        total_inserted = int(checkpoint["total_inserted"])
        if (
            not 0 <= resident_size <= self.capacity
            or not 0 <= write_cursor < self.capacity
            or total_inserted < resident_size
        ):
            raise ValueError(
                "Invalid DSRL replay counters: "
                f"{resident_size=}, {write_cursor=}, {total_inserted=}"
            )
        if resident_size > 0 and storage is None:
            raise ValueError("Non-empty DSRL replay checkpoint has no storage")
        self._storage = storage
        self.resident_size = resident_size
        self.write_cursor = write_cursor
        self.total_inserted = total_inserted
        self.seed = int(checkpoint.get("seed", self.seed))
        self.random_generator.set_state(checkpoint["rng_state"])

    def clear(self):
        self._storage = None
        self.write_cursor = 0
        self.resident_size = 0
        self.total_inserted = 0

    def close(self, wait: bool = True):
        del wait
