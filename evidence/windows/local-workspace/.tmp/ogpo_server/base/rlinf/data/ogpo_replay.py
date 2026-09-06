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

"""Bounded CPU replay for OGPO primitive RoboTwin transitions."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor

TensorObservation = dict[str, Tensor]


def _clone_observation(observation: Mapping[str, Tensor]) -> TensorObservation:
    if not observation:
        raise ValueError("observation must contain at least one tensor")
    cloned: TensorObservation = {}
    for name, value in observation.items():
        if not isinstance(name, str) or not name:
            raise ValueError("observation keys must be non-empty strings")
        if not isinstance(value, Tensor):
            raise TypeError(f"observation[{name!r}] must be a tensor")
        cloned[name] = value.detach().clone().cpu().contiguous()
    return cloned


def _validate_observation_pair(
    observation: Mapping[str, Tensor],
    next_observation: Mapping[str, Tensor],
) -> None:
    if observation.keys() != next_observation.keys():
        raise ValueError("observation and next_observation keys must match")
    for name in observation:
        current = observation[name]
        following = next_observation[name]
        if current.shape != following.shape or current.dtype != following.dtype:
            raise ValueError(
                f"observation field {name!r} changes shape or dtype: "
                f"{tuple(current.shape)}/{current.dtype} -> "
                f"{tuple(following.shape)}/{following.dtype}"
            )


def _stack_observations(
    observations: list[Mapping[str, Tensor]],
) -> TensorObservation:
    if not observations:
        raise ValueError("cannot stack an empty observation list")
    keys = observations[0].keys()
    if any(observation.keys() != keys for observation in observations[1:]):
        raise ValueError("all sampled observations must have identical keys")
    return {
        name: torch.stack([observation[name] for observation in observations])
        for name in keys
    }


@dataclass(frozen=True)
class OGPOPrimitiveRow:
    """One primitive transition stored by the actor-side replay.

    Observations are flat dictionaries of CPU-storable tensors. Prompt bytes
    and prompt length can therefore travel through the same structure as image
    and proprioception tensors without storing Python strings.
    """

    observation: TensorObservation
    next_observation: TensorObservation
    action_model: Tensor
    action: Tensor
    reward: float
    terminated: bool
    truncated: bool
    episode_id: int
    step_id: int
    policy_version: int = 0
    source_fingerprint: str = ""
    row_id: int | None = None


@dataclass(frozen=True)
class OGPOSequenceBatch:
    """Random replay sequences padded on the right to replay horizon C."""

    observation: TensorObservation
    next_observation: TensorObservation
    action_model: Tensor
    action: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    valid: Tensor
    h: Tensor
    bootstrap_mask: Tensor
    row_ids: Tensor
    episode_ids: Tensor
    start_step_ids: Tensor


@dataclass(frozen=True)
class OGPOSuccessBatch:
    """Full-length action sequences drawn only from successful episodes."""

    observation: TensorObservation
    action_model: Tensor
    action: Tensor
    row_ids: Tensor
    episode_ids: Tensor
    start_step_ids: Tensor


def _row_to_state(row: OGPOPrimitiveRow) -> dict[str, Any]:
    return {
        "observation": _clone_observation(row.observation),
        "next_observation": _clone_observation(row.next_observation),
        "action_model": row.action_model.detach().clone().cpu(),
        "action": row.action.detach().clone().cpu(),
        "reward": float(row.reward),
        "terminated": bool(row.terminated),
        "truncated": bool(row.truncated),
        "episode_id": int(row.episode_id),
        "step_id": int(row.step_id),
        "policy_version": int(row.policy_version),
        "source_fingerprint": str(row.source_fingerprint),
        "row_id": row.row_id,
    }


def _row_from_state(state: Mapping[str, Any]) -> OGPOPrimitiveRow:
    return OGPOPrimitiveRow(**dict(state))


class OGPOReplayBuffer:
    """A simple rank-local ring with sequence and success-row views.

    The success view stores row IDs only. Evicting a row from the bounded ring
    also removes that ID from the view, so image tensors are never duplicated
    merely because an episode succeeded.
    """

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        capacity: int,
        max_sequence_length: int = 10,
        action_dim: int = 14,
        model_action_dim: int = 32,
        seed: int = 0,
    ) -> None:
        for name, value in {
            "capacity": capacity,
            "max_sequence_length": max_sequence_length,
            "action_dim": action_dim,
            "model_action_dim": model_action_dim,
        }.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")

        self.capacity = int(capacity)
        self.max_sequence_length = int(max_sequence_length)
        self.action_dim = int(action_dim)
        self.model_action_dim = int(model_action_dim)
        self.seed = int(seed)

        self._slots: list[OGPOPrimitiveRow | None] = [None] * self.capacity
        self._cursor = 0
        self._size = 0
        self._total_inserted = 0
        self._next_row_id = 0
        self._row_id_to_slot: dict[int, int] = {}
        self._episode_step_to_row_id: dict[tuple[int, int], int] = {}
        self._success_row_ids: set[int] = set()
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)

    def __len__(self) -> int:
        return self._size

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def total_inserted(self) -> int:
        return self._total_inserted

    @property
    def success_size(self) -> int:
        return len(self._success_row_ids)

    @property
    def schema_version(self) -> int:
        return self._SCHEMA_VERSION

    @property
    def success_row_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._success_row_ids))

    def _normalize_new_row(self, row: OGPOPrimitiveRow) -> OGPOPrimitiveRow:
        if row.row_id is not None:
            raise ValueError("row_id is assigned by OGPOReplayBuffer.add")
        if int(row.step_id) < 0:
            raise ValueError("step_id must be non-negative")
        if int(row.policy_version) < 0:
            raise ValueError("policy_version must be non-negative")
        reward = float(row.reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")

        observation = _clone_observation(row.observation)
        next_observation = _clone_observation(row.next_observation)
        _validate_observation_pair(observation, next_observation)

        if not isinstance(row.action_model, Tensor) or not isinstance(
            row.action, Tensor
        ):
            raise TypeError("action_model and action must be tensors")
        action_model = row.action_model.detach().clone().cpu().float().contiguous()
        action = row.action.detach().clone().cpu().float().contiguous()
        if action_model.shape != (self.model_action_dim,):
            raise ValueError(
                "action_model must have shape "
                f"[{self.model_action_dim}], got {tuple(action_model.shape)}"
            )
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"action must have shape [{self.action_dim}], "
                f"got {tuple(action.shape)}"
            )
        if not torch.isfinite(action_model).all() or not torch.isfinite(action).all():
            raise ValueError("actions must be finite")

        return OGPOPrimitiveRow(
            observation=observation,
            next_observation=next_observation,
            action_model=action_model,
            action=action,
            reward=reward,
            terminated=bool(row.terminated),
            truncated=bool(row.truncated),
            episode_id=int(row.episode_id),
            step_id=int(row.step_id),
            policy_version=int(row.policy_version),
            source_fingerprint=str(row.source_fingerprint),
        )

    def add(self, row: OGPOPrimitiveRow) -> int:
        """Insert one primitive row and return its monotonically increasing ID."""
        normalized = self._normalize_new_row(row)
        episode_step = (normalized.episode_id, normalized.step_id)
        if episode_step in self._episode_step_to_row_id:
            raise ValueError(f"duplicate live episode/step row: {episode_step}")

        overwritten = self._slots[self._cursor]
        if overwritten is not None:
            assert overwritten.row_id is not None
            del self._row_id_to_slot[overwritten.row_id]
            del self._episode_step_to_row_id[
                (overwritten.episode_id, overwritten.step_id)
            ]
            self._success_row_ids.discard(overwritten.row_id)

        row_id = self._next_row_id
        stored = replace(normalized, row_id=row_id)
        self._slots[self._cursor] = stored
        self._row_id_to_slot[row_id] = self._cursor
        self._episode_step_to_row_id[episode_step] = row_id

        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._total_inserted += 1
        self._next_row_id += 1
        return row_id

    def add_episode(
        self,
        rows: Iterable[OGPOPrimitiveRow],
        *,
        success: bool = False,
    ) -> list[int]:
        """Insert a completed episode and optionally expose it to success BC."""
        rows = list(rows)
        if not rows:
            return []
        episode_ids = {int(row.episode_id) for row in rows}
        if len(episode_ids) != 1:
            raise ValueError("add_episode rows must share one episode_id")
        ordered_steps = [int(row.step_id) for row in rows]
        if ordered_steps != list(range(ordered_steps[0], ordered_steps[0] + len(rows))):
            raise ValueError("add_episode rows must have consecutive step_id values")

        row_ids = [self.add(row) for row in rows]
        if success:
            self.mark_episode_success(next(iter(episode_ids)))
        return row_ids

    def mark_episode_success(self, episode_id: int) -> tuple[int, ...]:
        """Add live rows from one successful episode to the ID-only view."""
        row_ids = []
        for row in self._slots:
            if row is None or row.episode_id != int(episode_id):
                continue
            assert row.row_id is not None
            row_ids.append(row.row_id)
        row_ids = tuple(sorted(row_ids))
        if not row_ids:
            raise KeyError(f"episode {episode_id} has no live replay rows")
        self._success_row_ids.update(row_ids)
        return row_ids

    def get_row(self, row_id: int) -> OGPOPrimitiveRow:
        """Return a detached CPU copy of one live row."""
        try:
            row = self._slots[self._row_id_to_slot[int(row_id)]]
        except KeyError as error:
            raise KeyError(f"row_id {row_id} is not live") from error
        assert row is not None
        return _row_from_state(_row_to_state(row))

    def _row_for_episode_step(
        self,
        episode_id: int,
        step_id: int,
    ) -> OGPOPrimitiveRow | None:
        row_id = self._episode_step_to_row_id.get((episode_id, step_id))
        if row_id is None:
            return None
        row = self._slots[self._row_id_to_slot[row_id]]
        assert row is not None
        return row

    def _collect_sequence(
        self,
        start_row_id: int,
        requested_h: int,
    ) -> list[OGPOPrimitiveRow]:
        start = self._slots[self._row_id_to_slot[start_row_id]]
        assert start is not None
        sequence: list[OGPOPrimitiveRow] = []
        for offset in range(requested_h):
            row = self._row_for_episode_step(
                start.episode_id,
                start.step_id + offset,
            )
            if row is None:
                break
            sequence.append(row)
            if row.terminated or row.truncated:
                break
        if not sequence:
            raise RuntimeError("live start row unexpectedly produced an empty sequence")
        return sequence

    def _pack_sequences(
        self,
        sequences: list[list[OGPOPrimitiveRow]],
    ) -> OGPOSequenceBatch:
        batch_size = len(sequences)
        horizon = self.max_sequence_length
        action_model = torch.zeros(
            batch_size, horizon, self.model_action_dim, dtype=torch.float32
        )
        action = torch.zeros(batch_size, horizon, self.action_dim, dtype=torch.float32)
        rewards = torch.zeros(batch_size, horizon, dtype=torch.float32)
        terminated = torch.zeros(batch_size, horizon, dtype=torch.bool)
        truncated = torch.zeros(batch_size, horizon, dtype=torch.bool)
        valid = torch.zeros(batch_size, horizon, dtype=torch.bool)
        row_ids = torch.full((batch_size, horizon), -1, dtype=torch.long)

        starts = []
        next_observations = []
        lengths = []
        episode_ids = []
        start_step_ids = []
        bootstrap_masks = []
        for batch_index, sequence in enumerate(sequences):
            length = len(sequence)
            if not 1 <= length <= horizon:
                raise ValueError(f"sequence length must be in [1, {horizon}]")
            for sequence_index, row in enumerate(sequence):
                action_model[batch_index, sequence_index].copy_(row.action_model)
                action[batch_index, sequence_index].copy_(row.action)
                rewards[batch_index, sequence_index] = row.reward
                terminated[batch_index, sequence_index] = row.terminated
                truncated[batch_index, sequence_index] = row.truncated
                valid[batch_index, sequence_index] = True
                assert row.row_id is not None
                row_ids[batch_index, sequence_index] = row.row_id

            first = sequence[0]
            final = sequence[-1]
            starts.append(first.observation)
            next_observations.append(final.next_observation)
            lengths.append(length)
            episode_ids.append(first.episode_id)
            start_step_ids.append(first.step_id)
            # Time-limit truncation preserves a real next state and bootstraps.
            # A true termination disables bootstrap even if both flags are set.
            bootstrap_masks.append(0.0 if final.terminated else 1.0)

        return OGPOSequenceBatch(
            observation=_stack_observations(starts),
            next_observation=_stack_observations(next_observations),
            action_model=action_model,
            action=action,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            valid=valid,
            h=torch.tensor(lengths, dtype=torch.long),
            bootstrap_mask=torch.tensor(bootstrap_masks, dtype=torch.float32),
            row_ids=row_ids,
            episode_ids=torch.tensor(episode_ids, dtype=torch.long),
            start_step_ids=torch.tensor(start_step_ids, dtype=torch.long),
        )

    def sample_sequences(self, batch_size: int) -> OGPOSequenceBatch:
        """Sample random starts and take the longest available prefix up to C."""
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        live_row_ids = sorted(self._row_id_to_slot)
        if not live_row_ids:
            raise RuntimeError("cannot sample from an empty OGPO replay")

        start_indices = torch.randint(
            len(live_row_ids),
            (int(batch_size),),
            generator=self._rng,
        ).tolist()
        sequences = [
            self._collect_sequence(live_row_ids[index], self.max_sequence_length)
            for index in start_indices
        ]
        return self._pack_sequences(sequences)

    def sample_success_sequences(
        self,
        batch_size: int,
    ) -> OGPOSuccessBatch | None:
        """Sample C-step BC targets, or return None when no full target exists."""
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if not self._success_row_ids:
            return None

        candidates: list[list[OGPOPrimitiveRow]] = []
        for row_id in sorted(self._success_row_ids):
            if row_id not in self._row_id_to_slot:
                continue
            sequence = self._collect_sequence(row_id, self.max_sequence_length)
            if len(sequence) != self.max_sequence_length:
                continue
            if any(
                row.row_id not in self._success_row_ids for row in sequence
            ):
                continue
            candidates.append(sequence)
        if not candidates:
            return None

        indices = torch.randint(
            len(candidates),
            (int(batch_size),),
            generator=self._rng,
        ).tolist()
        selected = [candidates[index] for index in indices]
        packed = self._pack_sequences(selected)
        return OGPOSuccessBatch(
            observation=packed.observation,
            action_model=packed.action_model,
            action=packed.action,
            row_ids=packed.row_ids,
            episode_ids=packed.episode_ids,
            start_step_ids=packed.start_step_ids,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return an exact CPU snapshot including ring and sampling RNG state."""
        return {
            "schema_version": self._SCHEMA_VERSION,
            "metadata": {
                "capacity": self.capacity,
                "max_sequence_length": self.max_sequence_length,
                "action_dim": self.action_dim,
                "model_action_dim": self.model_action_dim,
                "seed": self.seed,
            },
            "cursor": self._cursor,
            "size": self._size,
            "total_inserted": self._total_inserted,
            "next_row_id": self._next_row_id,
            "rng_state": self._rng.get_state().clone(),
            "slots": [
                _row_to_state(row) if row is not None else None
                for row in self._slots
            ],
            "success_row_ids": sorted(self._success_row_ids),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a snapshot after validating its local replay contract."""
        if state.get("schema_version") != self._SCHEMA_VERSION:
            raise ValueError("unsupported OGPO replay schema version")
        expected_metadata = {
            "capacity": self.capacity,
            "max_sequence_length": self.max_sequence_length,
            "action_dim": self.action_dim,
            "model_action_dim": self.model_action_dim,
            "seed": self.seed,
        }
        if state.get("metadata") != expected_metadata:
            raise ValueError(
                "OGPO replay metadata mismatch: "
                f"expected {expected_metadata}, got {state.get('metadata')}"
            )

        raw_slots = state.get("slots")
        if not isinstance(raw_slots, list) or len(raw_slots) != self.capacity:
            raise ValueError("snapshot slots do not match replay capacity")
        slots = [
            _row_from_state(row_state) if row_state is not None else None
            for row_state in raw_slots
        ]
        size = int(state["size"])
        cursor = int(state["cursor"])
        total_inserted = int(state["total_inserted"])
        next_row_id = int(state["next_row_id"])
        if not 0 <= size <= self.capacity:
            raise ValueError("snapshot size is out of range")
        if sum(row is not None for row in slots) != size:
            raise ValueError("snapshot size does not match live slots")
        if not 0 <= cursor < self.capacity:
            raise ValueError("snapshot cursor is out of range")
        if total_inserted < size or next_row_id < total_inserted:
            raise ValueError("snapshot lifetime counters are inconsistent")

        row_id_to_slot: dict[int, int] = {}
        episode_step_to_row_id: dict[tuple[int, int], int] = {}
        for slot, row in enumerate(slots):
            if row is None:
                continue
            if row.row_id is None:
                raise ValueError("snapshot row is missing row_id")
            normalized = self._normalize_new_row(replace(row, row_id=None))
            row = replace(normalized, row_id=int(row.row_id))
            slots[slot] = row
            if row.row_id in row_id_to_slot:
                raise ValueError("snapshot contains duplicate row_id")
            episode_step = (row.episode_id, row.step_id)
            if episode_step in episode_step_to_row_id:
                raise ValueError("snapshot contains duplicate episode/step")
            row_id_to_slot[row.row_id] = slot
            episode_step_to_row_id[episode_step] = row.row_id

        success_row_ids = {int(row_id) for row_id in state["success_row_ids"]}
        if not success_row_ids.issubset(row_id_to_slot):
            raise ValueError("success view references evicted or missing rows")

        restored_rng = torch.Generator(device="cpu")
        restored_rng.set_state(state["rng_state"])
        self._slots = slots
        self._cursor = cursor
        self._size = size
        self._total_inserted = total_inserted
        self._next_row_id = next_row_id
        self._row_id_to_slot = row_id_to_slot
        self._episode_step_to_row_id = episode_step_to_row_id
        self._success_row_ids = success_row_ids
        self._rng = restored_rng

    def save_snapshot(self, path: str | Path) -> None:
        """Save a trusted local replay snapshot."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), destination)

    def load_snapshot(self, path: str | Path) -> None:
        """Load a trusted local replay snapshot onto CPU."""
        try:
            state = torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(Path(path), map_location="cpu")
        self.load_state_dict(state)
