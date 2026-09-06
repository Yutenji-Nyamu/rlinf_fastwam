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

"""Rank-local bounded replay for fixed-N QAM macro transitions."""

import hashlib
import os
import uuid
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from rlinf.algorithms.qam.contracts import (
    QAM_REPLAY_SCHEMA_VERSION,
    QAMMacroTransition,
    QAMPolicyObservation,
    fixed_slot_discounted_return,
)


def _clone_tensor(tensor: Tensor | None) -> Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().clone().cpu().contiguous()


def _clone_observation(observation: QAMPolicyObservation) -> QAMPolicyObservation:
    return QAMPolicyObservation(
        cameras_uint8=_clone_tensor(observation.cameras_uint8),
        proprio=_clone_tensor(observation.proprio),
        prompt=observation.prompt,
        task_id=observation.task_id,
        transform_fingerprint=observation.transform_fingerprint,
    )


def _clone_transition(transition: QAMMacroTransition) -> QAMMacroTransition:
    values: dict[str, Any] = {}
    for field in fields(QAMMacroTransition):
        value = getattr(transition, field.name)
        values[field.name] = (
            _clone_tensor(value) if isinstance(value, Tensor) else value
        )
    return QAMMacroTransition(**values)


def _observation_to_state(observation: QAMPolicyObservation) -> dict[str, Any]:
    return {
        "cameras_uint8": _clone_tensor(observation.cameras_uint8),
        "proprio": _clone_tensor(observation.proprio),
        "prompt": observation.prompt,
        "task_id": observation.task_id,
        "transform_fingerprint": observation.transform_fingerprint,
    }


def _observation_from_state(state: dict[str, Any]) -> QAMPolicyObservation:
    return QAMPolicyObservation(**state)


def _transition_to_state(transition: QAMMacroTransition) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for field in fields(QAMMacroTransition):
        value = getattr(transition, field.name)
        state[field.name] = _clone_tensor(value) if isinstance(value, Tensor) else value
    return state


def _transition_from_state(state: dict[str, Any]) -> QAMMacroTransition:
    return QAMMacroTransition(**state)


@dataclass(frozen=True)
class QAMReplaySample:
    """A sampled transition plus its deduplicated raw policy views."""

    transition: QAMMacroTransition
    observation: QAMPolicyObservation
    next_observation: QAMPolicyObservation | None


class QAMTransitionReplay:
    """Capacity-bounded rank-local ring with exact save/resume state."""

    def __init__(
        self,
        *,
        capacity: int,
        rank: int,
        world_size: int,
        seed: int,
        gamma_slot: float,
        contract_fingerprint: str,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError(
                f"invalid rank/world_size combination: {rank}/{world_size}"
            )
        if not contract_fingerprint:
            raise ValueError("contract_fingerprint must be non-empty")
        if not 0.0 <= gamma_slot <= 1.0:
            raise ValueError("gamma_slot must be in [0, 1]")

        self.capacity = int(capacity)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.gamma_slot = float(gamma_slot)
        self.contract_fingerprint = contract_fingerprint

        self._slots: list[QAMMacroTransition | None] = [None] * self.capacity
        self._cursor = 0
        self._size = 0
        self._total_inserted = 0
        self._observations: dict[str, QAMPolicyObservation] = {}
        self._observation_refcounts: dict[str, int] = {}
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)

    @property
    def cursor(self) -> int:
        """Return the next physical insertion slot."""
        return self._cursor

    @property
    def total_inserted(self) -> int:
        """Return lifetime schema-valid inserts for UTD accounting."""
        return self._total_inserted

    @property
    def observation_store_size(self) -> int:
        """Return the number of referenced canonical raw observations."""
        return len(self._observations)

    @property
    def rank_world_fingerprint(self) -> str:
        """Return a stable replay ownership fingerprint."""
        material = (
            f"qam-replay-v{QAM_REPLAY_SCHEMA_VERSION}|"
            f"rank={self.rank}|world={self.world_size}|"
            f"contract={self.contract_fingerprint}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return self._size

    def _increment_observation(
        self,
        observation_id: str,
        observation: QAMPolicyObservation,
    ) -> None:
        if observation_id != observation.content_id():
            raise ValueError("observation ID does not match canonical content")
        if observation_id not in self._observations:
            self._observations[observation_id] = _clone_observation(observation)
            self._observation_refcounts[observation_id] = 0
        self._observation_refcounts[observation_id] += 1

    def _decrement_observation(self, observation_id: str | None) -> None:
        if observation_id is None:
            return
        remaining = self._observation_refcounts[observation_id] - 1
        if remaining == 0:
            del self._observation_refcounts[observation_id]
            del self._observations[observation_id]
        else:
            self._observation_refcounts[observation_id] = remaining

    def _validate_transition(
        self,
        transition: QAMMacroTransition,
        observation: QAMPolicyObservation,
        next_observation: QAMPolicyObservation | None,
    ) -> None:
        if transition.contract_fingerprint != self.contract_fingerprint:
            raise ValueError("transition/replay contract fingerprint mismatch")
        if transition.obs_id != observation.content_id():
            raise ValueError("transition obs_id does not match observation")
        if transition.next_state_valid:
            if next_observation is None:
                raise ValueError("valid next state requires raw next observation")
            if transition.next_obs_id != next_observation.content_id():
                raise ValueError(
                    "transition next_obs_id does not match next observation"
                )
        elif next_observation is not None:
            raise ValueError("invalid next state must not provide next observation")

        expected_return = fixed_slot_discounted_return(
            transition.chunk_rewards_native,
            gamma_slot=self.gamma_slot,
        ).item()
        if abs(expected_return - transition.reward_macro_discounted) > 1e-6:
            raise ValueError(
                "reward_macro_discounted does not match fixed-slot reduction"
            )

    def add(
        self,
        transition: QAMMacroTransition,
        *,
        observation: QAMPolicyObservation,
        next_observation: QAMPolicyObservation | None,
    ) -> None:
        """Insert one schema-valid macro and evict the overwritten ring row."""
        self._validate_transition(transition, observation, next_observation)

        overwritten = self._slots[self._cursor]
        if overwritten is not None:
            self._decrement_observation(overwritten.obs_id)
            self._decrement_observation(overwritten.next_obs_id)

        self._increment_observation(transition.obs_id, observation)
        if next_observation is not None:
            next_obs_id = transition.next_obs_id
            if next_obs_id is None:
                raise ValueError("valid next observation requires next_obs_id")
            self._increment_observation(next_obs_id, next_observation)

        self._slots[self._cursor] = _clone_transition(transition)
        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._total_inserted += 1

    def sample(self, batch_size: int) -> list[QAMReplaySample]:
        """Sample rank-local rows uniformly with replacement."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        live_slots = [slot for slot in self._slots if slot is not None]
        if not live_slots:
            raise RuntimeError("cannot sample from an empty QAM replay")

        indices = torch.randint(
            len(live_slots),
            (batch_size,),
            generator=self._rng,
        ).tolist()
        samples = []
        for index in indices:
            transition = live_slots[index]
            observation = self._observations[transition.obs_id]
            next_observation = (
                self._observations[transition.next_obs_id]
                if transition.next_obs_id is not None
                else None
            )
            samples.append(
                QAMReplaySample(
                    transition=_clone_transition(transition),
                    observation=_clone_observation(observation),
                    next_observation=(
                        _clone_observation(next_observation)
                        if next_observation is not None
                        else None
                    ),
                )
            )
        return samples

    def _checkpoint_state(
        self,
        snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "complete": True,
            "schema_version": QAM_REPLAY_SCHEMA_VERSION,
            "snapshot": snapshot,
            "metadata": {
                "capacity": self.capacity,
                "rank": self.rank,
                "world_size": self.world_size,
                "seed": self.seed,
                "gamma_slot": self.gamma_slot,
                "contract_fingerprint": self.contract_fingerprint,
                "rank_world_fingerprint": self.rank_world_fingerprint,
            },
            "cursor": self._cursor,
            "size": self._size,
            "total_inserted": self._total_inserted,
            "rng_state": self._rng.get_state(),
            "slots": [
                _transition_to_state(slot) if slot is not None else None
                for slot in self._slots
            ],
            "observations": {
                observation_id: _observation_to_state(observation)
                for observation_id, observation in self._observations.items()
            },
            "observation_refcounts": dict(self._observation_refcounts),
        }

    def save_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Atomically replace one trusted local replay checkpoint file."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            with temporary.open("wb") as file:
                torch.save(self._checkpoint_state(snapshot), file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _validate_checkpoint_metadata(self, metadata: dict[str, Any]) -> None:
        expected = {
            "capacity": self.capacity,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "gamma_slot": self.gamma_slot,
            "contract_fingerprint": self.contract_fingerprint,
            "rank_world_fingerprint": self.rank_world_fingerprint,
        }
        if metadata != expected:
            raise ValueError(
                f"QAM replay resume metadata mismatch: expected {expected}, "
                f"got {metadata}"
            )

    def load_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        expected_snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Load a complete replay checkpoint with exact RNG and ring cursor."""
        checkpoint = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("schema_version") != QAM_REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported QAM replay schema version")
        if checkpoint.get("complete") is not True:
            raise ValueError("incomplete QAM replay checkpoint")
        if (
            expected_snapshot is not None
            and checkpoint.get("snapshot") != expected_snapshot
        ):
            raise ValueError("QAM replay checkpoint snapshot mismatch")
        self._validate_checkpoint_metadata(checkpoint["metadata"])

        slots = [
            _transition_from_state(state) if state is not None else None
            for state in checkpoint["slots"]
        ]
        if len(slots) != self.capacity:
            raise ValueError("checkpoint ring capacity does not match metadata")
        observations = {
            observation_id: _observation_from_state(state)
            for observation_id, state in checkpoint["observations"].items()
        }
        for observation_id, observation in observations.items():
            if observation_id != observation.content_id():
                raise ValueError(
                    "checkpoint observation key does not match canonical content"
                )
        refcounts = {
            str(observation_id): int(count)
            for observation_id, count in checkpoint["observation_refcounts"].items()
        }

        computed_refcounts: dict[str, int] = {}
        for transition in slots:
            if transition is None:
                continue
            for observation_id in (transition.obs_id, transition.next_obs_id):
                if observation_id is None:
                    continue
                if observation_id not in observations:
                    raise ValueError(
                        f"checkpoint transition references missing {observation_id}"
                    )
                computed_refcounts[observation_id] = (
                    computed_refcounts.get(observation_id, 0) + 1
                )
            observation = observations[transition.obs_id]
            next_observation = (
                observations[transition.next_obs_id]
                if transition.next_obs_id is not None
                else None
            )
            self._validate_transition(
                transition,
                observation,
                next_observation,
            )
        if computed_refcounts != refcounts:
            raise ValueError("checkpoint observation refcounts are inconsistent")
        if set(observations) != set(refcounts):
            raise ValueError("checkpoint contains unreferenced observations")

        cursor = int(checkpoint["cursor"])
        size = int(checkpoint["size"])
        total_inserted = int(checkpoint["total_inserted"])
        if not 0 <= cursor < self.capacity:
            raise ValueError("checkpoint cursor is out of range")
        if not 0 <= size <= self.capacity:
            raise ValueError("checkpoint size is out of range")
        if sum(slot is not None for slot in slots) != size:
            raise ValueError("checkpoint size does not match live ring rows")
        if total_inserted < size:
            raise ValueError("checkpoint total_inserted is smaller than size")

        restored_rng = torch.Generator(device="cpu")
        restored_rng.set_state(checkpoint["rng_state"])
        self._slots = slots
        self._observations = observations
        self._observation_refcounts = refcounts
        self._cursor = cursor
        self._size = size
        self._total_inserted = total_inserted
        self._rng = restored_rng
