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

"""FSDP worker for Plain-QAM pi0 online adaptation on RoboTwin.

The FSDP policy contains only the frozen B1 behavior route and trainable F1
route.  The C1 critic, target critic, optimizer, and rank-local replay are
worker-owned sidecars and therefore never enter policy state-dict syncing.
"""

import copy
import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.algorithms.qam.contracts import (
    ACTIVE_ACTION_DIM,
    MODEL_ACTION_DIM,
    MODEL_HORIZON,
    PLANNED_HORIZON,
    QAMMacroTransition,
    QAMPolicyObservation,
    decode_qam_prompt,
    fixed_slot_bootstrap_discount,
    fixed_slot_discounted_return,
    macro_bootstrap_mask,
    project_planned_action,
)
from rlinf.algorithms.qam.core import (
    adjoint_matching_step_loss,
    clone_parameter_snapshot,
    ema_from_preupdate_,
    ensemble_critic_mse,
    q_chunk_td_target,
    reverse_behavior_adjoint,
    sample_memoryless_am_path,
    terminal_mean_q_adjoint,
)
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.qam_transition_replay import (
    QAMReplaySample,
    QAMTransitionReplay,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.qam_critic import QAMCriticEnsemble
from rlinf.scheduler import Channel, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.utils import clear_memory, get_rng_state, set_rng_state
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

QAM_PHASES = ("collect", "q_only", "am_on")
_PHASE_ORDER = {phase: index for index, phase in enumerate(QAM_PHASES)}


@dataclass
class QAMUpdateCredit:
    """Fractional global-insert UTD budget shared identically by all ranks."""

    utd_ratio: float
    pending: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.utd_ratio) or self.utd_ratio < 0:
            raise ValueError("utd_ratio must be finite and non-negative")
        if not math.isfinite(self.pending) or self.pending < 0:
            raise ValueError("pending update credit must be finite and non-negative")

    def add_global_inserts(self, count: int) -> None:
        if count < 0:
            raise ValueError("global insert count must be non-negative")
        self.pending += int(count) * self.utd_ratio

    def take(self, maximum: int) -> int:
        if maximum <= 0:
            raise ValueError("maximum updates must be positive")
        updates = min(int(math.floor(self.pending + 1e-12)), int(maximum))
        self.pending -= updates
        return updates


def classify_qam_end(
    *, terminated: bool, truncated: bool
) -> tuple[bool, bool, bool, bool]:
    """Return success/time-limit/other/next-valid for RoboTwin ends."""
    success_terminated = bool(terminated)
    time_limit_truncated = bool(truncated and not terminated)
    other_truncated = False
    # RoboTwin supplies the true query-final observation at a time limit.
    next_state_valid = not success_terminated
    return (
        success_terminated,
        time_limit_truncated,
        other_truncated,
        next_state_valid,
    )


def extract_qam_camera_triplet(
    obs: dict[str, Any],
    *,
    step: int,
    env: int,
) -> torch.Tensor:
    """Extract RoboTwin's main + left/right wrist cameras as ``uint8[3,H,W,3]``.

    The current trajectory payload is not ``main+wrist+extra``.  It contains
    one ``main_images`` camera and two cameras on the leading image axis of
    ``wrist_images``; ``extra_view_images`` may be absent/None.
    """

    def indexed(key: str) -> torch.Tensor:
        value = obs.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"QAM raw observation is missing tensor {key}")
        if step >= value.shape[0] or env >= value.shape[1]:
            raise ValueError(f"{key} does not contain trajectory row ({step}, {env})")
        return value[step, env].detach().clone().cpu().contiguous()

    main = indexed("main_images")
    wrists = indexed("wrist_images")
    if main.ndim != 3 or main.shape[-1] != 3:
        raise ValueError(f"main_images must resolve to HWC, got {tuple(main.shape)}")
    if wrists.ndim != 4 or wrists.shape[0] != 2 or wrists.shape[-1] != 3:
        raise ValueError(
            f"wrist_images must resolve to [2,H,W,3], got {tuple(wrists.shape)}"
        )
    if main.shape != wrists.shape[1:]:
        raise ValueError(
            "main and wrist camera shapes must match, got "
            f"{tuple(main.shape)} and {tuple(wrists.shape[1:])}"
        )
    if main.dtype != torch.uint8 or wrists.dtype != torch.uint8:
        raise ValueError(
            "QAM canonical cameras must remain uint8, got "
            f"{main.dtype} and {wrists.dtype}"
        )
    return torch.stack((main, wrists[0], wrists[1]), dim=0)


def validate_qam_prefix_block_lengths(
    value: torch.Tensor,
    *,
    feature_blocks: int,
    expected: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Validate and optionally compare the four C1 token block boundaries."""
    lengths = tuple(int(item) for item in value.detach().cpu().reshape(-1).tolist())
    if feature_blocks != 4 or len(lengths) != feature_blocks:
        raise ValueError(
            "QAM C1 requires four pooled prefix blocks and four lengths, got "
            f"{feature_blocks} blocks and {len(lengths)} lengths"
        )
    if any(length <= 0 for length in lengths):
        raise ValueError("QAM prefix block lengths must all be positive")
    if expected is not None and lengths != expected:
        raise ValueError(
            f"QAM prefix block lengths changed within a run: {expected} -> {lengths}"
        )
    return lengths


def _phase_transition_is_valid(saved: str, requested: str) -> bool:
    return (
        saved in _PHASE_ORDER
        and requested in _PHASE_ORDER
        and _PHASE_ORDER[requested] >= _PHASE_ORDER[saved]
    )


def _am_is_enabled_for_next_update(
    *,
    configured_phase: str,
    critic_updates: int,
    q_only_updates_before_am: int,
) -> bool:
    """Return whether the next logical update includes adjoint matching."""
    if critic_updates < 0 or q_only_updates_before_am < 0:
        raise ValueError("QAM update counters must be non-negative")
    return configured_phase == "am_on" and critic_updates >= q_only_updates_before_am


def _resume_update_credit(
    *,
    saved_phase: str,
    requested_phase: str,
    saved_pending: float,
    saved_anchor: int | None,
    global_total_inserts: int,
    warmup_global_inserts: int,
) -> tuple[float, int | None]:
    """Restore UTD state without training collect warm-up rows retroactively."""
    if requested_phase == "collect":
        return 0.0, None
    if saved_phase == "collect":
        anchor = (
            int(global_total_inserts)
            if global_total_inserts >= warmup_global_inserts
            else None
        )
        return 0.0, anchor
    return float(saved_pending), (
        int(saved_anchor) if saved_anchor is not None else None
    )


class QAMFSDPPolicy(EmbodiedFSDPActor):
    """Plain-QAM worker with B1+F1+C1 and fixed-N online macro replay."""

    _QAM_CHECKPOINT_SCHEMA_VERSION = 2

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.qam_cfg = cfg.algorithm.qam
        self.phase = str(self.qam_cfg.phase)
        self.runner_global_step = 0
        self.fine_policy_version = 0
        self.critic_updates = 0
        self.fine_updates = 0
        self.local_total_inserts = 0
        self.global_total_inserts = 0
        self.update_credit = QAMUpdateCredit(utd_ratio=float(self.qam_cfg.utd_ratio))
        self.q_only_anchor_global_inserts: int | None = None

        self.replay: QAMTransitionReplay | None = None
        self.contract_fingerprint: str | None = None
        self.prefix_block_lengths: tuple[int, ...] | None = None
        self.critic: QAMCriticEnsemble | None = None
        self.target_critic: QAMCriticEnsemble | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None
        self.critic_feature_dim: int | None = None
        self._last_ingest_metrics: dict[str, float] = {}

    def init_worker(self) -> None:
        """Build only the FSDP F1 policy; critic dimensions remain runtime-led."""
        self.setup_model_and_optimizer()
        self.param_names_need_sync = [
            name
            for name in self.param_names_need_sync
            if name.startswith("qam_fine.") or ".qam_fine." in name
        ]
        if not self.param_names_need_sync:
            raise RuntimeError(
                "QAM F1 produced no trainable sync parameters; check the "
                "OpenPI train_expert_only/use_qam contract"
            )
        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

    def set_global_step(self, global_step: int) -> None:
        """Record runner progress without aliasing it to policy sync version."""
        self.runner_global_step = int(global_step)
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def get_rollout_sync_version(self) -> int:
        """Return the independently monotonic active F1 policy version."""
        return int(self.fine_policy_version)

    def _init_replay(self, contract_fingerprint: str) -> None:
        if self.replay is not None:
            if contract_fingerprint != self.contract_fingerprint:
                raise ValueError("QAM projection contract changed within a run")
            return
        self.contract_fingerprint = str(contract_fingerprint)
        self.replay = QAMTransitionReplay(
            capacity=int(self.qam_cfg.replay_capacity),
            rank=self._rank,
            world_size=self._world_size,
            seed=int(self.cfg.actor.seed) + self._rank,
            gamma_slot=float(self.qam_cfg.gamma_slot),
            contract_fingerprint=self.contract_fingerprint,
        )

    def _resolve_worker_contract(
        self,
        *,
        projection_fingerprint: str,
        prefix_block_lengths: tuple[int, ...],
    ) -> str:
        material = (
            "qam-worker-contract-v1|"
            f"projection={projection_fingerprint}|"
            f"prefix_blocks={','.join(str(v) for v in prefix_block_lengths)}"
        )
        fingerprint = hashlib.sha256(material.encode("utf-8")).hexdigest()
        if self.prefix_block_lengths is None:
            self.prefix_block_lengths = prefix_block_lengths
        elif self.prefix_block_lengths != prefix_block_lengths:
            raise ValueError(
                "QAM prefix block lengths changed within a run: "
                f"{self.prefix_block_lengths} -> {prefix_block_lengths}"
            )
        self._init_replay(fingerprint)
        return fingerprint

    def _init_critic(self, feature_dim: int) -> None:
        feature_dim = int(feature_dim)
        if self.critic is not None:
            if feature_dim != self.critic_feature_dim:
                raise ValueError(
                    "C1 prefix width changed within a run: "
                    f"{self.critic_feature_dim} -> {feature_dim}"
                )
            return

        # Every rank executes the same CPU initialization seed.  A broadcast
        # then makes corresponding logical heads exact even if framework RNG
        # consumption before this point differed by rank.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(self.qam_cfg.critic_init_seed))
            critic = QAMCriticEnsemble(
                feature_dim=feature_dim,
                num_q_heads=int(self.qam_cfg.num_q_heads),
                hidden_dims=tuple(int(v) for v in self.qam_cfg.critic_hidden_dims),
            )
        critic = critic.to(device=self.device, dtype=torch.float32)
        if torch.distributed.is_initialized():
            for parameter in critic.parameters():
                torch.distributed.broadcast(parameter.data, src=0)

        self.critic = critic
        self.target_critic = copy.deepcopy(critic).eval().requires_grad_(False)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(self.qam_cfg.critic_lr),
            betas=(
                float(self.qam_cfg.critic_adam_beta1),
                float(self.qam_cfg.critic_adam_beta2),
            ),
            eps=float(self.qam_cfg.critic_adam_eps),
        )
        self.critic_feature_dim = feature_dim

    @staticmethod
    def _trajectory_end_flag(
        tensor: torch.Tensor | None,
        *,
        step: int,
        env: int,
        trajectory_steps: int,
    ) -> bool:
        if tensor is None:
            return False
        # EmbodiedRolloutResult exports one leading bootstrap slot.
        if tensor.shape[0] != trajectory_steps + 1:
            raise ValueError(
                "QAM requires terminal fields with one leading bootstrap slot; "
                f"got {tensor.shape[0]} for {trajectory_steps} transitions"
            )
        return bool(tensor[step + 1, env].to(torch.bool).reshape(-1).any())

    @staticmethod
    def _row(
        tensor: torch.Tensor,
        *,
        step: int,
        env: int,
    ) -> torch.Tensor:
        return tensor[step, env].detach().clone().cpu().contiguous()

    def _policy_observation(
        self,
        obs: dict[str, Any],
        *,
        step: int,
        env: int,
        transform_fingerprint: str,
        prompt: str,
    ) -> QAMPolicyObservation:
        cameras = extract_qam_camera_triplet(
            obs,
            step=step,
            env=env,
        )
        states = obs.get("states")
        if not isinstance(states, torch.Tensor):
            raise ValueError("QAM raw observation is missing states")
        proprio = self._row(states, step=step, env=env).reshape(-1)
        if proprio.numel() < ACTIVE_ACTION_DIM:
            raise ValueError(
                f"QAM raw state has {proprio.numel()} values, expected at least 14"
            )
        proprio = proprio[:ACTIVE_ACTION_DIM].to(dtype=torch.float32)
        return QAMPolicyObservation(
            cameras_uint8=cameras,
            proprio=proprio,
            prompt=prompt,
            task_id=str(self.cfg.env.train.task_config.task_name),
            transform_fingerprint=transform_fingerprint,
        )

    @staticmethod
    def _observations_to_env_obs(
        observations: list[QAMPolicyObservation],
    ) -> dict[str, Any]:
        if not observations:
            raise ValueError("cannot build QAM conditioning from no observations")
        cameras = torch.stack(
            [observation.cameras_uint8 for observation in observations],
            dim=0,
        )
        return {
            "main_images": cameras[:, 0],
            "wrist_images": cameras[:, 1:3],
            "extra_view_images": None,
            "states": torch.stack(
                [observation.proprio for observation in observations],
                dim=0,
            ),
            "task_descriptions": [observation.prompt for observation in observations],
        }

    @torch.no_grad()
    def _condition_observations(
        self,
        observations: list[QAMPolicyObservation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.model.training
        self.model.eval()
        conditioning = self.model(
            forward_type=ForwardType.QAM_FLOW,
            operation="conditioning",
            env_obs=self._observations_to_env_obs(observations),
        )
        self.model.train(was_training)
        return (
            conditioning["critic_feature"].detach().cpu().contiguous(),
            conditioning["state"][:, :ACTIVE_ACTION_DIM]
            .to(dtype=torch.float32)
            .detach()
            .cpu()
            .contiguous(),
        )

    @staticmethod
    def _fingerprint_at(
        forward_inputs: dict[str, Any],
        *,
        step: int,
        env: int,
    ) -> str:
        tensor = forward_inputs.get("qam_projection_fingerprint")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("QAM rollout is missing projection fingerprint")
        digest = tensor[step, env].detach().cpu().to(torch.uint8).reshape(-1)
        if digest.numel() != 32:
            raise ValueError("QAM projection fingerprint must contain 32 bytes")
        return bytes(digest.tolist()).hex()

    @staticmethod
    def _validate_projection_contract(
        forward_inputs: dict[str, Any],
        *,
        step: int,
        env: int,
    ) -> None:
        tensor = forward_inputs.get("qam_projection_contract")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("QAM rollout is missing projection contract")
        actual = tuple(
            int(value)
            for value in tensor[step, env].detach().cpu().reshape(-1).tolist()
        )
        expected = (
            MODEL_HORIZON,
            PLANNED_HORIZON,
            MODEL_ACTION_DIM,
            ACTIVE_ACTION_DIM,
        )
        if actual != expected:
            raise ValueError(
                f"QAM projection contract mismatch: expected {expected}, got {actual}"
            )

    def _ingest_trajectory(self, trajectory: Trajectory) -> int:
        if trajectory.actions is None or trajectory.rewards is None:
            raise ValueError("QAM online replay requires actions and rewards")
        forward_inputs = trajectory.forward_inputs
        required = {
            "qam_planned_action_normalized",
            "qam_obs_feature",
            "qam_prompt_utf8",
            "qam_prompt_length",
            "qam_prefix_block_lengths",
            "qam_proprio_normalized",
            "qam_projection_contract",
            "qam_projection_fingerprint",
        }
        missing = sorted(required - set(forward_inputs))
        if missing:
            raise ValueError(
                "QAM rollout is missing forward inputs: " + ", ".join(missing)
            )
        if not trajectory.curr_obs or not trajectory.next_obs:
            raise ValueError("QAM online replay requires current and next raw views")

        trajectory_steps, batch_size = trajectory.actions.shape[:2]
        if trajectory_steps <= 0 or batch_size <= 0:
            raise ValueError("QAM trajectory must contain at least one macro row")
        rows: list[dict[str, Any]] = []
        live_next_observations: list[QAMPolicyObservation] = []
        live_row_indices: list[int] = []
        alive = [True] * batch_size

        for step in range(trajectory_steps):
            for env in range(batch_size):
                if not alive[env]:
                    continue
                self._validate_projection_contract(forward_inputs, step=step, env=env)
                fingerprint = self._fingerprint_at(forward_inputs, step=step, env=env)
                feature_row = self._row(
                    forward_inputs["qam_obs_feature"],
                    step=step,
                    env=env,
                )
                block_lengths = validate_qam_prefix_block_lengths(
                    self._row(
                        forward_inputs["qam_prefix_block_lengths"],
                        step=step,
                        env=env,
                    ),
                    feature_blocks=int(feature_row.shape[0]),
                    expected=self.prefix_block_lengths,
                )
                worker_fingerprint = self._resolve_worker_contract(
                    projection_fingerprint=fingerprint,
                    prefix_block_lengths=block_lengths,
                )
                prompt = decode_qam_prompt(
                    self._row(
                        forward_inputs["qam_prompt_utf8"],
                        step=step,
                        env=env,
                    ),
                    self._row(
                        forward_inputs["qam_prompt_length"],
                        step=step,
                        env=env,
                    ),
                )

                terminated = self._trajectory_end_flag(
                    trajectory.terminations,
                    step=step,
                    env=env,
                    trajectory_steps=trajectory_steps,
                )
                truncated = self._trajectory_end_flag(
                    trajectory.truncations,
                    step=step,
                    env=env,
                    trajectory_steps=trajectory_steps,
                )
                (
                    success_terminated,
                    time_limit_truncated,
                    other_truncated,
                    next_state_valid,
                ) = classify_qam_end(
                    terminated=terminated,
                    truncated=truncated,
                )

                observation = self._policy_observation(
                    trajectory.curr_obs,
                    step=step,
                    env=env,
                    transform_fingerprint=worker_fingerprint,
                    prompt=prompt,
                )
                next_observation = None
                if next_state_valid:
                    next_observation = self._policy_observation(
                        trajectory.next_obs,
                        step=step,
                        env=env,
                        transform_fingerprint=worker_fingerprint,
                        prompt=prompt,
                    )

                planned = self._row(
                    forward_inputs["qam_planned_action_normalized"],
                    step=step,
                    env=env,
                ).reshape(PLANNED_HORIZON, ACTIVE_ACTION_DIM)
                planned_env = self._row(
                    trajectory.actions,
                    step=step,
                    env=env,
                ).reshape(PLANNED_HORIZON, ACTIVE_ACTION_DIM)
                rewards = self._row(
                    trajectory.rewards,
                    step=step,
                    env=env,
                ).reshape(-1)
                if rewards.numel() != PLANNED_HORIZON:
                    raise ValueError(
                        "QAM M2 requires one native reward per fixed slot; "
                        f"got {rewards.numel()}"
                    )
                feature = feature_row
                proprio = self._row(
                    forward_inputs["qam_proprio_normalized"],
                    step=step,
                    env=env,
                ).reshape(ACTIVE_ACTION_DIM)
                version = (
                    int(
                        self._row(
                            trajectory.versions,
                            step=step,
                            env=env,
                        )
                        .reshape(-1)[0]
                        .item()
                    )
                    if isinstance(trajectory.versions, torch.Tensor)
                    else self.fine_policy_version
                )
                row = {
                    "observation": observation,
                    "next_observation": next_observation,
                    "feature": feature,
                    "proprio": proprio,
                    "planned": planned,
                    "planned_env": planned_env,
                    "rewards": rewards.to(dtype=torch.float32),
                    "success_terminated": success_terminated,
                    "time_limit_truncated": time_limit_truncated,
                    "other_truncated": other_truncated,
                    "next_state_valid": next_state_valid,
                    "policy_version": version,
                    "episode_id": (
                        f"rank{self._rank}:runner{self.runner_global_step}:env{env}"
                    ),
                    "query_index": step,
                    "fingerprint": worker_fingerprint,
                }
                rows.append(row)
                if next_observation is not None:
                    live_row_indices.append(len(rows) - 1)
                    live_next_observations.append(next_observation)
                if terminated or truncated:
                    alive[env] = False

        # FSDP full-shard root calls are collective.  Even a rank whose rows
        # all terminated must execute the same one conditioning call; its
        # first current observation is a discarded dummy in that case.
        conditioning_observations = (
            live_next_observations
            if live_next_observations
            else [rows[0]["observation"]]
        )
        next_features, next_proprios = self._condition_observations(
            conditioning_observations
        )
        if live_next_observations:
            for index, feature, proprio in zip(
                live_row_indices,
                next_features,
                next_proprios,
            ):
                rows[index]["next_feature"] = feature
                rows[index]["next_proprio"] = proprio

        assert self.replay is not None
        for row in rows:
            observation = row["observation"]
            next_observation = row["next_observation"]
            next_state_valid = row["next_state_valid"]
            bootstrap_mask = macro_bootstrap_mask(
                success_terminated=row["success_terminated"],
                time_limit_truncated=row["time_limit_truncated"],
                other_truncated=row["other_truncated"],
                next_state_valid=next_state_valid,
            )
            transition = QAMMacroTransition(
                obs_id=observation.content_id(),
                next_obs_id=(
                    next_observation.content_id()
                    if next_observation is not None
                    else None
                ),
                obs_feature=row["feature"].to(dtype=torch.bfloat16),
                obs_proprio=row["proprio"].to(dtype=torch.float32),
                next_obs_feature=(
                    row["next_feature"].to(dtype=torch.bfloat16)
                    if next_state_valid
                    else None
                ),
                next_obs_proprio=(
                    row["next_proprio"].to(dtype=torch.float32)
                    if next_state_valid
                    else None
                ),
                next_state_valid=next_state_valid,
                planned_actions_normalized=row["planned"].to(dtype=torch.float32),
                planned_actions_env=row["planned_env"].to(dtype=torch.float32),
                chunk_rewards_native=row["rewards"],
                reward_macro_discounted=float(
                    fixed_slot_discounted_return(
                        row["rewards"],
                        gamma_slot=float(self.qam_cfg.gamma_slot),
                    ).item()
                ),
                success_terminated=row["success_terminated"],
                time_limit_truncated=row["time_limit_truncated"],
                other_truncated=row["other_truncated"],
                bootstrap_mask=bootstrap_mask,
                policy_version=row["policy_version"],
                episode_id=row["episode_id"],
                query_index=row["query_index"],
                contract_fingerprint=row["fingerprint"],
            )
            self.replay.add(
                transition,
                observation=observation,
                next_observation=next_observation,
            )
        return len(rows)

    def _global_sum_int(self, value: int) -> int:
        tensor = torch.tensor(
            int(value),
            device=self.device,
            dtype=torch.long,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.SUM,
            )
        return int(tensor.item())

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        clear_memory(sync=False)
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        trajectories = [
            await input_channel.get(async_op=True).async_wait()
            for _ in range(split_num)
        ]

        local_added = 0
        for trajectory in trajectories:
            if not isinstance(trajectory, Trajectory):
                raise TypeError("QAM actor expected Trajectory payloads")
            local_added += self._ingest_trajectory(trajectory)
        global_added = self._global_sum_int(local_added)
        self.local_total_inserts += local_added
        self.global_total_inserts += global_added
        self._accrue_update_credit(global_added)
        self._last_ingest_metrics = {
            "qam/local_replay_size": float(
                0 if self.replay is None else len(self.replay)
            ),
            "qam/local_inserts": float(local_added),
            "qam/global_inserts": float(global_added),
            "qam/global_total_inserts": float(self.global_total_inserts),
        }

    def compute_advantages_and_returns(self) -> dict[str, float]:
        """QAM is off-policy and does not construct PPO advantages."""
        return dict(self._last_ingest_metrics)

    def _accrue_update_credit(self, global_added: int) -> None:
        """Start UTD accounting only after the online warm-up boundary."""
        if self.phase == "collect":
            return
        if self.q_only_anchor_global_inserts is None:
            warmup = int(self.qam_cfg.warmup_global_inserts)
            if self.global_total_inserts >= warmup:
                previous_total = self.global_total_inserts - int(global_added)
                anchor = max(previous_total, warmup)
                self.q_only_anchor_global_inserts = anchor
                self.update_credit.pending = 0.0
                self.update_credit.add_global_inserts(
                    self.global_total_inserts - anchor
                )
            return
        self.update_credit.add_global_inserts(global_added)

    def _sample_batch(self) -> list[QAMReplaySample]:
        if self.replay is None:
            raise RuntimeError("QAM replay has not been initialized")
        local_batch = int(self.cfg.actor.global_batch_size) // self._world_size
        return self.replay.sample(local_batch)

    def _stack_critic_batch(
        self,
        samples: list[QAMReplaySample],
    ) -> dict[str, Any]:
        transitions = [sample.transition for sample in samples]
        feature = torch.stack(
            [transition.obs_feature for transition in transitions],
            dim=0,
        ).to(device=self.device, dtype=torch.float32)
        proprio = torch.stack(
            [transition.obs_proprio for transition in transitions],
            dim=0,
        ).to(device=self.device, dtype=torch.float32)
        action = torch.stack(
            [transition.planned_actions_normalized for transition in transitions],
            dim=0,
        ).to(device=self.device, dtype=torch.float32)
        next_feature = torch.stack(
            [
                (
                    transition.next_obs_feature
                    if transition.next_obs_feature is not None
                    else torch.zeros_like(transition.obs_feature)
                )
                for transition in transitions
            ],
            dim=0,
        ).to(device=self.device, dtype=torch.float32)
        next_proprio = torch.stack(
            [
                (
                    transition.next_obs_proprio
                    if transition.next_obs_proprio is not None
                    else torch.zeros_like(transition.obs_proprio)
                )
                for transition in transitions
            ],
            dim=0,
        ).to(device=self.device, dtype=torch.float32)
        return {
            "feature": feature,
            "proprio": proprio,
            "action": action,
            "next_feature": next_feature,
            "next_proprio": next_proprio,
            "return": torch.tensor(
                [transition.reward_macro_discounted for transition in transitions],
                device=self.device,
                dtype=torch.float32,
            ),
            "bootstrap_mask": torch.tensor(
                [transition.bootstrap_mask for transition in transitions],
                device=self.device,
                dtype=torch.float32,
            ),
            # A terminal row has no next policy view. It is harmless to sample
            # an unused action from the current view because its bootstrap mask
            # is exactly zero.
            "next_policy_observations": [
                sample.next_observation or sample.observation for sample in samples
            ],
        }

    @torch.no_grad()
    def _sample_td_next_action(
        self,
        observations: list[QAMPolicyObservation],
    ) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        output = self.model(
            forward_type=ForwardType.QAM_FLOW,
            operation="sample_ode",
            env_obs=self._observations_to_env_obs(observations),
            flow_steps=int(self.qam_cfg.flow_steps),
        )
        self.model.train(was_training)
        return output["qam_planned_action_normalized"].to(
            device=self.device,
            dtype=torch.float32,
        )

    def _average_critic_gradients(self) -> None:
        if self.critic is None or not torch.distributed.is_initialized():
            return
        for parameter in self.critic.parameters():
            if parameter.grad is None:
                continue
            torch.distributed.all_reduce(
                parameter.grad,
                op=torch.distributed.ReduceOp.SUM,
            )
            parameter.grad.div_(self._world_size)

    def _critic_update(
        self,
        samples: list[QAMReplaySample],
    ) -> tuple[
        dict[str, float],
        dict[str, Any],
        dict[str, torch.Tensor],
    ]:
        batch = self._stack_critic_batch(samples)
        self._init_critic(batch["feature"].shape[-1])
        assert self.critic is not None
        assert self.target_critic is not None
        assert self.critic_optimizer is not None

        with torch.no_grad():
            next_action = self._sample_td_next_action(batch["next_policy_observations"])
            next_target_q = self.target_critic(
                batch["next_feature"],
                batch["next_proprio"],
                next_action,
            )
            target = q_chunk_td_target(
                batch["return"],
                batch["bootstrap_mask"],
                next_target_q,
                discount_h=fixed_slot_bootstrap_discount(
                    gamma_slot=float(self.qam_cfg.gamma_slot)
                ),
                rho=float(self.qam_cfg.rho),
            )

        preupdate = clone_parameter_snapshot(self.critic)
        self.critic_optimizer.zero_grad(set_to_none=True)
        q_values = self.critic(
            batch["feature"],
            batch["proprio"],
            batch["action"],
        )
        critic_loss = ensemble_critic_mse(
            q_values,
            target,
            torch.ones_like(target, dtype=torch.bool),
        )
        critic_loss.backward()
        self._average_critic_gradients()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            float(self.qam_cfg.critic_grad_clip),
        )
        if not torch.isfinite(grad_norm):
            self.critic_optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("QAM critic gradient norm is non-finite")
        self.critic_optimizer.step()
        self.critic_updates += 1
        return (
            {
                "qam/critic_loss": float(critic_loss.detach().item()),
                "qam/critic_grad_norm": float(grad_norm.detach().item()),
                "qam/q_mean": float(q_values.detach().mean().item()),
                "qam/q_std_heads": float(
                    q_values.detach().std(dim=0, correction=0).mean().item()
                ),
                "qam/td_target_mean": float(target.detach().mean().item()),
            },
            batch,
            preupdate,
        )

    def _velocity_callable(
        self,
        *,
        conditioning: dict[str, Any],
        route: str,
    ):
        def velocity(flat_state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            if flat_state.ndim == 2:
                state = flat_state.reshape(
                    flat_state.shape[0],
                    MODEL_HORIZON,
                    MODEL_ACTION_DIM,
                )
                result = self.model(
                    forward_type=ForwardType.QAM_FLOW,
                    operation="velocity",
                    state=conditioning["state"],
                    x_t=state,
                    time_qam=time.reshape(flat_state.shape[0]),
                    prefix_pad_masks=conditioning["prefix_pad_masks"],
                    past_key_values=conditioning["past_key_values"],
                    route=route,
                )
                return result.reshape_as(flat_state)
            if flat_state.ndim == 3:
                outputs = [
                    velocity(flat_state[index], time[index])
                    for index in range(flat_state.shape[0])
                ]
                return torch.stack(outputs, dim=0)
            raise ValueError(
                f"QAM velocity expects [B,D] or [K,B,D], got {flat_state.shape}"
            )

        return velocity

    def _am_update(
        self,
        samples: list[QAMReplaySample],
        critic_batch: dict[str, Any],
    ) -> dict[str, float]:
        assert self.target_critic is not None
        observations = [sample.observation for sample in samples]
        was_training = self.model.training
        try:
            self.model.eval()
            conditioning = self.model(
                forward_type=ForwardType.QAM_FLOW,
                operation="conditioning",
                env_obs=self._observations_to_env_obs(observations),
            )
        finally:
            self.model.train(was_training)
        replay_feature = critic_batch["feature"]
        recomputed_feature = conditioning["critic_feature"].to(
            device=self.device,
            dtype=torch.float32,
        )
        if recomputed_feature.shape != replay_feature.shape:
            raise ValueError(
                "QAM frozen-prefix replay round-trip changed feature shape: "
                f"replay={tuple(replay_feature.shape)} "
                f"recomputed={tuple(recomputed_feature.shape)}"
            )
        if not torch.allclose(
            recomputed_feature,
            replay_feature,
            atol=2e-3,
            rtol=2e-3,
        ):
            absolute_error = (recomputed_feature - replay_feature).abs()
            raise ValueError(
                "QAM frozen-prefix replay round-trip changed the critic feature: "
                f"max_abs={absolute_error.max().item():.6g} "
                f"mean_abs={absolute_error.mean().item():.6g}"
            )

        batch_size = len(samples)
        flat_dim = MODEL_HORIZON * MODEL_ACTION_DIM
        initial_noise = torch.randn(
            batch_size,
            flat_dim,
            device=self.device,
            dtype=torch.float32,
        )
        step_noises = torch.randn(
            int(self.qam_cfg.flow_steps),
            batch_size,
            flat_dim,
            device=self.device,
            dtype=torch.float32,
        )
        fine_velocity = self._velocity_callable(
            conditioning=conditioning,
            route="fine",
        )
        behavior_velocity = self._velocity_callable(
            conditioning=conditioning,
            route="behavior",
        )
        path = sample_memoryless_am_path(
            fine_velocity,
            behavior_velocity,
            initial_noise,
            step_noises,
            flow_steps=int(self.qam_cfg.flow_steps),
        )

        def target_critic_from_full_action(full_action: torch.Tensor) -> torch.Tensor:
            model_action = full_action.reshape(
                full_action.shape[0],
                MODEL_HORIZON,
                MODEL_ACTION_DIM,
            )
            planned = project_planned_action(model_action)
            return self.target_critic(
                critic_batch["feature"],
                critic_batch["proprio"],
                planned,
            )

        terminal_adjoint = terminal_mean_q_adjoint(
            target_critic_from_full_action,
            path.endpoint,
            inv_temp=float(self.qam_cfg.inv_temp),
            clip_action=True,
        )
        adjoints = reverse_behavior_adjoint(
            behavior_velocity,
            path,
            terminal_adjoint,
            use_backward_vjp=True,
        )
        self.optimizer.zero_grad(set_to_none=True)
        am_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        for index in range(path.states.shape[0]):
            step_loss = adjoint_matching_step_loss(
                fine_velocity,
                behavior_velocity,
                path.states[index],
                path.times[index],
                path.sigmas[index],
                adjoints[index],
            )
            self.grad_scaler.scale(step_loss).backward()
            am_loss = am_loss + step_loss.detach().to(dtype=torch.float32)
        grad_norm, learning_rates = self.optimizer_step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError("QAM F1 gradient norm is non-finite")
        self.fine_updates += 1
        self.fine_policy_version += 1
        return {
            "qam/am_loss": float(am_loss.detach().item()),
            "qam/fine_grad_norm": float(torch.as_tensor(grad_norm).item()),
            "qam/fine_lr": float(learning_rates[0]),
            "qam/terminal_adjoint_norm": float(
                terminal_adjoint.norm(dim=-1).mean().item()
            ),
        }

    def _local_replay_ready(self) -> bool:
        if self.replay is None:
            return False
        local_batch = int(self.cfg.actor.global_batch_size) // self._world_size
        return len(self.replay) >= max(
            local_batch,
            int(self.qam_cfg.min_replay_per_rank),
        )

    def _all_ranks_replay_ready(self) -> bool:
        ready = torch.tensor(
            int(self._local_replay_ready()),
            device=self.device,
            dtype=torch.long,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                ready,
                op=torch.distributed.ReduceOp.MIN,
            )
        return bool(ready.item())

    def _updates_to_run(self) -> int:
        if self.phase == "collect":
            return 0
        if self.global_total_inserts < int(self.qam_cfg.warmup_global_inserts):
            return 0
        if not self._all_ranks_replay_ready():
            return 0
        return self.update_credit.take(int(self.qam_cfg.max_updates_per_step))

    @Worker.timer("run_training")
    def run_training(self) -> dict[str, float]:
        if self.enable_offload:
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        updates_to_run = self._updates_to_run()
        metrics: dict[str, list[float]] = {}
        am_updates_run = 0
        for _ in range(updates_to_run):
            run_am = _am_is_enabled_for_next_update(
                configured_phase=self.phase,
                critic_updates=self.critic_updates,
                q_only_updates_before_am=int(self.qam_cfg.q_only_updates_before_am),
            )
            samples = self._sample_batch()
            (
                update_metrics,
                critic_batch,
                critic_preupdate,
            ) = self._critic_update(samples)
            if run_am:
                update_metrics.update(self._am_update(samples, critic_batch))
                am_updates_run += 1
            assert self.target_critic is not None
            # Match the official joint-update order: critic and AM losses use
            # the old target; only after both optimizer steps does target-Q
            # read the online critic snapshot taken before its update.
            ema_from_preupdate_(
                self.target_critic,
                critic_preupdate,
                tau=float(self.qam_cfg.target_tau),
            )
            for key, value in update_metrics.items():
                metrics.setdefault(key, []).append(float(value))

        result = {key: float(np.mean(values)) for key, values in metrics.items()}
        am_enabled_next_update = _am_is_enabled_for_next_update(
            configured_phase=self.phase,
            critic_updates=self.critic_updates,
            q_only_updates_before_am=int(self.qam_cfg.q_only_updates_before_am),
        )
        executed_phase = (
            "am_on"
            if am_updates_run > 0
            else ("q_only" if self.phase == "am_on" else self.phase)
        )
        result.update(self._last_ingest_metrics)
        result.update(
            {
                "qam/phase": float(_PHASE_ORDER[executed_phase]),
                "qam/configured_phase": float(_PHASE_ORDER[self.phase]),
                "qam/am_enabled_next_update": float(am_enabled_next_update),
                "qam/am_updates_run": float(am_updates_run),
                "qam/updates_run": float(updates_to_run),
                "qam/critic_updates": float(self.critic_updates),
                "qam/fine_updates": float(self.fine_updates),
                "qam/fine_policy_version": float(self.fine_policy_version),
                "qam/pending_update_credit": float(self.update_credit.pending),
                "qam/q_only_anchor_global_inserts": float(
                    -1
                    if self.q_only_anchor_global_inserts is None
                    else self.q_only_anchor_global_inserts
                ),
                "qam/replay_ready": float(self._local_replay_ready()),
            }
        )
        if torch.distributed.is_initialized():
            result = all_reduce_dict(
                result,
                op=torch.distributed.ReduceOp.AVG,
            )
            torch.distributed.barrier()
        clear_memory()
        return result

    def _sidecar_path(self, base_path: str) -> Path:
        return Path(base_path) / "qam_components" / f"rank_{self._rank}.pt"

    def _replay_checkpoint_path(self, base_path: str) -> Path:
        return Path(base_path) / "qam_components" / f"replay_rank_{self._rank}.pt"

    @staticmethod
    def _completion_manifest_path(base_path: str) -> Path:
        return Path(base_path) / "qam_components" / "complete.json"

    @staticmethod
    def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _atomic_torch_dump(payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        try:
            with temporary.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _distributed_active() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _new_checkpoint_snapshot_id(self) -> str:
        values = [uuid.uuid4().hex if int(self._rank) == 0 else None]
        if self._distributed_active():
            torch.distributed.broadcast_object_list(values, src=0)
        if not isinstance(values[0], str) or not values[0]:
            raise ValueError("QAM checkpoint snapshot ID broadcast failed")
        return values[0]

    def _gather_checkpoint_status(
        self,
        local_status: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self._distributed_active():
            statuses = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(statuses, local_status)
        else:
            statuses = [local_status]
        if any(not isinstance(status, dict) for status in statuses):
            raise ValueError("QAM checkpoint gathered an invalid rank status")

        failures = sorted(
            (int(status["rank"]), str(status["error"]))
            for status in statuses
            if status["error"] is not None
        )
        if failures:
            details = "; ".join(f"rank {rank}: {error}" for rank, error in failures)
            raise ValueError(f"QAM checkpoint failed closed: {details}")

        statuses = sorted(statuses, key=lambda status: int(status["rank"]))
        ranks = [int(status["rank"]) for status in statuses]
        if ranks != list(range(int(self._world_size))):
            raise ValueError(f"QAM checkpoint rank set mismatch: {ranks}")
        signatures = {status["signature"] for status in statuses}
        if len(signatures) != 1:
            raise ValueError("QAM checkpoint rank metadata mismatch")
        return statuses

    @staticmethod
    def _checkpoint_signature(state: dict[str, Any]) -> tuple[Any, ...]:
        """Return state that must match to keep collective update counts equal."""
        snapshot = state["snapshot"]
        feature_dim = state["critic_feature_dim"]
        prefix_lengths = state.get("prefix_block_lengths")
        anchor = state.get("q_only_anchor_global_inserts")
        return (
            int(snapshot["checkpoint_step"]),
            str(snapshot["snapshot_id"]),
            int(state["world_size"]),
            str(state["saved_phase"]),
            state["contract_fingerprint"],
            (
                None
                if prefix_lengths is None
                else tuple(int(value) for value in prefix_lengths)
            ),
            None if feature_dim is None else int(feature_dim),
            bool(state["has_replay"]),
            int(state["runner_global_step"]),
            int(state["fine_policy_version"]),
            int(state["critic_updates"]),
            int(state["fine_updates"]),
            int(state["global_total_inserts"]),
            float(state["pending_update_credit"]),
            None if anchor is None else int(anchor),
            tuple(sorted(state["schedule_contract"].items())),
        )

    def _schedule_contract(self) -> dict[str, int | float]:
        return {
            "warmup_global_inserts": int(self.qam_cfg.warmup_global_inserts),
            "q_only_updates_before_am": int(self.qam_cfg.q_only_updates_before_am),
            "utd_ratio": float(self.qam_cfg.utd_ratio),
            "inv_temp": float(self.qam_cfg.inv_temp),
        }

    def _checkpoint_state(
        self,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "complete": True,
            "snapshot": snapshot,
            "rank": self._rank,
            "world_size": self._world_size,
            "saved_phase": self.phase,
            "contract_fingerprint": self.contract_fingerprint,
            "prefix_block_lengths": self.prefix_block_lengths,
            "critic_feature_dim": self.critic_feature_dim,
            "critic": (self.critic.state_dict() if self.critic is not None else None),
            "target_critic": (
                self.target_critic.state_dict()
                if self.target_critic is not None
                else None
            ),
            "critic_optimizer": (
                self.critic_optimizer.state_dict()
                if self.critic_optimizer is not None
                else None
            ),
            "has_replay": self.replay is not None,
            "runner_global_step": self.runner_global_step,
            "fine_policy_version": self.fine_policy_version,
            "critic_updates": self.critic_updates,
            "fine_updates": self.fine_updates,
            "local_total_inserts": self.local_total_inserts,
            "global_total_inserts": self.global_total_inserts,
            "pending_update_credit": self.update_credit.pending,
            "q_only_anchor_global_inserts": self.q_only_anchor_global_inserts,
            "schedule_contract": self._schedule_contract(),
            "rng_state": get_rng_state(),
        }

    def _write_qam_checkpoint_components(
        self,
        save_base_path: str,
        step: int,
    ) -> None:
        sidecar_path = self._sidecar_path(save_base_path)
        replay_path = self._replay_checkpoint_path(save_base_path)
        snapshot_id = self._new_checkpoint_snapshot_id()
        snapshot = {
            "schema_version": self._QAM_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_step": int(step),
            "snapshot_id": snapshot_id,
            "rank": int(self._rank),
            "world_size": int(self._world_size),
        }
        state = self._checkpoint_state(snapshot)
        error = None
        try:
            if self.replay is not None:
                self.replay.save_checkpoint(replay_path, snapshot=snapshot)
            self._atomic_torch_dump(state, sidecar_path)
        except Exception as exc:  # pragma: no cover - distributed smoke
            error = f"{type(exc).__name__}: {exc}"
        local_status = {
            "rank": int(self._rank),
            "error": error,
            "signature": self._checkpoint_signature(state),
        }
        self._gather_checkpoint_status(local_status)

        manifest_error = [None]
        if int(self._rank) == 0:
            try:
                manifest = {
                    "complete": True,
                    "schema_version": self._QAM_CHECKPOINT_SCHEMA_VERSION,
                    "checkpoint_step": int(step),
                    "snapshot_id": snapshot_id,
                    "world_size": int(self._world_size),
                }
                self._atomic_json_dump(
                    manifest,
                    self._completion_manifest_path(save_base_path),
                )
            except Exception as exc:  # pragma: no cover - shared filesystem
                manifest_error[0] = f"{type(exc).__name__}: {exc}"
        if self._distributed_active():
            torch.distributed.broadcast_object_list(manifest_error, src=0)
        if manifest_error[0] is not None:
            raise ValueError(f"could not complete QAM checkpoint: {manifest_error[0]}")

    def _preflight_qam_checkpoint(
        self,
        load_base_path: str,
    ) -> dict[str, Any]:
        state = None
        signature = None
        error = None
        try:
            manifest_path = self._completion_manifest_path(load_base_path)
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("complete") is not True:
                raise ValueError("QAM completion manifest is not complete")
            if (
                int(manifest.get("schema_version", -1))
                != self._QAM_CHECKPOINT_SCHEMA_VERSION
            ):
                raise ValueError("QAM completion manifest schema mismatch")
            if int(manifest.get("world_size", -1)) != int(self._world_size):
                raise ValueError("QAM completion manifest world-size mismatch")
            checkpoint_step = int(manifest["checkpoint_step"])
            snapshot_id = manifest.get("snapshot_id")
            if not isinstance(snapshot_id, str) or not snapshot_id:
                raise ValueError("QAM completion manifest snapshot ID is invalid")

            sidecar_path = self._sidecar_path(load_base_path)
            state = torch.load(
                sidecar_path,
                map_location="cpu",
                weights_only=False,
            )
            expected_snapshot = {
                "schema_version": self._QAM_CHECKPOINT_SCHEMA_VERSION,
                "checkpoint_step": checkpoint_step,
                "snapshot_id": snapshot_id,
                "rank": int(self._rank),
                "world_size": int(self._world_size),
            }
            if state.get("complete") is not True:
                raise ValueError("incomplete QAM worker checkpoint")
            if state.get("snapshot") != expected_snapshot:
                raise ValueError("QAM sidecar/manifest snapshot mismatch")
            if (state.get("rank"), state.get("world_size")) != (
                self._rank,
                self._world_size,
            ):
                raise ValueError("QAM resume requires the same rank/world size")
            saved_phase = str(state["saved_phase"])
            if not _phase_transition_is_valid(saved_phase, self.phase):
                raise ValueError(
                    "QAM resume phase must move monotonically "
                    "collect -> q_only -> am_on"
                )
            if "rng_state" not in state:
                raise ValueError("QAM checkpoint is missing rank-local RNG")
            if state.get("schedule_contract") != self._schedule_contract():
                raise ValueError("QAM resume schedule contract mismatch")

            critic_feature_dim = state["critic_feature_dim"]
            critic_components = (
                state.get("critic"),
                state.get("target_critic"),
                state.get("critic_optimizer"),
            )
            if critic_feature_dim is None:
                if any(component is not None for component in critic_components):
                    raise ValueError("QAM critic checkpoint is inconsistent")
            elif any(component is None for component in critic_components):
                raise ValueError("QAM critic checkpoint is incomplete")
            else:
                critic_feature_dim = int(critic_feature_dim)

            prefix_block_lengths = state.get("prefix_block_lengths")
            if prefix_block_lengths is not None:
                self.prefix_block_lengths = validate_qam_prefix_block_lengths(
                    torch.tensor(prefix_block_lengths),
                    feature_blocks=4,
                )
            contract_fingerprint = state["contract_fingerprint"]
            if contract_fingerprint is not None:
                self._init_replay(contract_fingerprint)

            signature = self._checkpoint_signature(state)
            has_replay = bool(state["has_replay"])
            replay_path = self._replay_checkpoint_path(load_base_path)
            if has_replay:
                if self.replay is None:
                    raise ValueError("QAM replay checkpoint is missing its contract")
                self.replay.load_checkpoint(
                    replay_path,
                    expected_snapshot=expected_snapshot,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        self._gather_checkpoint_status(
            {
                "rank": int(self._rank),
                "error": error,
                "signature": signature,
            }
        )
        assert state is not None
        return state

    def save_checkpoint(self, save_base_path: str, step: int) -> None:
        """Save F1 with FSDP and worker-only QAM state in rank sidecars."""
        manifest_error = [None]
        if int(self._rank) == 0:
            try:
                self._completion_manifest_path(save_base_path).unlink(missing_ok=True)
            except Exception as exc:  # pragma: no cover - shared filesystem
                manifest_error[0] = f"{type(exc).__name__}: {exc}"
        if self._distributed_active():
            torch.distributed.broadcast_object_list(manifest_error, src=0)
        if manifest_error[0] is not None:
            raise ValueError(
                "could not invalidate prior QAM completion manifest: "
                f"{manifest_error[0]}"
            )
        super().save_checkpoint(save_base_path, step)
        self._write_qam_checkpoint_components(save_base_path, step)

    def load_checkpoint(self, load_base_path: str) -> None:
        """Restore F1 and exact QAM sidecar/replay continuation state."""
        state = self._preflight_qam_checkpoint(load_base_path)
        super().load_checkpoint(load_base_path)

        feature_dim = state["critic_feature_dim"]
        if feature_dim is not None:
            self._init_critic(feature_dim)
            assert self.critic is not None
            assert self.target_critic is not None
            assert self.critic_optimizer is not None
            self.critic.load_state_dict(state["critic"], strict=True)
            self.target_critic.load_state_dict(
                state["target_critic"],
                strict=True,
            )
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])

        self.runner_global_step = int(state["runner_global_step"])
        self.fine_policy_version = int(state["fine_policy_version"])
        self.critic_updates = int(state["critic_updates"])
        self.fine_updates = int(state["fine_updates"])
        self.local_total_inserts = int(state["local_total_inserts"])
        self.global_total_inserts = int(state["global_total_inserts"])
        pending, anchor = _resume_update_credit(
            saved_phase=str(state["saved_phase"]),
            requested_phase=self.phase,
            saved_pending=float(state["pending_update_credit"]),
            saved_anchor=state.get("q_only_anchor_global_inserts"),
            global_total_inserts=self.global_total_inserts,
            warmup_global_inserts=int(self.qam_cfg.warmup_global_inserts),
        )
        self.update_credit = QAMUpdateCredit(
            utd_ratio=float(self.qam_cfg.utd_ratio),
            pending=pending,
        )
        self.q_only_anchor_global_inserts = anchor
        set_rng_state(state["rng_state"])
