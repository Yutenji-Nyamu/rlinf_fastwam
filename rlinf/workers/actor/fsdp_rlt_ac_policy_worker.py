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

import hashlib
import json
import os
import queue
import re

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.dvac_weighting import (
    FrozenGlobalZMoments,
    build_rlt_bc_targets_and_weights,
    centered_mean_one_weights,
    episode_success_flags,
    global_z_weights,
    masked_weight_totals,
    straight_through_scale_actions,
    summarize_weights,
)
from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    collect_trajectory_replay_metrics,
    compute_split_num,
    trajectory_has_bool_tensor,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
    AsyncEmbodiedSACFSDPPolicy,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class RLTACLossMixin:
    """RLT actor-critic losses on top of RLinf replay-buffer worker plumbing.

    Forward types follow the existing off-policy actor-critic API, while the
    RLT objective disables entropy/alpha and uses a fixed-std actor, min-Q
    critic target, Q1 actor objective, and BC regularization.
    """

    def _rlt_transition_replay_cfg(self):
        return self.cfg.algorithm.get("rlt_transition_replay", {}) or {}

    def _bootstrap_on_truncation(self) -> bool:
        return bool(
            self._rlt_transition_replay_cfg().get("bootstrap_on_truncation", False)
        )

    def _use_compact_rlt_transition(self) -> bool:
        return bool(self._rlt_transition_replay_cfg().get("compact", False))

    @staticmethod
    def _flatten_chunk(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() <= 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _chunk_shape(self) -> tuple[int, int]:
        chunk_len = int(self.cfg.actor.model.num_action_chunks)
        action_dim = int(self.cfg.actor.model.action_dim)
        return chunk_len, action_dim

    def _rlt_dvac_prepare_actor_actions(
        self,
        pi: torch.Tensor,
        curr_obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict | None, dict[str, float]]:
        """Keep shared sync/async RLT behavior unchanged when DVAC is off."""
        del curr_obs
        return pi, None, {"rlt_dvac/enabled": 0.0}

    def _rlt_dvac_context_metrics(self, payload, **kwargs) -> dict[str, float]:
        del payload, kwargs
        return {}

    def _maybe_write_rlt_dvac_trace(self, payload, **kwargs) -> None:
        del payload, kwargs

    def get_rollout_sync_version(self) -> int:
        """Expose learner update count when RLT warmup gates actor rollout."""
        if not self.use_rlt_schedule:
            return int(self.version)
        return int(self.update_step)

    def _ref_chunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        chunk_len, action_dim = self._chunk_shape()
        ref_chunk = self._flatten_chunk(obs["ref_chunk"]).reshape(
            obs["ref_chunk"].shape[0], -1, action_dim
        )
        return ref_chunk[:, :chunk_len].reshape(ref_chunk.shape[0], -1)

    @staticmethod
    def _require_twin_q(all_q_values: torch.Tensor) -> None:
        if all_q_values.shape[-1] < 2:
            raise ValueError(
                "RLT Stage 2 requires at least two Q heads for twin-Q training, "
                f"got Q shape {tuple(all_q_values.shape)}."
            )

    def _min_twin_q(self, all_q_values: torch.Tensor) -> torch.Tensor:
        self._require_twin_q(all_q_values)
        return torch.minimum(all_q_values[..., 0:1], all_q_values[..., 1:2])

    def _q1(self, all_q_values: torch.Tensor) -> torch.Tensor:
        self._require_twin_q(all_q_values)
        return all_q_values[..., 0:1]

    def _discounted_chunk_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards = rewards.reshape(rewards.shape[0], -1)
        rewards = rewards.to(self.torch_dtype)
        chunk_len = rewards.shape[-1]
        discounts = torch.pow(
            torch.as_tensor(self.cfg.algorithm.gamma, device=rewards.device),
            torch.arange(chunk_len, device=rewards.device, dtype=rewards.dtype),
        )
        return torch.sum(rewards * discounts, dim=-1, keepdim=True)

    def _bc_metrics(
        self,
        pi: torch.Tensor,
        actions: torch.Tensor,
        ref_chunk: torch.Tensor,
        intervene_flags: torch.Tensor | None,
        *,
        episode_success: torch.Tensor | None = None,
        success_weights: torch.Tensor | None = None,
        success_episode_bc: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        chunk_len, action_dim = self._chunk_shape()
        pi_chunk = self._flatten_chunk(pi).reshape(-1, chunk_len, action_dim)
        action_chunk = self._flatten_chunk(actions).reshape(-1, chunk_len, action_dim)
        bc_ref_chunk = self._flatten_chunk(ref_chunk).reshape(
            ref_chunk.shape[0], -1, action_dim
        )[:, :chunk_len]

        if intervene_flags is None:
            human_mask = torch.zeros(
                pi_chunk.shape[:2], dtype=torch.bool, device=pi_chunk.device
            )
        else:
            human_mask = (
                self._flatten_chunk(intervene_flags)
                .to(device=pi_chunk.device)
                .bool()
                .reshape(-1, chunk_len, action_dim)
                .any(dim=-1)
            )

        bc_target, bc_weights, success_mask, executed_target_mask = (
            build_rlt_bc_targets_and_weights(
                action_chunk,
                bc_ref_chunk,
                human_mask,
                episode_success=episode_success,
                success_weights=success_weights,
                success_episode_bc=success_episode_bc,
            )
        )
        bc_error = torch.mean(torch.square(pi_chunk - bc_target), dim=-1)
        bc_loss = torch.mean(bc_weights * bc_error)

        policy_mask = ~executed_target_mask
        ref_error = torch.mean(torch.square(pi_chunk - bc_ref_chunk), dim=-1)
        human_error = torch.mean(torch.square(pi_chunk - action_chunk), dim=-1)
        bc_ref = torch.sum(ref_error * policy_mask.to(ref_error.dtype)) / torch.clamp(
            torch.sum(policy_mask.to(ref_error.dtype)), min=1.0
        )
        bc_human = torch.sum(
            human_error * human_mask.to(human_error.dtype)
        ) / torch.clamp(torch.sum(human_mask.to(human_error.dtype)), min=1.0)

        human_ratio = torch.mean(human_mask.to(torch.float32)).item()
        success_count = torch.sum(success_mask.to(torch.float32))
        success_denom = torch.clamp(success_count, min=1.0)
        success_unweighted = (
            torch.sum(bc_error * success_mask.to(bc_error.dtype)) / success_denom
        )
        success_weighted = (
            torch.sum(bc_weights * bc_error * success_mask.to(bc_error.dtype))
            / success_denom
        )
        success_weight_mean = (
            torch.sum(bc_weights * success_mask.to(bc_weights.dtype)) / success_denom
        )
        success_weight_sum = torch.sum(
            bc_weights * success_mask.to(bc_weights.dtype), dim=-1
        )
        success_weight_sum_sq = torch.sum(
            bc_weights.square() * success_mask.to(bc_weights.dtype), dim=-1
        )
        success_horizon = torch.sum(success_mask.to(bc_weights.dtype), dim=-1)
        success_query = success_horizon > 0
        success_ess = success_weight_sum.square() / (
            success_horizon * success_weight_sum_sq
        ).clamp_min(1e-12)
        success_ess_mean = torch.sum(
            success_ess * success_query.to(success_ess.dtype)
        ) / torch.clamp(torch.sum(success_query.to(success_ess.dtype)), min=1.0)
        executed_ref_distance = torch.mean(
            torch.square(action_chunk - bc_ref_chunk), dim=-1
        )
        success_executed_ref_distance = (
            torch.sum(
                executed_ref_distance * success_mask.to(executed_ref_distance.dtype)
            )
            / success_denom
        )
        metrics = {
            "bc_loss": bc_loss.detach().item(),
            "bc_unweighted_loss": bc_error.detach().mean().item(),
            "bc_ref_loss": bc_ref.detach().item(),
            "bc_human_loss": bc_human.detach().item(),
            "human_mask_ratio": human_ratio,
            "policy_mask_ratio": 1.0 - human_ratio,
            "rlt_dvac_bc/success_query_count": float(success_query.sum().item()),
            "rlt_dvac_bc/success_action_count": float(success_count.item()),
            "rlt_dvac_bc/executed_target_ratio": float(
                executed_target_mask.float().mean().item()
            ),
            "rlt_dvac_bc/success_unweighted_loss": float(
                success_unweighted.detach().item()
            ),
            "rlt_dvac_bc/success_weighted_loss": float(
                success_weighted.detach().item()
            ),
            "rlt_dvac_bc/success_weight_mean": float(
                success_weight_mean.detach().item()
            ),
            "rlt_dvac_bc/success_weight_ess_ratio": float(
                success_ess_mean.detach().item()
            ),
            "rlt_dvac_bc/success_executed_ref_mse": float(
                success_executed_ref_distance.detach().item()
            ),
        }
        return bc_loss, metrics

    def _actor_objective_weights(self) -> tuple[float, float, dict[str, float]]:
        """Resolve RLT actor-objective BC/Q weights."""
        schedule_cfg = self.cfg.algorithm.get("actor_weight_schedule", {})
        schedule_enabled = bool(schedule_cfg.get("enable", False))
        if not schedule_enabled:
            bc_weight = float(self.cfg.algorithm.get("bc_weight", 1.0))
            q_weight = float(self.cfg.algorithm.get("q_weight", 1.0))
            return (
                bc_weight,
                q_weight,
                {
                    "bc_weight": bc_weight,
                    "q_weight": q_weight,
                    "actor_weight_schedule_enabled": 0.0,
                    "actor_weight_in_warmup": 0.0,
                    "actor_weight_ramp_progress": 1.0,
                },
            )

        weight_warmup_updates = int(schedule_cfg.get("warmup_updates", 0))
        ramp_updates = int(schedule_cfg.get("ramp_updates", 0))
        in_warmup = int(self.update_step) < weight_warmup_updates
        warmup_bc_weight = float(
            schedule_cfg.get(
                "warmup_bc_weight",
                self.cfg.algorithm.get("bc_weight", 1.0),
            )
        )
        warmup_q_weight = float(
            schedule_cfg.get(
                "warmup_q_weight",
                self.cfg.algorithm.get("q_weight", 1.0),
            )
        )
        online_bc_weight = float(
            schedule_cfg.get(
                "online_bc_weight",
                self.cfg.algorithm.get("bc_weight", 1.0),
            )
        )
        online_q_weight = float(
            schedule_cfg.get(
                "online_q_weight",
                self.cfg.algorithm.get("q_weight", 1.0),
            )
        )
        if in_warmup:
            bc_weight = warmup_bc_weight
            q_weight = warmup_q_weight
            ramp_progress = 0.0
        elif ramp_updates > 0:
            ramp_progress = min(
                1.0,
                max(
                    0.0,
                    float(int(self.update_step) - weight_warmup_updates + 1)
                    / float(ramp_updates),
                ),
            )
            bc_weight = warmup_bc_weight + ramp_progress * (
                online_bc_weight - warmup_bc_weight
            )
            q_weight = warmup_q_weight + ramp_progress * (
                online_q_weight - warmup_q_weight
            )
        else:
            bc_weight = online_bc_weight
            q_weight = online_q_weight
            ramp_progress = 1.0

        metrics = {
            "bc_weight": bc_weight,
            "q_weight": q_weight,
            "actor_weight_schedule_enabled": 1.0,
            "actor_weight_in_warmup": float(in_warmup),
            "actor_weight_ramp_progress": ramp_progress,
        }
        return bc_weight, q_weight, metrics

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        done_source = batch["terminations"]
        if (
            use_simulator_transition_replay(self.cfg)
            and not self._bootstrap_on_truncation()
        ):
            done_source = batch["dones"]
        done_source = done_source.to(self.torch_dtype)
        not_done = ~done_source.reshape(done_source.shape[0], -1).bool().any(
            dim=-1, keepdim=True
        )

        with torch.no_grad():
            next_actions, _, _ = self.model(
                forward_type=ForwardType.SAC,
                obs=next_obs,
            )

            if not use_crossq:
                all_qf_next_target = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_obs,
                    actions=next_actions,
                )
                q_next = self._min_twin_q(all_qf_next_target)
            else:
                _, all_qf_next = self.model(
                    forward_type=ForwardType.CROSSQ_Q,
                    obs=curr_obs,
                    actions=actions,
                    next_obs=next_obs,
                    next_actions=next_actions,
                )
                q_next = self._min_twin_q(all_qf_next.detach())

            reward_target = self._discounted_chunk_rewards(rewards)
            reward_horizon = int(rewards.reshape(rewards.shape[0], -1).shape[-1])
            bootstrap_discount = self.cfg.algorithm.gamma**reward_horizon
            if bootstrap_type == "always":
                target_q_values = reward_target + bootstrap_discount * q_next
            elif bootstrap_type == "standard":
                target_q_values = reward_target + not_done * bootstrap_discount * q_next
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        if not use_crossq:
            all_data_q_values = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions,
            )
        else:
            all_data_q_values, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=actions,
                next_obs=next_obs,
                next_actions=next_actions,
            )

        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        return critic_loss, {"q_data": all_data_q_values.mean().item()}

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"

        curr_obs = batch["curr_obs"]
        reference_dropout_prob = float(
            self.cfg.algorithm.get("reference_dropout_prob", 0.0)
        )
        pi, log_pi, _ = self.model(
            forward_type=ForwardType.SAC,
            obs=curr_obs,
            apply_reference_dropout=True,
            reference_dropout_prob=reference_dropout_prob,
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)
        pi_for_q, dvac_payload, dvac_metrics = self._rlt_dvac_prepare_actor_actions(
            pi, curr_obs
        )

        if not use_crossq:
            all_qf_pi = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=pi_for_q,
                detach_encoder=True,
            )
        else:
            all_qf_pi, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=pi_for_q,
                next_obs=None,
                next_actions=None,
                detach_encoder=True,
            )

        num_q_values = all_qf_pi.shape[-1]
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(num_q_values)
        }
        qf_pi = self._q1(all_qf_pi)
        metrics["q_pi"] = qf_pi.mean().item()

        ref_chunk = self._ref_chunk(curr_obs)
        bc_loss, rlt_metrics = self._bc_metrics(
            pi=pi,
            actions=batch["actions"],
            ref_chunk=ref_chunk,
            intervene_flags=batch.get("intervene_flags", None),
            episode_success=curr_obs.get("episode_success"),
            success_weights=(
                dvac_payload["weights"] if dvac_payload is not None else None
            ),
            success_episode_bc=bool(
                dvac_payload is not None
                and dvac_payload.get("success_episode_bc_applied", False)
            ),
        )
        metrics.update(rlt_metrics)
        metrics.update(
            self._rlt_dvac_context_metrics(
                dvac_payload,
                pi=pi,
                ref_chunk=ref_chunk,
                batch=batch,
                all_qf_pi=all_qf_pi,
            )
        )
        metrics.update(dvac_metrics)
        self._maybe_write_rlt_dvac_trace(
            dvac_payload,
            pi=pi,
            ref_chunk=ref_chunk,
            curr_obs=curr_obs,
            batch=batch,
        )

        entropy = -log_pi.mean()
        bc_weight, q_weight, weight_metrics = self._actor_objective_weights()
        actor_loss = -q_weight * qf_pi.mean() + bc_weight * bc_loss
        metrics.update(weight_metrics)
        metrics["action_ref_abs_mean"] = (
            (self._flatten_chunk(pi) - self._flatten_chunk(ref_chunk))
            .abs()
            .mean()
            .detach()
            .item()
        )
        metrics["weighted_q"] = (q_weight * qf_pi.mean()).detach().item()
        metrics["weighted_bc"] = (bc_weight * bc_loss).detach().item()
        metrics["reference_dropout_prob"] = reference_dropout_prob

        return actor_loss, entropy, metrics

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        del batch
        raise NotImplementedError(
            "RLT AC disables entropy/alpha training. Use "
            "algorithm.entropy_tuning.alpha_type=fixed_alpha."
        )


class RLTACReplayMixin:
    """Shared rollout-to-replay ingestion for sync and async RLT AC workers."""

    def _accumulate_rlt_dvac_baseline(
        self, replay_trajectories: list[Trajectory]
    ) -> None:
        """No-op for workers that do not opt into teacher-DVAC weighting."""
        del replay_trajectories

    @staticmethod
    def _trajectory_transition_count(traj: Trajectory) -> int:
        if traj.actions is None:
            return 0
        return int(traj.actions.shape[0] * traj.actions.shape[1])

    @staticmethod
    def _trajectory_completed_episodes(traj: Trajectory) -> int:
        dones = traj.dones
        if dones is None:
            return 0
        return int(dones.reshape(dones.shape[0], dones.shape[1], -1).any(dim=-1).sum())

    @staticmethod
    def _transition_reward_value(traj: Trajectory) -> float | None:
        rewards = traj.rewards
        if not isinstance(rewards, torch.Tensor) or rewards.numel() == 0:
            return None
        return float(rewards.detach().float().reshape(-1).sum().item())

    @staticmethod
    def _transition_done_value(traj: Trajectory) -> bool | None:
        dones = traj.dones
        if not isinstance(dones, torch.Tensor) or dones.numel() == 0:
            return None
        return bool(dones.detach().to(torch.bool).reshape(-1).any().item())

    @staticmethod
    def _row_tensor(tensor: torch.Tensor, idx: int) -> torch.Tensor:
        return tensor[idx].detach().clone().unsqueeze(0).unsqueeze(0).cpu().contiguous()

    @staticmethod
    def _step_env_tensor(
        tensor: torch.Tensor, step_idx: int, env_idx: int
    ) -> torch.Tensor:
        return (
            tensor[step_idx, env_idx]
            .detach()
            .clone()
            .unsqueeze(0)
            .unsqueeze(0)
            .cpu()
            .contiguous()
        )

    def _row_tensor_dict(
        self,
        tensor_dict: dict[str, object],
        idx: int,
    ) -> dict[str, torch.Tensor]:
        row_dict = {}
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor) and idx < value.shape[0]:
                row_dict[key] = self._row_tensor(value, idx)
        return row_dict

    def _rlt_obs_from_flat_dict(
        self,
        flat: dict,
        dict_key: str,
        idx: int,
    ) -> dict[str, torch.Tensor] | None:
        value = flat.get(dict_key)
        if not isinstance(value, dict):
            return None
        obs = self._row_tensor_dict(value, idx)
        return obs if obs else None

    @staticmethod
    def _flat_record_transition(flat: dict, idx: int) -> bool:
        forward_inputs = flat.get("forward_inputs")
        if not isinstance(forward_inputs, dict):
            return False
        record_transition = forward_inputs.get("record_transition")
        if not isinstance(record_transition, torch.Tensor):
            return False
        if idx >= record_transition.shape[0]:
            return False
        return bool(record_transition[idx].detach().to(torch.bool).reshape(-1).all())

    def _transition_replay_trajectories(
        self,
        trajectory: Trajectory,
    ) -> tuple[list[Trajectory], int]:
        if (
            trajectory.actions is None
            or trajectory.rewards is None
            or self.replay_buffer is None
        ):
            return [], 0

        flat = self.replay_buffer._flatten_trajectory(trajectory)
        actions = flat.get("actions")
        rewards = flat.get("rewards")
        if not isinstance(actions, torch.Tensor) or not isinstance(
            rewards, torch.Tensor
        ):
            return [], 0

        if self._use_compact_rlt_transition():
            tensor_fields = (
                "actions",
                "intervene_flags",
                "rewards",
                "terminations",
                "truncations",
                "dones",
            )
            dict_fields = ()
        else:
            tensor_fields = (
                "actions",
                "intervene_flags",
                "rewards",
                "terminations",
                "truncations",
                "dones",
                "prev_logprobs",
                "prev_values",
                "versions",
            )
            dict_fields = ("forward_inputs",)
        replay_trajectories = []
        completed_episodes = 0
        traj_len = int(trajectory.actions.shape[0])
        bsz = int(trajectory.actions.shape[1])
        num_rows = int(actions.shape[0])
        auto_reset = bool(self.cfg.env.train.get("auto_reset", False))
        add_episode_success = bool(
            getattr(self, "rlt_dvac_application", "q_gradient") == "success_episode_bc"
            and getattr(self, "rlt_dvac_mode", "off") != "off"
        )
        success_by_env = (
            episode_success_flags(trajectory.rewards) if add_episode_success else None
        )

        for env_idx in range(bsz):
            for t in range(traj_len):
                idx = t * bsz + env_idx
                if idx >= num_rows:
                    break
                if not self._flat_record_transition(flat, idx):
                    continue

                transition = Trajectory(
                    max_episode_length=1,
                    model_weights_id=trajectory.model_weights_id,
                )
                for field_name in tensor_fields:
                    value = flat.get(field_name)
                    if isinstance(value, torch.Tensor) and idx < value.shape[0]:
                        setattr(transition, field_name, self._row_tensor(value, idx))
                for field_name in dict_fields:
                    value = flat.get(field_name)
                    if isinstance(value, dict):
                        setattr(
                            transition, field_name, self._row_tensor_dict(value, idx)
                        )

                curr_obs = self._rlt_obs_from_flat_dict(flat, "curr_obs", idx)
                if curr_obs is None:
                    raise ValueError(
                        "RLT transition replay requires curr_obs. Ensure "
                        "update_rlt_transitions() populated transition obs "
                        f"before replay ingestion, got row index {idx}."
                    )
                transition.curr_obs = curr_obs
                if success_by_env is not None:
                    transition.curr_obs["episode_success"] = (
                        success_by_env[env_idx]
                        .detach()
                        .clone()
                        .reshape(1, 1, 1)
                        .cpu()
                        .contiguous()
                    )

                # Dones have one extra initial slot, so transition t reads
                # terminal flags from t+1. Rewards are already action-aligned
                # by EmbodiedRolloutResult because the initial empty reward is
                # skipped and the final reward is appended after rollout.
                done_idx = min(
                    t + 1,
                    int(trajectory.dones.shape[0]) - 1
                    if isinstance(trajectory.dones, torch.Tensor)
                    else traj_len - 1,
                )
                for done_field in ("dones", "terminations", "truncations"):
                    done_value = getattr(trajectory, done_field, None)
                    if (
                        isinstance(done_value, torch.Tensor)
                        and done_idx < done_value.shape[0]
                        and env_idx < done_value.shape[1]
                    ):
                        setattr(
                            transition,
                            done_field,
                            self._step_env_tensor(done_value, done_idx, env_idx),
                        )

                is_done = (
                    isinstance(transition.dones, torch.Tensor)
                    and transition.dones.reshape(-1).to(torch.bool).any()
                )
                is_termination = (
                    isinstance(transition.terminations, torch.Tensor)
                    and transition.terminations.reshape(-1).to(torch.bool).any()
                )
                is_truncation = (
                    isinstance(transition.truncations, torch.Tensor)
                    and transition.truncations.reshape(-1).to(torch.bool).any()
                )
                bootstrap_truncation = bool(
                    self._bootstrap_on_truncation()
                    and is_truncation
                    and not is_termination
                )
                if is_done and not bootstrap_truncation:
                    next_obs = curr_obs
                else:
                    next_obs = self._rlt_obs_from_flat_dict(flat, "next_obs", idx)
                if next_obs is not None:
                    transition.next_obs = next_obs
                else:
                    raise ValueError(
                        "RLT transition replay requires next_obs for non-terminal "
                        "transitions. Ensure update_rlt_transitions() populated "
                        f"transition obs before replay ingestion, got row index {idx}."
                    )

                replay_trajectories.append(transition)
                if is_done:
                    completed_episodes += 1
                    if not auto_reset:
                        break

        return replay_trajectories, completed_episodes

    def _transition_replay_metrics(
        self,
        replay_trajectories: list[Trajectory],
    ) -> dict[str, float]:
        metrics = {"replay/transition_count": float(len(replay_trajectories))}
        reward_values = [
            reward
            for traj in replay_trajectories
            if (reward := self._transition_reward_value(traj)) is not None
        ]
        if reward_values:
            metrics["replay/reward_mean"] = float(
                sum(reward_values) / len(reward_values)
            )
            metrics["replay/reward_positive_rate"] = float(
                sum(reward > 0.0 for reward in reward_values) / len(reward_values)
            )
        done_values = [
            done
            for traj in replay_trajectories
            if (done := self._transition_done_value(traj)) is not None
        ]
        if done_values:
            metrics["replay/done_rate"] = float(
                sum(bool(done) for done in done_values) / len(done_values)
            )
        return metrics

    def _ingest_rollout_trajectories(
        self,
        recv_list: list[Trajectory],
    ) -> tuple[int, int]:
        self._last_replay_metrics = {}

        if use_simulator_transition_replay(self.cfg):
            replay_list = []
            completed = 0
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                transition_trajs, completed_count = (
                    self._transition_replay_trajectories(traj)
                )
                replay_list.extend(transition_trajs)
                completed += completed_count
            self._accumulate_rlt_dvac_baseline(replay_list)
            self._last_replay_metrics = {
                **self._transition_replay_metrics(replay_list),
                **collect_trajectory_replay_metrics(recv_list, reducer=all_reduce_dict),
            }
            self.replay_buffer.add_trajectories(replay_list)

            if self.demo_buffer is not None:
                intervene_traj_list = [
                    traj
                    for traj in replay_list
                    if trajectory_has_bool_tensor(traj.intervene_flags)
                ]
                if len(intervene_traj_list) > 0:
                    self.demo_buffer.add_trajectories(intervene_traj_list)

            return len(replay_list), completed

        self.replay_buffer.add_trajectories(recv_list)

        if self.demo_buffer is not None:
            intervene_traj_list = []
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                intervene_trajs = traj.extract_intervene_traj()
                if intervene_trajs is not None:
                    intervene_traj_list.extend(intervene_trajs)

            if len(intervene_traj_list) > 0:
                self.demo_buffer.add_trajectories(intervene_traj_list)

        added = sum(self._trajectory_transition_count(traj) for traj in recv_list)
        completed = sum(self._trajectory_completed_episodes(traj) for traj in recv_list)
        self._last_replay_metrics = collect_trajectory_replay_metrics(
            recv_list, reducer=all_reduce_dict
        )
        return added, completed

    def _update_rollout_ingest_counters(self, added: int, completed: int) -> None:
        if not getattr(self, "use_rlt_schedule", False):
            return
        if not hasattr(self, "transitions_since_train"):
            return
        self.transitions_since_train += added
        self.episodes_since_train += completed
        self.total_transitions_added += added
        self.total_episodes_added += completed


class RLTACFSDPPolicy(RLTACLossMixin, RLTACReplayMixin, EmbodiedSACFSDPPolicy):
    """Synchronous RLT AC worker with transition replay and warmup scheduling."""

    _RLT_TRAINER_STATE_SCHEMA_VERSION = 1
    _RLT_REQUIRED_CONTRACT_KEYS = (
        "stage1_manifest_path",
        "stage1_manifest_id",
        "stage1_manifest_sha256",
        "norm_stats_sha256",
        "canonical_adapter_version",
    )
    _RLT_REQUIRED_STATE_KEYS = (
        "schema_version",
        "rank",
        "saved_runner_step",
        "actor_world_size",
        "rlt_resume_contract",
        "rlt_resume_contract_sha256",
        "update_step",
        "local_total_transitions_added",
        "local_total_episodes_added",
        "global_warmup_ready_total_transitions",
        "global_warmup_ready_total_episodes",
    )

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rlt_schedule_cfg = cfg.algorithm.get("rlt_schedule", {}) or {}
        self.use_rlt_schedule = bool(self.rlt_schedule_cfg.get("enable", False))
        self.rlt_resume_cfg = cfg.algorithm.get("rlt_resume", {}) or {}
        self.use_rlt_resume = bool(self.rlt_resume_cfg.get("enable", False))
        self.transitions_since_train = 0
        self.episodes_since_train = 0
        self.total_transitions_added = 0
        self.total_episodes_added = 0
        self._warmup_ready_total_transitions: int | None = None
        self._warmup_ready_total_episodes: int | None = None
        self.pending_update_budget = 0

        dvac_cfg = cfg.algorithm.get("rlt_dvac", {}) or {}
        self.rlt_dvac_cfg = dict(self._plain_config(dvac_cfg))
        self.rlt_dvac_mode = str(self.rlt_dvac_cfg.get("mode", "off")).lower()
        if self.rlt_dvac_mode not in {"off", "observe", "apply"}:
            raise ValueError(
                "algorithm.rlt_dvac.mode must be off, observe, or apply, got "
                f"{self.rlt_dvac_mode!r}."
            )
        self.rlt_dvac_l_values = tuple(
            int(value) for value in self.rlt_dvac_cfg.get("l_values", (2, 3, 4))
        )
        if self.rlt_dvac_l_values != (2, 3, 4):
            raise ValueError(
                "RLT teacher telemetry stores DVAC rows in fixed L=(2,3,4) order."
            )
        self.rlt_dvac_selected_l = int(self.rlt_dvac_cfg.get("selected_l", 3))
        if self.rlt_dvac_selected_l not in self.rlt_dvac_l_values:
            raise ValueError("RLT DVAC selected_l must be included in l_values.")
        self.rlt_dvac_horizon = int(
            self.rlt_dvac_cfg.get(
                "applied_horizon", self.cfg.actor.model.num_action_chunks
            )
        )
        if self.rlt_dvac_horizon != int(self.cfg.actor.model.num_action_chunks):
            raise ValueError(
                "RLT DVAC applied_horizon must match the student action chunk."
            )
        self.rlt_dvac_z_clip = float(self.rlt_dvac_cfg.get("z_clip", 2.0))
        self.rlt_dvac_strength = float(self.rlt_dvac_cfg.get("strength", 0.5))
        self.rlt_dvac_application = str(
            self.rlt_dvac_cfg.get("application", "q_gradient")
        ).lower()
        if self.rlt_dvac_application not in {"q_gradient", "success_episode_bc"}:
            raise ValueError(
                "algorithm.rlt_dvac.application must be q_gradient or "
                f"success_episode_bc, got {self.rlt_dvac_application!r}."
            )
        self.rlt_dvac_success_scale = float(self.rlt_dvac_cfg.get("success_scale", 1.0))
        if self.rlt_dvac_success_scale <= 0:
            raise ValueError("RLT DVAC success_scale must be positive.")
        if self.rlt_dvac_application == "success_episode_bc":
            if bool(self.cfg.env.train.get("auto_reset", False)):
                raise ValueError(
                    "Success-episode RLT DVAC BC currently requires "
                    "env.train.auto_reset=false."
                )
            if 2.0 * self.rlt_dvac_z_clip * self.rlt_dvac_strength > 1.0 + 1e-8:
                raise ValueError(
                    "Success-episode RLT DVAC BC requires "
                    "2 * z_clip * strength <= 1 for non-negative weights."
                )
            horizon_range = (
                2.0
                * self.rlt_dvac_z_clip
                * (self.rlt_dvac_horizon - 1)
                / max(self.rlt_dvac_horizon, 1)
            )
            self.rlt_dvac_weight_min = self.rlt_dvac_success_scale * (
                1.0 - self.rlt_dvac_strength * horizon_range
            )
            self.rlt_dvac_weight_max = self.rlt_dvac_success_scale * (
                1.0 + self.rlt_dvac_strength * horizon_range
            )
        else:
            self.rlt_dvac_weight_min = 1.0 - (
                self.rlt_dvac_strength * self.rlt_dvac_z_clip
            )
            self.rlt_dvac_weight_max = 1.0 + (
                self.rlt_dvac_strength * self.rlt_dvac_z_clip
            )
        self.rlt_dvac_stats = None
        if self.rlt_dvac_mode != "off":
            self.rlt_dvac_stats = FrozenGlobalZMoments(
                log_eps=float(self.rlt_dvac_cfg.get("log_eps", 1e-12)),
                std_floor=float(self.rlt_dvac_cfg.get("std_floor", 1e-6)),
            )
        self._rlt_dvac_last_trace_update = -1

    def _rlt_dvac_selected_variances(
        self, obs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        teacher_v = obs.get("teacher_dvac_v")
        if not isinstance(teacher_v, torch.Tensor):
            raise ValueError(
                "RLT DVAC mode requires curr_obs.teacher_dvac_v from the frozen "
                "pi0 teacher."
            )
        selected_index = self.rlt_dvac_l_values.index(self.rlt_dvac_selected_l)
        if teacher_v.shape[-2] != len(self.rlt_dvac_l_values):
            raise ValueError(
                "RLT teacher_dvac_v must store one row per configured L, got "
                f"shape={tuple(teacher_v.shape)}."
            )
        return teacher_v[..., selected_index, : self.rlt_dvac_horizon]

    def _rlt_dvac_prepare_actor_actions(
        self,
        pi: torch.Tensor,
        curr_obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict | None, dict[str, float]]:
        if self.rlt_dvac_mode == "off":
            return pi, None, {"rlt_dvac/enabled": 0.0}

        selected_v = self._rlt_dvac_selected_variances(curr_obs)
        if self.rlt_dvac_stats.frozen:
            legacy_weights, z_scores, log_variances = global_z_weights(
                selected_v,
                mean=self.rlt_dvac_stats.mean,
                std=self.rlt_dvac_stats.std,
                log_eps=self.rlt_dvac_stats.log_eps,
                std_floor=self.rlt_dvac_stats.std_floor,
                z_clip=self.rlt_dvac_z_clip,
                strength=self.rlt_dvac_strength,
            )
            candidate_weights = (
                centered_mean_one_weights(z_scores, strength=self.rlt_dvac_strength)
                * self.rlt_dvac_success_scale
                if self.rlt_dvac_application == "success_episode_bc"
                else legacy_weights
            )
        else:
            log_variances = torch.log(
                selected_v.float().clamp_min(0) + self.rlt_dvac_stats.log_eps
            )
            z_scores = torch.zeros_like(log_variances)
            candidate_weights = torch.ones_like(log_variances)

        apply_weights = self.rlt_dvac_mode == "apply" and self.rlt_dvac_stats.frozen
        weights = (
            candidate_weights if apply_weights else torch.ones_like(candidate_weights)
        )
        apply_q_weights = apply_weights and self.rlt_dvac_application == "q_gradient"
        pi_for_q = (
            straight_through_scale_actions(
                pi,
                weights,
                action_dim=int(self.cfg.actor.model.action_dim),
            )
            if apply_q_weights
            else pi
        )

        metrics = summarize_weights(
            weights,
            z_scores,
            log_variances,
            weight_min=self.rlt_dvac_weight_min,
            weight_max=self.rlt_dvac_weight_max,
        )
        metrics.update(
            {
                "rlt_dvac/enabled": 1.0,
                "rlt_dvac/mode_observe": float(self.rlt_dvac_mode == "observe"),
                "rlt_dvac/mode_apply": float(self.rlt_dvac_mode == "apply"),
                "rlt_dvac/application_q_gradient": float(
                    self.rlt_dvac_application == "q_gradient"
                ),
                "rlt_dvac/application_success_episode_bc": float(
                    self.rlt_dvac_application == "success_episode_bc"
                ),
                "rlt_dvac/baseline_frozen": float(self.rlt_dvac_stats.frozen),
                "rlt_dvac/baseline_count": float(self.rlt_dvac_stats.count),
                "rlt_dvac/baseline_mean": float(self.rlt_dvac_stats.mean),
                "rlt_dvac/baseline_std": float(self.rlt_dvac_stats.std),
                "rlt_dvac/teacher_v_mean": float(selected_v.float().mean().item()),
                "rlt_dvac/query_mean_z_std": float(
                    z_scores.float().mean(dim=-1).std(unbiased=False).item()
                ),
                "rlt_dvac/within_query_z_std": float(
                    z_scores.float().std(dim=-1, unbiased=False).mean().item()
                ),
            }
        )
        for h_index, h_weight in enumerate(weights.float().mean(dim=0)):
            metrics[f"rlt_dvac/weight_h{h_index:02d}"] = float(h_weight.item())

        payload = {
            "teacher_v": curr_obs["teacher_dvac_v"].detach(),
            "selected_v": selected_v.detach(),
            "log_variances": log_variances.detach(),
            "z_scores": z_scores.detach(),
            "weights": weights.detach(),
            "success_episode_bc_applied": bool(
                apply_weights and self.rlt_dvac_application == "success_episode_bc"
            ),
        }
        for key in ("dvac_collection_version", "actor_switch"):
            if key in curr_obs:
                payload[key] = curr_obs[key].detach()
        return pi_for_q, payload, metrics

    @staticmethod
    def _rlt_dvac_corr(left: torch.Tensor, right: torch.Tensor) -> float:
        left = left.detach().float().reshape(-1)
        right = right.detach().float().reshape(-1)
        left = left - left.mean()
        right = right - right.mean()
        denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
        if float(denominator.item()) <= 1e-12:
            return 0.0
        return float((left * right).sum().div(denominator).item())

    def _rlt_dvac_context_metrics(
        self,
        payload: dict | None,
        *,
        pi: torch.Tensor,
        ref_chunk: torch.Tensor,
        batch: dict,
        all_qf_pi: torch.Tensor,
    ) -> dict[str, float]:
        if payload is None:
            return {}
        chunk_len, action_dim = self._chunk_shape()
        pi_chunk = self._flatten_chunk(pi).reshape(-1, chunk_len, action_dim)
        ref = self._flatten_chunk(ref_chunk).reshape(-1, chunk_len, action_dim)
        student_ref_error = (pi_chunk - ref).abs().mean(dim=-1)
        metrics = {
            "rlt_dvac/weight_student_ref_error_corr": self._rlt_dvac_corr(
                payload["weights"], student_ref_error
            )
        }
        if all_qf_pi.shape[-1] >= 2:
            q_gap = (all_qf_pi[..., 0] - all_qf_pi[..., 1]).abs().reshape(-1)
            metrics["rlt_dvac/query_weight_q_gap_corr"] = self._rlt_dvac_corr(
                payload["weights"].float().mean(dim=-1), q_gap
            )

        query_weight = payload["weights"].float().mean(dim=-1)
        positive = torch.zeros_like(query_weight, dtype=torch.bool)
        rewards = batch.get("rewards")
        if isinstance(rewards, torch.Tensor):
            positive = rewards.reshape(rewards.shape[0], -1).sum(dim=-1) > 0

        actor_switch = payload.get("actor_switch")
        route_mask = torch.zeros_like(query_weight, dtype=torch.bool)
        if isinstance(actor_switch, torch.Tensor):
            route_mask = actor_switch.reshape(-1).bool()
        metrics["rlt_dvac/actor_switch_rate"] = float(route_mask.float().mean().item())

        for prefix, mask in (
            ("positive_reward", positive),
            ("nonpositive_reward", ~positive),
            ("student_route", route_mask),
            ("reference_route", ~route_mask),
        ):
            weight_sum, count = masked_weight_totals(query_weight, mask)
            metrics[f"rlt_dvac/{prefix}_weight_sum"] = weight_sum
            metrics[f"rlt_dvac/{prefix}_count"] = count

        collection_version = payload.get("dvac_collection_version")
        if isinstance(collection_version, torch.Tensor):
            replay_age = int(self.update_step) - collection_version.float().reshape(-1)
            metrics["rlt_dvac/replay_age_mean"] = float(replay_age.mean().item())
            metrics["rlt_dvac/replay_age_p95"] = float(
                torch.quantile(replay_age, 0.95).item()
            )
        return metrics

    @staticmethod
    def _finalize_rlt_dvac_context_metrics(
        metrics: dict[str, float],
    ) -> dict[str, float]:
        for prefix in (
            "positive_reward",
            "nonpositive_reward",
            "student_route",
            "reference_route",
        ):
            sum_key = f"rlt_dvac/{prefix}_weight_sum"
            count_key = f"rlt_dvac/{prefix}_count"
            if sum_key not in metrics or count_key not in metrics:
                continue
            count = float(metrics[count_key])
            metrics[f"rlt_dvac/{prefix}_weight_mean"] = (
                float(metrics[sum_key]) / count if count > 0 else 0.0
            )
        return metrics

    def _maybe_write_rlt_dvac_trace(
        self,
        payload: dict | None,
        *,
        pi: torch.Tensor,
        ref_chunk: torch.Tensor,
        curr_obs: dict[str, torch.Tensor],
        batch: dict,
    ) -> None:
        if payload is None:
            return
        interval = int(self.rlt_dvac_cfg.get("raw_trace_interval_updates", 0))
        if interval <= 0 or int(self.update_step) % interval != 0:
            return
        if self._rlt_dvac_last_trace_update == int(self.update_step):
            return
        output_dir = self.rlt_dvac_cfg.get("output_dir")
        if not output_dir:
            raise ValueError(
                "RLT DVAC raw tracing requires algorithm.rlt_dvac.output_dir."
            )
        query_count = min(
            int(self.rlt_dvac_cfg.get("raw_trace_queries_per_update", 4)),
            int(pi.shape[0]),
        )
        chunk_len, action_dim = self._chunk_shape()
        student_actions = self._flatten_chunk(pi).reshape(-1, chunk_len, action_dim)
        reference_actions = self._flatten_chunk(ref_chunk).reshape(
            -1, chunk_len, action_dim
        )
        rank_dir = os.path.join(str(output_dir), f"actor_rank{int(self._rank):02d}")
        os.makedirs(rank_dir, exist_ok=True)
        final_path = os.path.join(rank_dir, f"update_{int(self.update_step):08d}.npz")
        temp_path = f"{final_path}.partial.npz"

        arrays = {
            "update_step": np.asarray([int(self.update_step)], dtype=np.int64),
            "teacher_dvac_v": payload["teacher_v"][:query_count].float().cpu().numpy(),
            "selected_v": payload["selected_v"][:query_count].float().cpu().numpy(),
            "z_scores": payload["z_scores"][:query_count].float().cpu().numpy(),
            "weights": payload["weights"][:query_count].float().cpu().numpy(),
            "student_actions": student_actions[:query_count]
            .detach()
            .float()
            .cpu()
            .numpy(),
            "ref_chunk": reference_actions[:query_count].detach().float().cpu().numpy(),
        }
        executed_actions = self._flatten_chunk(batch["actions"]).reshape(
            -1, chunk_len, action_dim
        )
        arrays["executed_actions"] = (
            executed_actions[:query_count].detach().float().cpu().numpy()
        )
        if "episode_success" in curr_obs:
            arrays["episode_success"] = (
                curr_obs["episode_success"][:query_count].detach().cpu().numpy()
            )
        for key in ("dvac_collection_version", "actor_switch"):
            if key in curr_obs:
                arrays[key] = curr_obs[key][:query_count].cpu().numpy()
        np.savez_compressed(temp_path, **arrays)
        os.replace(temp_path, final_path)
        self._rlt_dvac_last_trace_update = int(self.update_step)

    def _accumulate_rlt_dvac_baseline(
        self, replay_trajectories: list[Trajectory]
    ) -> None:
        if self.rlt_dvac_stats is None or self.rlt_dvac_stats.frozen:
            return
        for trajectory in replay_trajectories:
            self.rlt_dvac_stats.update_variances(
                self._rlt_dvac_selected_variances(trajectory.curr_obs)
            )

    def _freeze_rlt_dvac_baseline(self) -> None:
        if self.rlt_dvac_stats is None or self.rlt_dvac_stats.frozen:
            return
        count, total, total_sq = self.rlt_dvac_stats.sufficient_statistics()
        statistics = torch.tensor(
            [float(count), float(total), float(total_sq)],
            dtype=torch.float64,
            device=self.device,
        )
        if self._distributed_active():
            torch.distributed.all_reduce(statistics, op=torch.distributed.ReduceOp.SUM)
        self.rlt_dvac_stats.freeze_from_statistics(
            int(statistics[0].item()),
            float(statistics[1].item()),
            float(statistics[2].item()),
        )

    def _rlt_dvac_baseline_metrics(self) -> dict[str, float]:
        if self.rlt_dvac_stats is None:
            return {"rlt_dvac/enabled": 0.0}
        return {
            "rlt_dvac/enabled": 1.0,
            "rlt_dvac/baseline_frozen": float(self.rlt_dvac_stats.frozen),
            "rlt_dvac/baseline_count": float(self.rlt_dvac_stats.count),
            "rlt_dvac/baseline_mean": float(self.rlt_dvac_stats.mean),
            "rlt_dvac/baseline_std": float(self.rlt_dvac_stats.std),
        }

    @staticmethod
    def _plain_config(value):
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return value

    def _select_config(self, path: str, default=None):
        return self._plain_config(OmegaConf.select(self.cfg, path, default=default))

    @staticmethod
    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _atomic_json_dump(payload: dict, path: str) -> None:
        temp_path = f"{path}.tmp.{os.getpid()}"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)

    def _rlt_resume_state_dir(self, base_path: str) -> str:
        return os.path.join(base_path, "sac_components/rlt_trainer_state")

    def _rlt_resume_state_path(self, base_path: str) -> str:
        return os.path.join(
            self._rlt_resume_state_dir(base_path),
            f"checkpoint_rank_{self._rank}.pt",
        )

    def _rlt_resume_manifest_path(self, base_path: str) -> str:
        return os.path.join(
            self._rlt_resume_state_dir(base_path),
            "rlt_trainer_state_complete.json",
        )

    @staticmethod
    def _checkpoint_runner_step(base_path: str) -> int | None:
        for part in reversed(os.path.normpath(base_path).split(os.sep)):
            match = re.fullmatch(r"global_step_(\d+)", part)
            if match is not None:
                return int(match.group(1))
        return None

    @staticmethod
    def _distributed_active() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _barrier_if_distributed(self) -> None:
        if self._distributed_active():
            torch.distributed.barrier()

    def _all_gather_object(self, value):
        if not self._distributed_active():
            return [value]
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, value)
        return gathered

    def _raise_collective_errors(self, context: str, local_error: str | None) -> None:
        errors = self._all_gather_object(
            {"rank": int(self._rank), "error": local_error}
        )
        failures = [
            f"rank {entry['rank']}: {entry['error']}"
            for entry in errors
            if entry["error"] is not None
        ]
        if failures:
            raise ValueError(f"{context} failed closed: {'; '.join(failures)}")

    @staticmethod
    def _validate_contract_value(key: str, value) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"RLT resume contract key {key!r} must be a string.")
        normalized = value.strip()
        upper = normalized.upper()
        if (
            upper.startswith("UNRESOLVED")
            or upper.startswith("PLACEHOLDER")
            or "/path/to/" in normalized.lower()
        ):
            raise ValueError(
                f"RLT resume contract key {key!r} is unresolved: {normalized!r}."
            )
        return normalized

    def _rlt_resume_contract(self) -> tuple[str, str]:
        configured_contract = self._plain_config(
            self.rlt_resume_cfg.get("contract", {})
        )
        if not isinstance(configured_contract, dict):
            raise ValueError("algorithm.rlt_resume.contract must be a mapping.")
        configured_contract = dict(configured_contract)
        for key in self._RLT_REQUIRED_CONTRACT_KEYS:
            configured_contract[key] = self._validate_contract_value(
                key, configured_contract.get(key)
            )
        for key in ("stage1_manifest_sha256", "norm_stats_sha256"):
            value = configured_contract[key]
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(
                    f"RLT resume contract key {key!r} must be lowercase SHA256."
                )

        configured_schema = int(
            self.rlt_resume_cfg.get(
                "schema_version", self._RLT_TRAINER_STATE_SCHEMA_VERSION
            )
        )
        if configured_schema != self._RLT_TRAINER_STATE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported RLT trainer-state schema version: "
                f"{configured_schema} != {self._RLT_TRAINER_STATE_SCHEMA_VERSION}."
            )

        adapter = self._select_config(
            "rollout.rlt_feature_model.openpi.rlt_action_adapter",
            default="identity",
        )
        if configured_contract["canonical_adapter_version"] != adapter:
            raise ValueError(
                "RLT resume canonical adapter mismatch between contract and "
                f"feature model: {configured_contract['canonical_adapter_version']!r} "
                f"!= {adapter!r}."
            )

        manifest_path = configured_contract["stage1_manifest_path"]
        if not os.path.isfile(manifest_path):
            raise ValueError(f"RLT Stage 1 manifest does not exist: {manifest_path}")
        manifest_sha256 = self._sha256_file(manifest_path)
        if manifest_sha256 != configured_contract["stage1_manifest_sha256"]:
            raise ValueError(
                "RLT Stage 1 manifest SHA256 mismatch: "
                f"{manifest_sha256} != "
                f"{configured_contract['stage1_manifest_sha256']}."
            )
        with open(manifest_path, encoding="utf-8") as handle:
            stage1_manifest = json.load(handle)
        if (
            stage1_manifest.get("manifest_id")
            != configured_contract["stage1_manifest_id"]
        ):
            raise ValueError(
                "RLT Stage 1 manifest ID mismatch: "
                f"{stage1_manifest.get('manifest_id')!r} != "
                f"{configured_contract['stage1_manifest_id']!r}."
            )
        if stage1_manifest.get("accepted") is not True:
            raise ValueError("RLT Stage 1 manifest is not marked accepted.")
        if int(stage1_manifest.get("schema_version", -1)) != 1:
            raise ValueError(
                "Unsupported RLT Stage 1 artifact-manifest schema: "
                f"{stage1_manifest.get('schema_version')!r}."
            )

        manifest_stage1 = stage1_manifest.get("stage1")
        if not isinstance(manifest_stage1, dict):
            raise ValueError("RLT Stage 1 manifest is missing the stage1 mapping.")
        manifest_model_path = manifest_stage1.get("model_path")
        configured_model_path = self._select_config(
            "rollout.rlt_feature_model.model_path"
        )
        if not isinstance(manifest_model_path, str) or not manifest_model_path:
            raise ValueError("RLT Stage 1 manifest has no model_path.")
        if not isinstance(configured_model_path, str) or not os.path.isdir(
            configured_model_path
        ):
            raise ValueError(
                "RLT Stage 2 requires an existing feature-model directory, got "
                f"{configured_model_path!r}."
            )
        if os.path.realpath(manifest_model_path) != os.path.realpath(
            configured_model_path
        ):
            raise ValueError(
                "RLT Stage 1 manifest/model path mismatch: "
                f"{manifest_model_path!r} != {configured_model_path!r}."
            )

        manifest_model_contract = stage1_manifest.get("model_contract")
        if not isinstance(manifest_model_contract, dict):
            raise ValueError(
                "RLT Stage 1 manifest is missing the model_contract mapping."
            )
        feature_action_chunk = int(
            self._select_config("rollout.rlt_feature_model.openpi.action_chunk")
        )
        actor_action_chunk = int(self._select_config("actor.model.num_action_chunks"))
        actor_ref_action_chunk = int(
            self._select_config("actor.model.ref_num_action_chunks")
        )
        if (
            len(
                {
                    feature_action_chunk,
                    actor_action_chunk,
                    actor_ref_action_chunk,
                }
            )
            != 1
        ):
            raise ValueError(
                "RLT Stage 2 action-chunk config mismatch: "
                f"feature={feature_action_chunk}, actor={actor_action_chunk}, "
                f"reference={actor_ref_action_chunk}."
            )
        feature_action_dim = int(
            self._select_config("rollout.rlt_feature_model.openpi.action_env_dim")
        )
        actor_action_dim = int(self._select_config("actor.model.action_dim"))
        if feature_action_dim != actor_action_dim:
            raise ValueError(
                "RLT Stage 2 action-dimension config mismatch: "
                f"{feature_action_dim} != {actor_action_dim}."
            )
        feature_z_dim = int(
            self._select_config("rollout.rlt_feature_model.openpi.rlt_embed_dim")
        )
        actor_z_dim = int(self._select_config("actor.model.z_dim"))
        if feature_z_dim != actor_z_dim:
            raise ValueError(
                "RLT Stage 2 z-dimension config mismatch: "
                f"{feature_z_dim} != {actor_z_dim}."
            )
        expected_manifest_model_contract = {
            "norm_stats_sha256": configured_contract["norm_stats_sha256"],
            "canonical_adapter_version": configured_contract[
                "canonical_adapter_version"
            ],
            "action_horizon": int(
                self._select_config("rollout.rlt_feature_model.openpi.action_horizon")
            ),
            "action_chunk": feature_action_chunk,
            "action_dim": feature_action_dim,
            "z_rl_dim": feature_z_dim,
            "image_prefix_shape": [
                int(
                    self._select_config(
                        "rollout.rlt_feature_model.openpi.rlt_prefix_seq_len"
                    )
                ),
                int(
                    self._select_config(
                        "rollout.rlt_feature_model.openpi.rlt_input_dim"
                    )
                ),
            ],
        }
        for key, expected_value in expected_manifest_model_contract.items():
            if manifest_model_contract.get(key) != expected_value:
                raise ValueError(
                    "RLT Stage 1 manifest model-contract mismatch for "
                    f"{key}: {manifest_model_contract.get(key)!r} != "
                    f"{expected_value!r}."
                )

        norm_stats_path = self._select_config(
            "rollout.rlt_feature_model.openpi_data.norm_stats_path"
        )
        if not isinstance(norm_stats_path, str) or not os.path.isfile(norm_stats_path):
            raise ValueError(
                "RLT resume requires an existing explicit norm_stats_path, got "
                f"{norm_stats_path!r}."
            )
        norm_stats_sha256 = self._sha256_file(norm_stats_path)
        if norm_stats_sha256 != configured_contract["norm_stats_sha256"]:
            raise ValueError(
                "RLT norm-stats SHA256 mismatch: "
                f"{norm_stats_sha256} != "
                f"{configured_contract['norm_stats_sha256']}."
            )

        payload = {
            "schema_version": configured_schema,
            "contract": configured_contract,
            "schedule": self._select_config("algorithm.rlt_schedule", {}),
            "route": self._select_config("algorithm.rlt_route", {}),
            "transition_replay": self._select_config(
                "algorithm.rlt_transition_replay", {}
            ),
            "replay_buffer": self._select_config("algorithm.replay_buffer", {}),
            "actor_model": self._select_config("actor.model", {}),
            "feature_model": self._select_config("rollout.rlt_feature_model", {}),
            "optimization": {
                "loss_type": self._select_config("algorithm.loss_type"),
                "agg_q": self._select_config("algorithm.agg_q"),
                "actor_agg_q": self._select_config("algorithm.actor_agg_q"),
                "bootstrap_type": self._select_config("algorithm.bootstrap_type"),
                "gamma": self._select_config("algorithm.gamma"),
                "tau": self._select_config("algorithm.tau"),
                "update_epoch": self._select_config("algorithm.update_epoch"),
                "critic_actor_ratio": self._select_config(
                    "algorithm.critic_actor_ratio"
                ),
                "target_update_freq": self._select_config(
                    "algorithm.target_update_freq"
                ),
                "target_update_type": self._select_config(
                    "algorithm.target_update_type"
                ),
                "reference_dropout_prob": self._select_config(
                    "algorithm.reference_dropout_prob"
                ),
                "q_weight": self._select_config("algorithm.q_weight"),
                "bc_weight": self._select_config("algorithm.bc_weight"),
                "actor_weight_schedule": self._select_config(
                    "algorithm.actor_weight_schedule", {}
                ),
                "actor_optim": self._select_config("actor.optim", {}),
                "critic_optim": self._select_config("actor.critic_optim", {}),
                "global_batch_size": self._select_config("actor.global_batch_size"),
                "micro_batch_size": self._select_config("actor.micro_batch_size"),
            },
            "distributed": {
                "actor_world_size": int(self._world_size),
                "weight_sync_interval": self._select_config(
                    "runner.weight_sync_interval"
                ),
                "weight_syncer": self._select_config("weight_syncer", {}),
            },
        }
        if self.rlt_dvac_mode != "off":
            payload["optimization"]["rlt_dvac"] = self._select_config(
                "algorithm.rlt_dvac", {}
            )
        contract_json = json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        contract_sha256 = hashlib.sha256(contract_json.encode("utf-8")).hexdigest()
        return contract_json, contract_sha256

    def _validate_rlt_trainer_state_payload(self, state) -> dict:
        if not isinstance(state, dict):
            raise TypeError(
                f"RLT trainer state must be a mapping, got {type(state).__name__}."
            )
        missing = [key for key in self._RLT_REQUIRED_STATE_KEYS if key not in state]
        if missing:
            raise ValueError(f"RLT trainer state is missing keys: {missing}.")
        if int(state["schema_version"]) != self._RLT_TRAINER_STATE_SCHEMA_VERSION:
            raise ValueError(
                "RLT trainer-state schema mismatch: "
                f"{state['schema_version']} != "
                f"{self._RLT_TRAINER_STATE_SCHEMA_VERSION}."
            )
        if int(state["rank"]) != int(self._rank):
            raise ValueError(
                f"RLT trainer-state rank mismatch: {state['rank']} != {self._rank}."
            )
        if int(state["actor_world_size"]) != int(self._world_size):
            raise ValueError(
                "RLT trainer-state world-size mismatch: "
                f"{state['actor_world_size']} != {self._world_size}."
            )
        for key in (
            "saved_runner_step",
            "update_step",
            "local_total_transitions_added",
            "local_total_episodes_added",
        ):
            if int(state[key]) < 0:
                raise ValueError(f"RLT trainer-state {key} must be nonnegative.")
        for key in (
            "global_warmup_ready_total_transitions",
            "global_warmup_ready_total_episodes",
        ):
            if state[key] is not None and int(state[key]) < 0:
                raise ValueError(
                    f"RLT trainer-state {key} must be null or nonnegative."
                )
        contract_json = state["rlt_resume_contract"]
        contract_sha256 = state["rlt_resume_contract_sha256"]
        if not isinstance(contract_json, str) or not isinstance(contract_sha256, str):
            raise TypeError("RLT trainer-state contract fields must be strings.")
        actual_contract_sha256 = hashlib.sha256(
            contract_json.encode("utf-8")
        ).hexdigest()
        if actual_contract_sha256 != contract_sha256:
            raise ValueError(
                "RLT trainer-state embedded contract SHA256 mismatch: "
                f"{actual_contract_sha256} != {contract_sha256}."
            )
        if self.rlt_dvac_mode != "off" and not isinstance(
            state.get("rlt_dvac_baseline"), dict
        ):
            raise ValueError(
                "RLT DVAC trainer state is missing the frozen baseline payload."
            )
        return state

    def _rlt_trainer_state_payload(self, runner_step: int) -> dict:
        contract_json, contract_sha256 = self._rlt_resume_contract()
        return {
            "schema_version": self._RLT_TRAINER_STATE_SCHEMA_VERSION,
            "rank": int(self._rank),
            "saved_runner_step": int(runner_step),
            "actor_world_size": int(self._world_size),
            "rlt_resume_contract": contract_json,
            "rlt_resume_contract_sha256": contract_sha256,
            "update_step": int(self.update_step),
            "local_total_transitions_added": int(self.total_transitions_added),
            "local_total_episodes_added": int(self.total_episodes_added),
            "global_warmup_ready_total_transitions": (
                None
                if self._warmup_ready_total_transitions is None
                else int(self._warmup_ready_total_transitions)
            ),
            "global_warmup_ready_total_episodes": (
                None
                if self._warmup_ready_total_episodes is None
                else int(self._warmup_ready_total_episodes)
            ),
            "rlt_dvac_baseline": (
                None
                if self.rlt_dvac_stats is None
                else self.rlt_dvac_stats.state_dict()
            ),
        }

    def _mark_rlt_checkpoint_incomplete(
        self, save_base_path: str, runner_step: int
    ) -> None:
        local_error = None
        if int(self._rank) == 0:
            try:
                state_dir = self._rlt_resume_state_dir(save_base_path)
                os.makedirs(state_dir, exist_ok=True)
                self._atomic_json_dump(
                    {
                        "complete": False,
                        "schema_version": self._RLT_TRAINER_STATE_SCHEMA_VERSION,
                        "actor_world_size": int(self._world_size),
                        "saved_runner_step": int(runner_step),
                        "update_step": int(self.update_step),
                    },
                    self._rlt_resume_manifest_path(save_base_path),
                )
            except Exception as exc:  # pragma: no cover - distributed filesystem
                local_error = f"{type(exc).__name__}: {exc}"
        self._raise_collective_errors(
            "RLT trainer-state invalidation marker", local_error
        )
        self._barrier_if_distributed()

    def _save_rlt_trainer_state(self, save_base_path: str, runner_step: int) -> None:
        path_runner_step = self._checkpoint_runner_step(save_base_path)
        if path_runner_step is not None and path_runner_step != int(runner_step):
            raise ValueError(
                "RLT trainer-state save path/step mismatch: "
                f"{path_runner_step} != {runner_step}."
            )
        state_dir = self._rlt_resume_state_dir(save_base_path)
        state_path = self._rlt_resume_state_path(save_base_path)
        local_error = None
        local_metadata = None
        try:
            os.makedirs(state_dir, exist_ok=True)
            state = self._rlt_trainer_state_payload(runner_step)
            temp_path = f"{state_path}.tmp.{os.getpid()}"
            torch.save(state, temp_path)
            os.replace(temp_path, state_path)
            local_metadata = {
                "rank": int(self._rank),
                "path": os.path.basename(state_path),
                "sha256": self._sha256_file(state_path),
                "saved_runner_step": int(runner_step),
                "update_step": int(state["update_step"]),
                "warmup_transitions": state["global_warmup_ready_total_transitions"],
                "warmup_episodes": state["global_warmup_ready_total_episodes"],
                "contract_sha256": state["rlt_resume_contract_sha256"],
            }
        except Exception as exc:  # pragma: no cover - exercised by distributed smoke
            local_error = f"{type(exc).__name__}: {exc}"
        self._raise_collective_errors("RLT trainer-state save", local_error)

        metadata = self._all_gather_object(local_metadata)
        ranks = sorted(int(entry["rank"]) for entry in metadata)
        expected_ranks = list(range(int(self._world_size)))
        if ranks != expected_ranks:
            raise ValueError(
                f"RLT trainer-state save rank set mismatch: {ranks} != "
                f"{expected_ranks}."
            )
        common_fields = (
            "saved_runner_step",
            "update_step",
            "warmup_transitions",
            "warmup_episodes",
            "contract_sha256",
        )
        for key in common_fields:
            values = {entry[key] for entry in metadata}
            if len(values) != 1:
                raise ValueError(
                    f"RLT trainer-state save mismatch for {key}: {values}."
                )

        self._barrier_if_distributed()
        manifest_error = None
        if int(self._rank) == 0:
            try:
                manifest = {
                    "complete": True,
                    "schema_version": self._RLT_TRAINER_STATE_SCHEMA_VERSION,
                    "actor_world_size": int(self._world_size),
                    "saved_runner_step": int(runner_step),
                    "update_step": int(self.update_step),
                    "rlt_resume_contract_sha256": metadata[0]["contract_sha256"],
                    "files": sorted(metadata, key=lambda item: int(item["rank"])),
                }
                self._atomic_json_dump(
                    manifest, self._rlt_resume_manifest_path(save_base_path)
                )
            except Exception as exc:  # pragma: no cover - distributed filesystem
                manifest_error = f"{type(exc).__name__}: {exc}"
        self._raise_collective_errors(
            "RLT trainer-state completion manifest", manifest_error
        )
        self._barrier_if_distributed()

    def _preflight_rlt_trainer_state(self, load_base_path: str) -> dict:
        state_path = self._rlt_resume_state_path(load_base_path)
        manifest_path = self._rlt_resume_manifest_path(load_base_path)
        local_error = None
        state = None
        try:
            if not os.path.isfile(manifest_path):
                raise FileNotFoundError(
                    f"missing RLT completion manifest: {manifest_path}"
                )
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            if not manifest.get("complete", False):
                raise ValueError("RLT completion manifest is not marked complete.")
            if int(manifest.get("schema_version", -1)) != int(
                self._RLT_TRAINER_STATE_SCHEMA_VERSION
            ):
                raise ValueError(
                    "RLT completion manifest schema mismatch: "
                    f"{manifest.get('schema_version')} != "
                    f"{self._RLT_TRAINER_STATE_SCHEMA_VERSION}."
                )
            if int(manifest.get("actor_world_size", -1)) != int(self._world_size):
                raise ValueError(
                    "RLT completion manifest world-size mismatch: "
                    f"{manifest.get('actor_world_size')} != {self._world_size}."
                )
            path_runner_step = self._checkpoint_runner_step(load_base_path)
            manifest_runner_step = int(manifest.get("saved_runner_step", -1))
            if (
                path_runner_step is not None
                and manifest_runner_step != path_runner_step
            ):
                raise ValueError(
                    "RLT completion manifest path/step mismatch: "
                    f"{manifest_runner_step} != {path_runner_step}."
                )
            if not os.path.isfile(state_path):
                raise FileNotFoundError(f"missing RLT rank state: {state_path}")
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            state = self._validate_rlt_trainer_state_payload(state)
            contract_json, contract_sha256 = self._rlt_resume_contract()
            if (
                state.get("rlt_resume_contract") != contract_json
                or state.get("rlt_resume_contract_sha256") != contract_sha256
            ):
                raise ValueError("RLT resume contract fingerprint mismatch.")
            raw_manifest_files = manifest.get("files", [])
            if not isinstance(raw_manifest_files, list):
                raise ValueError("RLT completion manifest files must be a list.")
            manifest_files = {int(entry["rank"]): entry for entry in raw_manifest_files}
            if len(manifest_files) != len(raw_manifest_files):
                raise ValueError("RLT completion manifest contains duplicate ranks.")
            if sorted(manifest_files) != list(range(int(self._world_size))):
                raise ValueError(
                    "RLT completion manifest rank set is incomplete: "
                    f"{sorted(manifest_files)}."
                )
            file_entry = manifest_files[int(self._rank)]
            if file_entry.get("path") != os.path.basename(state_path):
                raise ValueError("RLT completion manifest path mismatch.")
            if file_entry.get("sha256") != self._sha256_file(state_path):
                raise ValueError("RLT trainer-state file SHA256 mismatch.")
            metadata_state_pairs = (
                ("saved_runner_step", "saved_runner_step"),
                ("update_step", "update_step"),
                (
                    "warmup_transitions",
                    "global_warmup_ready_total_transitions",
                ),
                ("warmup_episodes", "global_warmup_ready_total_episodes"),
                ("contract_sha256", "rlt_resume_contract_sha256"),
            )
            for metadata_key, state_key in metadata_state_pairs:
                if file_entry.get(metadata_key) != state[state_key]:
                    raise ValueError(
                        "RLT completion manifest file metadata mismatch for "
                        f"{metadata_key}."
                    )
            if manifest_runner_step != int(state["saved_runner_step"]):
                raise ValueError("RLT completion manifest/state runner-step mismatch.")
            if int(manifest.get("update_step", -1)) != int(state["update_step"]):
                raise ValueError("RLT completion manifest/state update-step mismatch.")
            if manifest.get("rlt_resume_contract_sha256") != contract_sha256:
                raise ValueError("RLT completion manifest contract mismatch.")
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        self._raise_collective_errors("RLT trainer-state preflight", local_error)
        return state

    def _restore_rlt_trainer_state(self, state: dict) -> None:
        local_error = None
        summary = None
        try:
            state = self._validate_rlt_trainer_state_payload(state)
            summary = {
                "rank": int(self._rank),
                "saved_runner_step": int(state["saved_runner_step"]),
                "update_step": int(state["update_step"]),
                "warmup_transitions": state["global_warmup_ready_total_transitions"],
                "warmup_episodes": state["global_warmup_ready_total_episodes"],
                "contract_sha256": state["rlt_resume_contract_sha256"],
            }
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        self._raise_collective_errors(
            "RLT trainer-state restore validation", local_error
        )
        summaries = self._all_gather_object(summary)
        if sorted(int(entry["rank"]) for entry in summaries) != list(
            range(int(self._world_size))
        ):
            raise ValueError("RLT trainer-state restore rank set mismatch.")
        for key in (
            "saved_runner_step",
            "update_step",
            "warmup_transitions",
            "warmup_episodes",
            "contract_sha256",
        ):
            values = {entry[key] for entry in summaries}
            if len(values) != 1:
                raise ValueError(
                    f"RLT trainer-state restore mismatch for {key}: {values}."
                )

        self.update_step = int(state["update_step"])
        self.total_transitions_added = int(state["local_total_transitions_added"])
        self.total_episodes_added = int(state["local_total_episodes_added"])
        self._warmup_ready_total_transitions = state[
            "global_warmup_ready_total_transitions"
        ]
        self._warmup_ready_total_episodes = state["global_warmup_ready_total_episodes"]
        if self.rlt_dvac_stats is not None:
            self.rlt_dvac_stats.load_state_dict(state["rlt_dvac_baseline"])
        self.transitions_since_train = 0
        self.episodes_since_train = 0
        self.pending_update_budget = 0

    def save_checkpoint(self, save_base_path, step):
        if self.use_rlt_resume:
            path_runner_step = self._checkpoint_runner_step(save_base_path)
            if path_runner_step is not None and path_runner_step != int(step):
                raise ValueError(
                    "RLT checkpoint save path/step mismatch before base save: "
                    f"{path_runner_step} != {step}."
                )
            self._mark_rlt_checkpoint_incomplete(save_base_path, step)
        super().save_checkpoint(save_base_path, step)
        if self.use_rlt_resume:
            self._save_rlt_trainer_state(save_base_path, step)

    def load_checkpoint(self, load_base_path):
        state = None
        if self.use_rlt_resume:
            state = self._preflight_rlt_trainer_state(load_base_path)
        super().load_checkpoint(load_base_path)
        if self.use_rlt_resume:
            self._restore_rlt_trainer_state(state)

    def setup_sac_components(self):
        """Initialize replay components and let RLT schedule own readiness."""
        super().setup_sac_components()
        if self.use_rlt_schedule:
            self.buffer_dataset.min_replay_buffer_size = 1

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel):
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        added, completed = self._ingest_rollout_trajectories(recv_list)
        self._update_rollout_ingest_counters(added, completed)

    def _global_rlt_counters(self) -> dict[str, float]:
        summed = all_reduce_dict(
            {
                "transitions_since_train": float(self.transitions_since_train),
                "episodes_since_train": float(self.episodes_since_train),
                "total_transitions_added": float(self.total_transitions_added),
                "total_episodes_added": float(self.total_episodes_added),
            },
            op=torch.distributed.ReduceOp.SUM,
        )
        minimums = all_reduce_dict(
            {
                "min_replay_size": float(self.replay_buffer.total_samples),
                "min_demo_size": float(
                    0 if self.demo_buffer is None else self.demo_buffer.total_samples
                ),
            },
            op=torch.distributed.ReduceOp.MIN,
        )
        summed.update(minimums)
        return summed

    def _rlt_updates_to_run(self) -> tuple[int, dict[str, float]]:
        replay_cfg = self.cfg.algorithm.replay_buffer
        schedule_cfg = self.rlt_schedule_cfg
        min_buffer_size = int(
            schedule_cfg.get("warmup_min_size", replay_cfg.get("min_buffer_size", 1))
        )
        counters = self._global_rlt_counters()
        buffer_ready = counters["min_replay_size"] >= min_buffer_size
        if buffer_ready:
            self._freeze_rlt_dvac_baseline()
        warmup_required_updates = int(
            schedule_cfg.get("warmup_post_collect_updates", 0)
        )
        if buffer_ready and self._warmup_ready_total_transitions is None:
            self._warmup_ready_total_transitions = int(
                counters["total_transitions_added"]
            )
            self._warmup_ready_total_episodes = int(counters["total_episodes_added"])

        train_every_transitions = int(schedule_cfg.get("train_every_transitions", 0))
        train_every_episodes = int(schedule_cfg.get("train_every_episodes", 0))
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))
        max_updates = int(schedule_cfg.get("max_updates_per_train_step", 0))

        updates_to_run = 0
        skip_reason = 0
        desired_total_updates = 0
        pending_updates = 0
        updates_scheduled = 0
        if update_epoch <= 0:
            skip_reason = 3
        elif not buffer_ready:
            skip_reason = 1
        else:
            online_transitions = max(
                int(counters["total_transitions_added"])
                - int(self._warmup_ready_total_transitions or 0),
                0,
            )
            online_episodes = max(
                int(counters["total_episodes_added"])
                - int(self._warmup_ready_total_episodes or 0),
                0,
            )
            if train_every_transitions <= 0 and train_every_episodes <= 0:
                online_cycles = online_transitions
            else:
                transition_cycles = (
                    online_transitions // train_every_transitions
                    if train_every_transitions > 0
                    else 0
                )
                episode_cycles = (
                    online_episodes // train_every_episodes
                    if train_every_episodes > 0
                    else 0
                )
                online_cycles = max(transition_cycles, episode_cycles)
            desired_total_updates = (
                warmup_required_updates + online_cycles * update_epoch
            )
            pending_updates = max(desired_total_updates - int(self.update_step), 0)
            updates_scheduled = pending_updates
            updates_to_run = pending_updates
            if max_updates > 0:
                updates_to_run = min(updates_to_run, max_updates)
            if updates_to_run <= 0:
                skip_reason = 2
        self.pending_update_budget = int(pending_updates)

        metrics = {
            "rlt/update_step": float(self.update_step),
            "rlt/ready_for_online": float(
                int(self.update_step) >= warmup_required_updates
            ),
            "rlt/warmup_required_updates": float(warmup_required_updates),
            "rlt/update_epoch": float(update_epoch),
            "rlt/max_updates_per_train_step": float(max_updates),
            "rlt/train_every_transitions": float(train_every_transitions),
            "rlt/train_every_episodes": float(train_every_episodes),
            "rlt/desired_total_updates": float(desired_total_updates),
            "rlt/pending_update_budget": float(self.pending_update_budget),
            "rlt/updates_scheduled": float(updates_scheduled),
            "rlt/updates_to_run": float(updates_to_run),
            "rlt/critic_updates_run": 0.0,
            "rlt/actor_updates_run": 0.0,
            "rlt/should_train": float(updates_to_run > 0),
            "rlt/skip_reason": float(skip_reason),
            "rlt/global_min_replay_size": float(counters["min_replay_size"]),
            "rlt/min_replay_buffer_size": float(min_buffer_size),
            "rlt/global_transitions_since_train": float(
                counters["transitions_since_train"]
            ),
            "rlt/global_total_transitions_added": float(
                counters["total_transitions_added"]
            ),
        }
        metrics.update(getattr(self, "_last_replay_metrics", {}))
        metrics.update(self._rlt_dvac_baseline_metrics())
        return updates_to_run, metrics

    def run_training(self):
        if not self.use_rlt_schedule:
            mean_metric_dict = super().run_training()
            replay_metrics = getattr(self, "_last_replay_metrics", {})
            if replay_metrics:
                mean_metric_dict = {**mean_metric_dict, **replay_metrics}
            return mean_metric_dict

        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        updates_to_run, schedule_metrics = self._rlt_updates_to_run()
        if updates_to_run <= 0:
            mean_metric_dict = self.process_train_metrics(schedule_metrics)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            torch.cuda.empty_cache()
            return mean_metric_dict

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}
        critic_updates_run = 0
        actor_updates_run = 0
        for _ in range(updates_to_run):
            update_actor = int(self.update_step) % int(self.critic_actor_ratio) == 0
            metrics_data = self.update_one_epoch(train_actor=True)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1
            critic_updates_run += 1
            actor_updates_run += int(update_actor)

        schedule_metrics["rlt/critic_updates_run"] = float(critic_updates_run)
        schedule_metrics["rlt/actor_updates_run"] = float(actor_updates_run)
        self.pending_update_budget = max(
            int(self.pending_update_budget) - critic_updates_run,
            0,
        )
        schedule_metrics["rlt/pending_update_budget"] = float(
            self.pending_update_budget
        )
        append_to_dict(metrics, schedule_metrics)
        mean_metric_dict = self.process_train_metrics(metrics)
        mean_metric_dict = self._finalize_rlt_dvac_context_metrics(mean_metric_dict)
        self.transitions_since_train = 0
        self.episodes_since_train = 0

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict


class AsyncRLTACFSDPPolicy(
    RLTACLossMixin, RLTACReplayMixin, AsyncEmbodiedSACFSDPPolicy
):
    def __init__(self, cfg):
        dvac_cfg = cfg.algorithm.get("rlt_dvac", {}) or {}
        dvac_mode = str(dvac_cfg.get("mode", "off")).lower()
        if dvac_mode != "off":
            raise NotImplementedError(
                "RLT teacher-DVAC weighting currently supports the synchronous "
                "RLTACFSDPPolicy path only."
            )
        super().__init__(cfg)
        self.rlt_schedule_cfg = cfg.algorithm.get("rlt_schedule", {}) or {}
        self.use_rlt_schedule = bool(self.rlt_schedule_cfg.get("enable", False))

    def _drain_received_trajectories(self, max_trajectories: int | None = None):
        if getattr(self, "_recv_queue", None) is None:
            return
        recv_list = []
        processed = 0
        while True:
            try:
                recv_list.append(self._recv_queue.get_nowait())
                processed += 1
                if max_trajectories is not None and processed >= max_trajectories:
                    break
            except queue.Empty:
                break
        if not recv_list:
            return

        added, completed = self._ingest_rollout_trajectories(recv_list)
        self._update_rollout_ingest_counters(added, completed)

    async def run_training(self):
        mean_metric_dict = await super().run_training()
        replay_metrics = getattr(self, "_last_replay_metrics", {})
        if replay_metrics:
            mean_metric_dict = {**mean_metric_dict, **replay_metrics}
        return mean_metric_dict
