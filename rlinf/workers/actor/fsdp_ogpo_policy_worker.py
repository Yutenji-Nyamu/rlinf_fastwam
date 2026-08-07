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

"""Actor-first OGPO+CA worker for pi0 RoboTwin primitive replay."""

from __future__ import annotations

import copy
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.algorithms.ogpo.core import (
    clipped_ppo_loss,
    conservative_group_advantages,
    h_step_td_target,
)
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.ogpo_replay import (
    OGPOPrimitiveRow,
    OGPOReplayBuffer,
    OGPOSequenceBatch,
    OGPOSuccessBatch,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.ogpo_critic import OGPOCriticEnsemble
from rlinf.scheduler import Channel, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def _slice_env_obs(env_obs: dict[str, Any], start: int, stop: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in env_obs.items():
        if torch.is_tensor(value):
            sliced[key] = value[start:stop]
        elif isinstance(value, list):
            sliced[key] = value[start:stop]
        else:
            sliced[key] = value
    return sliced


class EmbodiedOGPOFSDPPolicy(EmbodiedFSDPActor):
    """Online pi0 actor, EMA actor, FP32 10Q, and rank-local online replay."""

    _SIDECAR_VERSION = 3

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.ogpo_cfg = cfg.algorithm.ogpo
        if float(self.ogpo_cfg.utd_q) != float(self.ogpo_cfg.utd_pi):
            raise ValueError("OGPO v1 requires paired utd_q == utd_pi")
        capacity_per_rank = math.ceil(
            int(self.ogpo_cfg.replay_capacity) / self._world_size
        )
        self.replay = OGPOReplayBuffer(
            capacity=capacity_per_rank,
            max_sequence_length=int(self.ogpo_cfg.execution_horizon),
            action_dim=int(self.ogpo_cfg.active_action_dim),
            model_action_dim=int(self.ogpo_cfg.model_action_dim),
            seed=int(cfg.actor.seed) + self._rank,
        )
        self.critic: OGPOCriticEnsemble | None = None
        self.target_critic: OGPOCriticEnsemble | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None
        self.critic_feature_dim: int | None = None
        self.runner_global_step = 0
        self.policy_version = 0
        self.actor_updates = 0
        self.critic_updates = 0
        self.global_online_rows = 0
        self.pending_actor_updates = 0.0
        self.pending_critic_updates = 0.0
        self._episode_counter = 0
        self._last_ingest_metrics: dict[str, float] = {}

    def init_worker(self) -> None:
        self.setup_model_and_optimizer()
        target_names = [
            name
            for name in self.get_rollout_state_dict()
            if name.startswith("ogpo_target.") or ".ogpo_target." in name
        ]
        if not target_names:
            raise RuntimeError("OGPO EMA target produced no rollout-sync state")
        self.param_names_need_sync = sorted(
            set(self.param_names_need_sync).union(target_names)
        )
        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

    def set_global_step(self, global_step: int) -> None:
        self.runner_global_step = int(global_step)
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def get_rollout_sync_version(self) -> int:
        return int(self.policy_version)

    @staticmethod
    def _decode_prompt(prompt_bytes: torch.Tensor, prompt_length: torch.Tensor) -> str:
        length = int(prompt_length.item())
        raw = bytes(prompt_bytes[:length].to(torch.uint8).tolist())
        return raw.decode("utf-8")

    @staticmethod
    def _batch_to_env_obs(observation: dict[str, torch.Tensor]) -> dict[str, Any]:
        prompts = [
            EmbodiedOGPOFSDPPolicy._decode_prompt(prompt, length)
            for prompt, length in zip(
                observation["prompt_utf8"], observation["prompt_length"]
            )
        ]
        return {
            "main_images": observation["main_images"],
            "wrist_images": observation.get("wrist_images"),
            "extra_view_images": None,
            "states": observation["states"],
            "task_descriptions": prompts,
        }

    def _new_episode_id(self) -> int:
        episode_id = (int(self._rank) << 48) + self._episode_counter
        self._episode_counter += 1
        return episode_id

    def _ingest_trajectory(
        self, trajectory: Trajectory, *, max_rows: int | None = None
    ) -> int:
        forward = trajectory.forward_inputs
        required = {
            "ogpo_obs_main_images",
            "ogpo_obs_states",
            "ogpo_action_model",
            "ogpo_action_q",
            "ogpo_prompt_utf8",
            "ogpo_prompt_length",
            "ogpo_rewards",
            "ogpo_terminations",
            "ogpo_truncations",
            "ogpo_valid",
        }
        missing = sorted(required - forward.keys())
        if missing:
            raise ValueError(f"OGPO trajectory is missing fields: {missing}")
        valid = forward["ogpo_valid"].to(torch.bool)
        if valid.ndim != 3:
            raise ValueError("ogpo_valid must have shape [chunk_step, env, C]")
        chunk_steps, env_count, chunk_size = valid.shape
        if chunk_size != int(self.ogpo_cfg.execution_horizon):
            raise ValueError("OGPO primitive trace has the wrong execution horizon")

        inserted = 0
        for env_index in range(env_count):
            episode_id = self._new_episode_id()
            episode_step = 0
            episode_success = False
            episode_open = False
            for chunk_index in range(chunk_steps):
                prompt_bytes = forward["ogpo_prompt_utf8"][chunk_index, env_index]
                prompt_length = forward["ogpo_prompt_length"][chunk_index, env_index]
                version = (
                    int(trajectory.versions[chunk_index, env_index].reshape(-1)[0])
                    if trajectory.versions is not None
                    else self.policy_version
                )
                for primitive_index in range(chunk_size):
                    if max_rows is not None and inserted >= max_rows:
                        return inserted
                    if not bool(valid[chunk_index, env_index, primitive_index]):
                        continue
                    episode_open = True
                    observation = {
                        "main_images": forward["ogpo_obs_main_images"][
                            chunk_index, env_index, primitive_index
                        ],
                        "states": forward["ogpo_obs_states"][
                            chunk_index, env_index, primitive_index
                        ],
                        "prompt_utf8": prompt_bytes,
                        "prompt_length": prompt_length,
                    }
                    next_observation = {
                        "main_images": forward["ogpo_obs_main_images"][
                            chunk_index, env_index, primitive_index + 1
                        ],
                        "states": forward["ogpo_obs_states"][
                            chunk_index, env_index, primitive_index + 1
                        ],
                        "prompt_utf8": prompt_bytes,
                        "prompt_length": prompt_length,
                    }
                    if "ogpo_obs_wrist_images" in forward:
                        observation["wrist_images"] = forward[
                            "ogpo_obs_wrist_images"
                        ][chunk_index, env_index, primitive_index]
                        next_observation["wrist_images"] = forward[
                            "ogpo_obs_wrist_images"
                        ][chunk_index, env_index, primitive_index + 1]
                    reward = float(
                        forward["ogpo_rewards"][
                            chunk_index, env_index, primitive_index
                        ].item()
                    )
                    terminated = bool(
                        forward["ogpo_terminations"][
                            chunk_index, env_index, primitive_index
                        ]
                    )
                    truncated = bool(
                        forward["ogpo_truncations"][
                            chunk_index, env_index, primitive_index
                        ]
                    )
                    episode_success = episode_success or terminated or reward > 0.0
                    self.replay.add(
                        OGPOPrimitiveRow(
                            observation=observation,
                            next_observation=next_observation,
                            action_model=forward["ogpo_action_model"][
                                chunk_index, env_index, primitive_index
                            ],
                            action=forward["ogpo_action_q"][
                                chunk_index, env_index, primitive_index
                            ],
                            reward=reward,
                            terminated=terminated,
                            truncated=truncated,
                            episode_id=episode_id,
                            step_id=episode_step,
                            policy_version=version,
                            source_fingerprint=str(
                                self.ogpo_cfg.source_fingerprint
                            ),
                        )
                    )
                    inserted += 1
                    episode_step += 1
                    if terminated or truncated:
                        if episode_success:
                            self.replay.mark_episode_success(episode_id)
                        episode_id = self._new_episode_id()
                        episode_step = 0
                        episode_success = False
                        episode_open = False
            if episode_open:
                # EnvWorker resets before the next returned trajectory, so a
                # partial final episode cannot share an ID with a later batch.
                episode_id = self._new_episode_id()
        return inserted

    @Worker.timer("actor/recv_ogpo_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        clear_memory(sync=False)
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        local_inserted = 0
        remaining = max(
            0,
            int(self.ogpo_cfg.total_online_rows) - self.global_online_rows,
        )
        local_quota = remaining // self._world_size + int(
            self._rank < remaining % self._world_size
        )
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            local_inserted += self._ingest_trajectory(
                trajectory,
                max_rows=max(0, local_quota - local_inserted),
            )

        inserted_tensor = torch.tensor(local_inserted, device=self.device, dtype=torch.long)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(inserted_tensor, op=torch.distributed.ReduceOp.SUM)
        global_inserted = int(inserted_tensor.item())
        previous_rows = self.global_online_rows
        self.global_online_rows += global_inserted
        start_rows = int(self.ogpo_cfg.start_training_rows)
        new_training_rows = max(0, self.global_online_rows - start_rows) - max(
            0, previous_rows - start_rows
        )
        self.pending_actor_updates += new_training_rows * float(self.ogpo_cfg.utd_pi)
        self.pending_critic_updates += new_training_rows * float(self.ogpo_cfg.utd_q)
        self._last_ingest_metrics = {
            "ogpo/local_inserted_rows": float(local_inserted),
            "ogpo/global_inserted_rows": float(global_inserted),
            "ogpo/previous_online_rows": float(previous_rows),
            "ogpo/total_online_rows": float(self.global_online_rows),
            "ogpo/replay_rows": float(len(self.replay)),
            "ogpo/success_rows": float(self.replay.success_size),
        }

    def compute_advantages_and_returns(self) -> dict[str, float]:
        return dict(self._last_ingest_metrics)

    def _init_critic(self, feature_dim: int) -> None:
        if self.critic is not None:
            if feature_dim != self.critic_feature_dim:
                raise ValueError("OGPO critic feature width changed within a run")
            return
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(self.cfg.actor.seed))
            critic = OGPOCriticEnsemble(
                feature_dim=feature_dim,
                proprio_dim=int(self.ogpo_cfg.active_action_dim),
                action_horizon=int(self.ogpo_cfg.execution_horizon),
                action_dim=int(self.ogpo_cfg.active_action_dim),
                num_q_heads=int(self.ogpo_cfg.num_q_heads),
                hidden_dims=tuple(int(v) for v in self.ogpo_cfg.critic_hidden_dims),
            )
        critic = critic.to(device=self.device, dtype=torch.float32)
        if torch.distributed.is_initialized():
            for parameter in critic.parameters():
                torch.distributed.broadcast(parameter.data, src=0)
        self.critic = critic
        self.target_critic = copy.deepcopy(critic).eval().requires_grad_(False)
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=float(self.ogpo_cfg.critic_lr),
            betas=(
                float(self.ogpo_cfg.critic_adam_beta1),
                float(self.ogpo_cfg.critic_adam_beta2),
            ),
            eps=float(self.ogpo_cfg.critic_adam_eps),
            weight_decay=float(self.ogpo_cfg.critic_weight_decay),
        )
        self.critic_feature_dim = int(feature_dim)

    @torch.no_grad()
    def _target_action(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        output = self.model(
            forward_type=ForwardType.OGPO_FLOW,
            operation="target_action",
            env_obs=env_obs,
            group_size=1,
        )
        self._init_critic(int(output["critic_feature"].shape[-1]))
        return output

    def _actor_update(
        self,
        sequence: OGPOSequenceBatch,
        success: OGPOSuccessBatch | None,
    ) -> dict[str, float]:
        assert self.target_critic is not None
        env_obs = self._batch_to_env_obs(sequence.observation)
        group_size = int(self.ogpo_cfg.candidate_group_size)
        micro_states = max(
            1, int(self.ogpo_cfg.candidate_microbatch_per_rank) // group_size
        )
        local_batch = sequence.h.shape[0]
        ppo_ranges = [
            (start, min(start + micro_states, local_batch))
            for start in range(0, local_batch, micro_states)
        ]

        local_has_success = success is not None
        active_success_ranks = torch.tensor(
            int(local_has_success), device=self.device, dtype=torch.long
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                active_success_ranks, op=torch.distributed.ReduceOp.SUM
            )
        success_rank_count = int(active_success_ranks.item())

        success_obs = None
        success_action_model = None
        success_ranges: list[tuple[int, int]] = []
        if success_rank_count > 0:
            if success is None:
                # FSDP ranks must execute the same forward/backward graph.  A
                # rank without success data uses a zero-weight ordinary replay
                # payload solely to participate in collectives; it contributes
                # no BC gradient.
                success_obs = self._batch_to_env_obs(sequence.observation)
                success_action_model = sequence.action_model
            else:
                success_obs = self._batch_to_env_obs(success.observation)
                success_action_model = success.action_model
            success_ranges = [
                (start, min(start + micro_states, local_batch))
                for start in range(0, local_batch, micro_states)
            ]
        backward_count = len(ppo_ranges) + len(success_ranges)
        backward_index = 0
        self.optimizer.zero_grad(set_to_none=True)
        ppo_losses = []
        bc_losses = []
        ratios = []

        for start, stop in ppo_ranges:
            output = self.model(
                forward_type=ForwardType.OGPO_FLOW,
                operation="actor_batch",
                env_obs=_slice_env_obs(env_obs, start, stop),
                group_size=group_size,
            )
            batch_size = stop - start
            with torch.no_grad():
                feature = output["critic_feature"].to(self.device, torch.float32)
                proprio = output["state"][:, : int(self.ogpo_cfg.active_action_dim)]
                actions = output["canonical_action"].to(self.device, torch.float32)
                q_values = self.target_critic(
                    feature[:, None].expand(-1, group_size, -1, -1).reshape(
                        batch_size * group_size, feature.shape[1], feature.shape[2]
                    ),
                    proprio[:, None].expand(-1, group_size, -1).reshape(
                        batch_size * group_size, -1
                    ),
                    actions.reshape(
                        batch_size * group_size,
                        actions.shape[-2],
                        actions.shape[-1],
                    ),
                ).reshape(batch_size, group_size, -1)
                advantages = conservative_group_advantages(q_values)
            loss = clipped_ppo_loss(
                output["current_chain_score"],
                output["old_chain_score"],
                advantages,
                clip_epsilon=float(self.ogpo_cfg.clip_epsilon),
            )
            scaled_loss = loss * (batch_size / local_batch)
            backward_index += 1
            context = self.before_micro_batch(
                self.model, is_last_micro_batch=backward_index == backward_count
            )
            with context:
                self.grad_scaler.scale(scaled_loss).backward()
            ppo_losses.append(float(loss.detach().item()))
            ratios.append(
                float(
                    torch.exp(
                        output["current_chain_score"].detach()
                        - output["old_chain_score"]
                    ).mean()
                )
            )

        if success_rank_count > 0 and success_obs is not None:
            assert success_action_model is not None
            success_size = success_action_model.shape[0]
            distributed_success_scale = (
                self._world_size / success_rank_count if local_has_success else 0.0
            )
            for start, stop in success_ranges:
                actions = torch.zeros(
                    stop - start,
                    int(self.ogpo_cfg.model_horizon),
                    int(self.ogpo_cfg.model_action_dim),
                    dtype=torch.float32,
                )
                actions[:, : int(self.ogpo_cfg.execution_horizon)] = (
                    success_action_model[start:stop]
                )
                bc_loss = self.model(
                    forward_type=ForwardType.OGPO_FLOW,
                    operation="bc_loss",
                    env_obs=_slice_env_obs(success_obs, start, stop),
                    actions=actions,
                )
                scaled_bc = (
                    float(self.ogpo_cfg.bc_coeff)
                    * distributed_success_scale
                    * bc_loss
                    * ((stop - start) / success_size)
                )
                backward_index += 1
                context = self.before_micro_batch(
                    self.model, is_last_micro_batch=backward_index == backward_count
                )
                with context:
                    self.grad_scaler.scale(scaled_bc).backward()
                if local_has_success:
                    bc_losses.append(
                        float(
                            bc_loss.detach().item() * distributed_success_scale
                        )
                    )

        grad_norm, learning_rates = self.optimizer_step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.model(
            forward_type=ForwardType.OGPO_FLOW,
            operation="ema_update",
            tau=float(self.ogpo_cfg.actor_tau),
        )
        self.actor_updates += 1
        self.policy_version += 1
        return {
            "ogpo/actor_loss": float(np.mean(ppo_losses)),
            "ogpo/bc_loss": float(np.mean(bc_losses)) if bc_losses else 0.0,
            "ogpo/ratio": float(np.mean(ratios)),
            "ogpo/actor_grad_norm": float(torch.as_tensor(grad_norm).item()),
            "ogpo/actor_lr": float(learning_rates[0]),
        }

    def _average_critic_gradients(self) -> None:
        assert self.critic is not None
        if not torch.distributed.is_initialized():
            return
        for parameter in self.critic.parameters():
            if parameter.grad is None:
                continue
            torch.distributed.all_reduce(
                parameter.grad, op=torch.distributed.ReduceOp.SUM
            )
            parameter.grad.div_(self._world_size)

    @torch.no_grad()
    def _update_target_critic(self) -> None:
        assert self.critic is not None and self.target_critic is not None
        tau = float(self.ogpo_cfg.critic_tau)
        for target, online in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target.mul_(1.0 - tau).add_(online, alpha=tau)

    def _critic_update(
        self,
        sequence: OGPOSequenceBatch,
        target_next: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        assert self.critic is not None
        assert self.target_critic is not None
        assert self.critic_optimizer is not None
        env_obs = self._batch_to_env_obs(sequence.observation)
        with torch.no_grad():
            conditioning = self.model(
                forward_type=ForwardType.OGPO_FLOW,
                operation="conditioning",
                env_obs=env_obs,
            )
            next_feature = target_next["critic_feature"].to(
                self.device, torch.float32
            )
            next_proprio = target_next["state"][
                :, : int(self.ogpo_cfg.active_action_dim)
            ].to(self.device, torch.float32)
            next_action = target_next["canonical_action"][:, 0].to(
                self.device, torch.float32
            )
            next_q = self.target_critic(
                next_feature, next_proprio, next_action
            ).mean(dim=-1)
            target = h_step_td_target(
                sequence.rewards.to(self.device),
                sequence.bootstrap_mask.to(self.device),
                next_q,
                gamma=float(self.cfg.algorithm.gamma),
                valid_mask=sequence.valid.to(self.device),
            )

        feature = conditioning["critic_feature"].to(self.device, torch.float32)
        proprio = conditioning["state"][
            :, : int(self.ogpo_cfg.active_action_dim)
        ].to(self.device, torch.float32)
        predicted = self.critic(
            feature,
            proprio,
            sequence.action.to(self.device),
            sequence.h.to(self.device),
        )
        loss = (predicted - target[:, None]).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._average_critic_gradients()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), float(self.ogpo_cfg.critic_grad_clip)
        )
        self.critic_optimizer.step()
        self._update_target_critic()
        self.critic_updates += 1
        return {
            "ogpo/critic_loss": float(loss.detach().item()),
            "ogpo/critic_grad_norm": float(torch.as_tensor(grad_norm).item()),
            "ogpo/q_mean": float(predicted.detach().mean().item()),
            "ogpo/td_target_mean": float(target.mean().item()),
        }

    def _local_batch_size(self) -> int:
        global_batch = int(self.ogpo_cfg.state_batch_size)
        if global_batch % self._world_size:
            raise ValueError("OGPO state_batch_size must divide actor world size")
        return global_batch // self._world_size

    def _updates_to_run(self) -> int:
        local_batch = self._local_batch_size()
        credit_updates = min(
            int(self.pending_actor_updates), int(self.pending_critic_updates)
        )
        if not torch.distributed.is_initialized():
            return credit_updates if len(self.replay) >= local_batch else 0

        credit_bounds = torch.tensor(
            [credit_updates, credit_updates],
            device=self.device,
            dtype=torch.long,
        )
        torch.distributed.all_reduce(
            credit_bounds[0], op=torch.distributed.ReduceOp.MIN
        )
        torch.distributed.all_reduce(
            credit_bounds[1], op=torch.distributed.ReduceOp.MAX
        )
        if credit_bounds[0].item() != credit_bounds[1].item():
            raise RuntimeError("OGPO update credits diverged across actor ranks")

        all_ready = torch.tensor(
            int(len(self.replay) >= local_batch),
            device=self.device,
            dtype=torch.long,
        )
        torch.distributed.all_reduce(all_ready, op=torch.distributed.ReduceOp.MIN)
        return credit_updates if bool(all_ready.item()) else 0

    @Worker.timer("run_training")
    def run_training(self) -> dict[str, float]:
        if self.enable_offload:
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)
        self.model.train()
        updates = self._updates_to_run()
        metrics: dict[str, list[float]] = {}
        local_batch = self._local_batch_size()
        for _ in range(updates):
            sequence = self.replay.sample_sequences(local_batch)
            success = self.replay.sample_success_sequences(local_batch)
            # This target action is sampled before actor EMA changes, matching
            # the update-start target snapshot used by the released JAX code.
            target_next = self._target_action(
                self._batch_to_env_obs(sequence.next_observation)
            )
            update_metrics = self._actor_update(sequence, success)
            update_metrics.update(self._critic_update(sequence, target_next))
            for key, value in update_metrics.items():
                metrics.setdefault(key, []).append(float(value))
        self.pending_actor_updates -= updates
        self.pending_critic_updates -= updates

        result = {
            key: float(np.mean(values)) for key, values in metrics.items()
        }
        result.update(self._last_ingest_metrics)
        result.update(
            {
                "ogpo/updates_run": float(updates),
                "ogpo/actor_updates": float(self.actor_updates),
                "ogpo/critic_updates": float(self.critic_updates),
                "ogpo/policy_version": float(self.policy_version),
                "ogpo/pending_actor_updates": float(self.pending_actor_updates),
                "ogpo/pending_critic_updates": float(self.pending_critic_updates),
            }
        )
        if torch.distributed.is_initialized():
            result = all_reduce_dict(
                result, op=torch.distributed.ReduceOp.AVG
            )
            torch.distributed.barrier()
        clear_memory()
        return result

    def _sidecar_path(self, base_path: str) -> Path:
        return Path(base_path) / "ogpo_components" / f"rank_{self._rank}.pt"

    def _checkpoint_contract(self) -> dict[str, Any]:
        return {
            "version": self._SIDECAR_VERSION,
            "world_size": int(self._world_size),
            "source_fingerprint": str(self.ogpo_cfg.source_fingerprint),
            "norm_fingerprint": str(self.ogpo_cfg.norm_fingerprint),
            "replay_schema_version": int(self.replay.schema_version),
            "model_horizon": int(self.ogpo_cfg.model_horizon),
            "execution_horizon": int(self.ogpo_cfg.execution_horizon),
            "model_action_dim": int(self.ogpo_cfg.model_action_dim),
            "active_action_dim": int(self.ogpo_cfg.active_action_dim),
            "flow_steps": int(self.ogpo_cfg.flow_steps),
            "num_q_heads": int(self.ogpo_cfg.num_q_heads),
            "critic_hidden_dims": [
                int(value) for value in self.ogpo_cfg.critic_hidden_dims
            ],
            "replay_capacity": int(self.ogpo_cfg.replay_capacity),
            "replay_capacity_per_rank": int(self.replay.capacity),
            "actor_seed": int(self.cfg.actor.seed),
            "state_batch_size": int(self.ogpo_cfg.state_batch_size),
            "candidate_group_size": int(self.ogpo_cfg.candidate_group_size),
            "start_training_rows": int(self.ogpo_cfg.start_training_rows),
            "total_online_rows": int(self.ogpo_cfg.total_online_rows),
            "utd_q": float(self.ogpo_cfg.utd_q),
            "utd_pi": float(self.ogpo_cfg.utd_pi),
            "gamma": float(self.cfg.algorithm.gamma),
            "sigma_init": float(self.ogpo_cfg.sigma_init),
            "gaussian_clip": float(self.ogpo_cfg.gaussian_clip),
            "normalize_denoising_horizon": bool(
                self.ogpo_cfg.normalize_denoising_horizon
            ),
            "normalize_act_space_dimension": bool(
                self.ogpo_cfg.normalize_act_space_dimension
            ),
            "clip_epsilon": float(self.ogpo_cfg.clip_epsilon),
            "bc_coeff": float(self.ogpo_cfg.bc_coeff),
            "actor_tau": float(self.ogpo_cfg.actor_tau),
            "critic_tau": float(self.ogpo_cfg.critic_tau),
            "critic_lr": float(self.ogpo_cfg.critic_lr),
            "critic_adam_beta1": float(self.ogpo_cfg.critic_adam_beta1),
            "critic_adam_beta2": float(self.ogpo_cfg.critic_adam_beta2),
            "critic_adam_eps": float(self.ogpo_cfg.critic_adam_eps),
            "critic_weight_decay": float(self.ogpo_cfg.critic_weight_decay),
            "critic_grad_clip": float(self.ogpo_cfg.critic_grad_clip),
        }

    def _raise_distributed_checkpoint_error(
        self, stage: str, local_error: str | None
    ) -> None:
        if torch.distributed.is_initialized():
            errors: list[str | None] = [None] * self._world_size
            torch.distributed.all_gather_object(errors, local_error)
        else:
            errors = [local_error]
        failures = [
            f"rank {rank}: {error}"
            for rank, error in enumerate(errors)
            if error is not None
        ]
        if failures:
            raise RuntimeError(
                f"OGPO checkpoint {stage} failed; " + "; ".join(failures)
            )

    def _shared_checkpoint_signature(self, state: dict[str, Any]) -> dict[str, Any]:
        def tensor_schema(value: Any) -> tuple[tuple[str, tuple[int, ...], str], ...]:
            if not isinstance(value, dict):
                raise ValueError("checkpoint tensor state must be a dictionary")
            schema = []
            for name, tensor in value.items():
                if not torch.is_tensor(tensor):
                    raise ValueError(f"checkpoint state entry is not a tensor: {name}")
                schema.append((str(name), tuple(tensor.shape), str(tensor.dtype)))
            return tuple(sorted(schema))

        feature_dim = state["critic_feature_dim"]
        critic_state = state["critic"]
        target_critic_state = state["target_critic"]
        critic_optimizer = state["critic_optimizer"]
        if feature_dim is None:
            if any(
                value is not None
                for value in (critic_state, target_critic_state, critic_optimizer)
            ):
                raise ValueError("uninitialized critic checkpoint contains critic state")
            critic_schema = None
        else:
            if critic_optimizer is None:
                raise ValueError("initialized critic checkpoint omitted optimizer state")
            critic_schema = tensor_schema(critic_state)
            if tensor_schema(target_critic_state) != critic_schema:
                raise ValueError("online and target critic schemas differ")

        shadow = state["actor_ema_shadow"]
        if not isinstance(shadow, dict):
            raise ValueError("actor EMA shadow must be a dictionary")
        for name, value in shadow.items():
            if not torch.is_tensor(value) or value.dtype != torch.float32:
                raise ValueError(f"invalid FP32 actor EMA shadow: {name}")

        return {
            "snapshot_id": str(state["snapshot_id"]),
            "step": int(state["step"]),
            "global_online_rows": int(state["global_online_rows"]),
            "pending_actor_updates": float(state["pending_actor_updates"]),
            "pending_critic_updates": float(state["pending_critic_updates"]),
            "actor_updates": int(state["actor_updates"]),
            "critic_updates": int(state["critic_updates"]),
            "policy_version": int(state["policy_version"]),
            "critic_feature_dim": None if feature_dim is None else int(feature_dim),
            "critic_schema": critic_schema,
            "actor_ema_shadow_keys": tuple(sorted(str(name) for name in shadow)),
        }

    def _validate_loaded_sidecar(
        self, state: dict[str, Any], manifest: dict[str, Any]
    ) -> dict[str, Any]:
        required = {
            "contract",
            "snapshot_id",
            "rank",
            "step",
            "critic_feature_dim",
            "critic",
            "target_critic",
            "critic_optimizer",
            "actor_ema_shadow",
            "replay",
            "global_online_rows",
            "pending_actor_updates",
            "pending_critic_updates",
            "actor_updates",
            "critic_updates",
            "policy_version",
            "episode_counter",
        }
        missing = sorted(required - state.keys())
        if missing:
            raise ValueError(f"sidecar omitted fields: {missing}")
        if state["contract"] != self._checkpoint_contract():
            raise ValueError("sidecar contract differs from current run")
        if int(state["rank"]) != self._rank:
            raise ValueError("sidecar rank differs from worker rank")
        for field in ("snapshot_id", "step", "global_online_rows", "policy_version"):
            if state[field] != manifest[field]:
                raise ValueError(f"sidecar {field} differs from completion manifest")

        replay = state["replay"]
        if not isinstance(replay, dict):
            raise ValueError("replay sidecar state must be a dictionary")
        expected_replay_metadata = {
            "capacity": self.replay.capacity,
            "max_sequence_length": self.replay.max_sequence_length,
            "action_dim": self.replay.action_dim,
            "model_action_dim": self.replay.model_action_dim,
            "seed": self.replay.seed,
        }
        if replay.get("schema_version") != self.replay.schema_version:
            raise ValueError("replay schema differs from current worker")
        if replay.get("metadata") != expected_replay_metadata:
            raise ValueError("replay metadata differs from current worker")
        slots = replay.get("slots")
        if not isinstance(slots, list) or len(slots) != self.replay.capacity:
            raise ValueError("replay slot count differs from configured capacity")
        replay_size = int(replay["size"])
        if not 0 <= replay_size <= self.replay.capacity:
            raise ValueError("replay size is out of range")
        if sum(slot is not None for slot in slots) != replay_size:
            raise ValueError("replay size differs from live slot count")
        return self._shared_checkpoint_signature(state)

    def _check_shared_checkpoint_signature(
        self, local_signature: dict[str, Any]
    ) -> None:
        if torch.distributed.is_initialized():
            signatures: list[dict[str, Any] | None] = [None] * self._world_size
            torch.distributed.all_gather_object(signatures, local_signature)
        else:
            signatures = [local_signature]
        reference = signatures[0]
        if any(signature != reference for signature in signatures[1:]):
            raise RuntimeError("OGPO checkpoint shared rank signatures differ")

    @staticmethod
    def _load_sidecar_file(sidecar: Path) -> dict[str, Any]:
        try:
            return torch.load(sidecar, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(sidecar, map_location="cpu")

    def save_checkpoint(self, save_base_path: str, step: int) -> None:
        sidecar = self._sidecar_path(save_base_path)
        manifest = sidecar.parent / "complete.json"
        invalidate_error = None
        if self._rank == 0:
            try:
                manifest.unlink(missing_ok=True)
            except Exception as error:
                invalidate_error = f"{type(error).__name__}: {error}"
        self._raise_distributed_checkpoint_error(
            "manifest invalidation", invalidate_error
        )

        snapshot_holder = [uuid.uuid4().hex if self._rank == 0 else None]
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(snapshot_holder, src=0)
        snapshot_id = str(snapshot_holder[0])
        super().save_checkpoint(save_base_path, step)
        temporary_sidecar = sidecar.with_name(
            f".{sidecar.name}.tmp-{os.getpid()}"
        )
        sidecar_error = None
        try:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            shadow_state = self.model(
                forward_type=ForwardType.OGPO_FLOW,
                operation="ema_shadow_state",
            )
            sidecar_state = {
                "contract": self._checkpoint_contract(),
                "snapshot_id": snapshot_id,
                "rank": int(self._rank),
                "step": int(step),
                "critic_feature_dim": self.critic_feature_dim,
                "critic": None if self.critic is None else self.critic.state_dict(),
                "target_critic": (
                    None
                    if self.target_critic is None
                    else self.target_critic.state_dict()
                ),
                "critic_optimizer": (
                    None
                    if self.critic_optimizer is None
                    else self.critic_optimizer.state_dict()
                ),
                "actor_ema_shadow": shadow_state,
                "replay": self.replay.state_dict(),
                "global_online_rows": self.global_online_rows,
                "pending_actor_updates": self.pending_actor_updates,
                "pending_critic_updates": self.pending_critic_updates,
                "actor_updates": self.actor_updates,
                "critic_updates": self.critic_updates,
                "policy_version": self.policy_version,
                "episode_counter": self._episode_counter,
            }
            torch.save(sidecar_state, temporary_sidecar)
            os.replace(temporary_sidecar, sidecar)
        except Exception as error:
            sidecar_error = f"{type(error).__name__}: {error}"
            try:
                temporary_sidecar.unlink(missing_ok=True)
            except OSError:
                pass
        self._raise_distributed_checkpoint_error("sidecar write", sidecar_error)

        manifest_error = None
        if self._rank == 0:
            manifest_payload = {
                "complete": True,
                "contract": self._checkpoint_contract(),
                "snapshot_id": snapshot_id,
                "step": int(step),
                "global_online_rows": int(self.global_online_rows),
                "policy_version": int(self.policy_version),
                "sidecars": [
                    f"rank_{rank}.pt" for rank in range(self._world_size)
                ],
            }
            temporary_manifest = manifest.with_name(
                f".{manifest.name}.tmp-{os.getpid()}"
            )
            try:
                temporary_manifest.write_text(
                    json.dumps(
                        manifest_payload,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary_manifest, manifest)
            except Exception as error:
                manifest_error = f"{type(error).__name__}: {error}"
                try:
                    temporary_manifest.unlink(missing_ok=True)
                except OSError:
                    pass
        self._raise_distributed_checkpoint_error("manifest publish", manifest_error)

    def load_checkpoint(self, load_base_path: str) -> None:
        sidecar = self._sidecar_path(load_base_path)
        manifest_path = sidecar.parent / "complete.json"
        manifest = None
        state = None
        signature = None
        preflight_error = None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not manifest.get("complete", False):
                raise ValueError("completion marker is false")
            if manifest.get("contract") != self._checkpoint_contract():
                raise ValueError("checkpoint contract differs from current run")
            expected_sidecars = [
                f"rank_{rank}.pt" for rank in range(self._world_size)
            ]
            if manifest.get("sidecars") != expected_sidecars:
                raise ValueError("sidecar rank list is incomplete")
            if not sidecar.is_file() or sidecar.stat().st_size <= 0:
                raise ValueError(f"missing or empty sidecar {sidecar.name}")
            state = self._load_sidecar_file(sidecar)
            signature = self._validate_loaded_sidecar(state, manifest)
        except Exception as error:  # synchronized below before DCP collectives
            preflight_error = f"{type(error).__name__}: {error}"
        self._raise_distributed_checkpoint_error("preflight", preflight_error)
        assert manifest is not None and state is not None and signature is not None
        self._check_shared_checkpoint_signature(signature)

        super().load_checkpoint(load_base_path)

        shadow_error = None
        try:
            self.model(
                forward_type=ForwardType.OGPO_FLOW,
                operation="load_ema_shadow_state",
                state=state["actor_ema_shadow"],
            )
        except Exception as error:
            shadow_error = f"{type(error).__name__}: {error}"
        self._raise_distributed_checkpoint_error("EMA shadow restore", shadow_error)

        feature_dim = state["critic_feature_dim"]
        critic_error = None
        try:
            if feature_dim is not None:
                self._init_critic(int(feature_dim))
                assert self.critic is not None
                assert self.target_critic is not None
                assert self.critic_optimizer is not None
                self.critic.load_state_dict(state["critic"])
                self.target_critic.load_state_dict(state["target_critic"])
                self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        except Exception as error:
            critic_error = f"{type(error).__name__}: {error}"
        self._raise_distributed_checkpoint_error("critic restore", critic_error)

        replay_error = None
        try:
            self.replay.load_state_dict(state["replay"])
            self.global_online_rows = int(state["global_online_rows"])
            self.pending_actor_updates = float(state["pending_actor_updates"])
            self.pending_critic_updates = float(state["pending_critic_updates"])
            self.actor_updates = int(state["actor_updates"])
            self.critic_updates = int(state["critic_updates"])
            self.policy_version = int(state["policy_version"])
            self._episode_counter = int(state["episode_counter"])
        except Exception as error:
            replay_error = f"{type(error).__name__}: {error}"
        self._raise_distributed_checkpoint_error("replay restore", replay_error)
