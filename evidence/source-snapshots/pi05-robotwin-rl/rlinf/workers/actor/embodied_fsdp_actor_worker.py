# Copyright 2025 The RLinf Authors.
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

import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.dvac_train_weighting import (
    DVACRecentStats,
    DVACStepStats,
    local_log_v_sufficient_statistics,
    straight_through_scale_logprobs,
)
from rlinf.algorithms.expert import build_expert_model_config
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.config import SupportedModel
from rlinf.data.schema.embodied_types import Trajectory, convert_trajectories_to_batch
from rlinf.data.storage.lerobot import resolve_lerobot_repo_id
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.distributed import (
    all_reduce_dict,
)
from rlinf.utils.metric_utils import (
    CRITIC_EXPLAINED_VARIANCE_KEY,
    append_to_dict,
    compute_critic_explained_variance_from_stats,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
    pop_critic_explained_variance_stats,
)
from rlinf.utils.nested_dict_process import (
    flatten_nested_tensor_time_batch,
    process_nested_dict_for_adv,
    process_nested_dict_for_train,
    put_tensor_device,
    split_dict_to_chunk,
    trim_nested_tensor_time_dim,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    masked_mean,
    reshape_entropy,
)


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self._opd_teacher_model = None
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")

        self.enable_sft_co_train = cfg.actor.get("enable_sft_co_train", False)
        self.version = 0

        dvac_cfg = OmegaConf.select(
            cfg, "algorithm.dvac_gradient_weighting", default=None
        )
        self.dvac_train_cfg = (
            {} if dvac_cfg is None else OmegaConf.to_container(dvac_cfg, resolve=True)
        )
        self.dvac_train_mode = str(self.dvac_train_cfg.get("mode", "off")).lower()
        if self.dvac_train_mode not in {"off", "apply"}:
            raise ValueError(
                "algorithm.dvac_gradient_weighting.mode must be 'off' or 'apply'"
            )
        self.dvac_train_enabled = self.dvac_train_mode == "apply"
        self.dvac_train_application = str(
            self.dvac_train_cfg.get("application", "logprob_st")
        ).lower()
        if self.dvac_train_application not in {
            "logprob_st",
            "action_advantage",
        }:
            raise ValueError(
                "algorithm.dvac_gradient_weighting.application must be "
                "'logprob_st' or 'action_advantage'"
            )
        self.dvac_selected_l = int(self.dvac_train_cfg.get("selected_l", 3))
        self.dvac_recent_stats = (
            self._new_dvac_recent_stats() if self.dvac_train_enabled else None
        )
        self.dvac_output_dir: Path | None = None
        self._dvac_pending_step: dict | None = None
        if self.enable_sft_co_train:
            self._build_sft_data_loader()

        # create weight syncer
        weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer")
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        self.update_epoch = self.cfg.algorithm.get("update_epoch", 1)

        self._sync_weight_comm_options = self.weight_syncer.comm_options

        self._is_weight_sender = self._rank == 0
        self._actor_world_size = self._world_size
        self._rollout_all_ranks = list(
            range(self._component_placement.get_world_size("rollout"))
        )

    def _new_dvac_recent_stats(self) -> DVACRecentStats:
        weight_min = self.dvac_train_cfg.get("weight_min", None)
        weight_max = self.dvac_train_cfg.get("weight_max", None)
        return DVACRecentStats(
            window_steps=int(self.dvac_train_cfg.get("window_steps", 5)),
            warmup_steps=int(self.dvac_train_cfg.get("warmup_steps", 1)),
            log_eps=float(self.dvac_train_cfg.get("log_eps", 1e-12)),
            std_floor=float(self.dvac_train_cfg.get("std_floor", 1e-6)),
            z_clip=float(self.dvac_train_cfg.get("z_clip", 2.0)),
            strength=float(self.dvac_train_cfg.get("strength", 0.5)),
            weight_min=None if weight_min is None else float(weight_min),
            weight_max=None if weight_max is None else float(weight_max),
        )

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        if self.dvac_train_enabled:
            if SupportedModel(self.cfg.actor.model.model_type) != SupportedModel.OPENPI:
                raise ValueError("DVAC train weighting requires OpenPI.")
            expected_logprob_type = (
                "chunk_level"
                if self.dvac_train_application == "logprob_st"
                else "action_level"
            )
            if self.cfg.algorithm.logprob_type != expected_logprob_type:
                raise ValueError(
                    f"DVAC {self.dvac_train_application} requires "
                    f"algorithm.logprob_type={expected_logprob_type}."
                )
            if (
                self.dvac_train_application == "action_advantage"
                and self.cfg.algorithm.reward_type != "chunk_level"
            ):
                raise ValueError(
                    "DVAC action_advantage requires trajectory-level chunk rewards."
                )
            output_dir = self.dvac_train_cfg.get("output_dir")
            if not output_dir:
                raise ValueError(
                    "algorithm.dvac_gradient_weighting.output_dir is required."
                )
            self.dvac_output_dir = Path(str(output_dir)) / (
                f"actor_rank{self._rank:02d}"
            )
            self.dvac_output_dir.mkdir(parents=True, exist_ok=True)

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def get_rollout_state_dict(self) -> dict:
        return self.get_model_state_dict(cpu_offload=False, full_state_dict=False)

    @Worker.timer("actor/sync_model_to_rollout")
    async def sync_model_to_rollout(self) -> None:
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        state_dict = self.get_rollout_state_dict()

        async def send_func(data):
            if not self._is_weight_sender:
                return
            await self.broadcast(
                data,
                groups=[
                    (self._group_name, 0),
                    (self._rollout_group_name, self._rollout_all_ranks),
                ],
                src=(self._group_name, 0),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def recv_func():
            return await self.recv(
                src_group_name=self._rollout_group_name,
                src_rank=0,
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        if not self.weight_syncer.sender_initialized():
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
                param_names_need_sync=self.param_names_need_sync,
                is_sender=self._is_weight_sender,
            )

        version = (
            self.get_rollout_sync_version()
            if hasattr(self, "get_rollout_sync_version")
            else self.version
        )
        await self.weight_syncer.sync(state_dict, send_func, version=version)

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad(True)

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.rollout_batch = convert_trajectories_to_batch(recv_list)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.env.train.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        if self.cfg.algorithm.adv_type == "opd":
            self.compute_opd_teacher_logprobs()

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "prev_logprobs": self.rollout_batch.get("prev_logprobs", None),
            "teacher_logprobs": self.rollout_batch.get("teacher_logprobs", None),
            "num_action_chunks": self.cfg.actor.model.num_action_chunks,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
            "advantage_mode": self.cfg.algorithm.get("advantage_mode", None),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    @Worker.timer("actor/compute_opd_teacher_logprobs")
    def compute_opd_teacher_logprobs(self) -> None:
        assert self.rollout_batch.get("teacher_logprobs", None) is None, (
            "OPD teacher_logprobs must be computed after rollout on actor workers."
        )
        assert self.cfg.rollout.get("expert_model", None) is not None, (
            "OPD requires rollout.expert_model as teacher model config."
        )
        assert "forward_inputs" in self.rollout_batch, (
            "OPD teacher logprob computation requires rollout forward_inputs."
        )
        assert "prev_logprobs" in self.rollout_batch, (
            "OPD teacher logprob computation requires student prev_logprobs."
        )
        assert SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ], "OPD teacher logprob computation currently supports OpenVLA models."

        prev_logprobs = self.rollout_batch["prev_logprobs"]
        time_dim, batch_dim = prev_logprobs.shape[:2]
        flat_batch_size = time_dim * batch_dim

        assert self.enable_offload and self.is_weight_offloaded, (
            "OPD teacher logprob computation expects actor weights to be "
            "offloaded before moving the teacher model to GPU."
        )
        teacher_model = self._get_opd_teacher_model()
        teacher_model.to(self.device)

        flat_forward_inputs = flatten_nested_tensor_time_batch(
            self.rollout_batch["forward_inputs"], ("forward_inputs",)
        )
        num_chunks = (
            flat_batch_size + self.cfg.actor.micro_batch_size - 1
        ) // self.cfg.actor.micro_batch_size
        teacher_logprobs = []
        kwargs = {
            "temperature": self.cfg.rollout.sampling_params.temperature_train,
            "top_k": self.cfg.rollout.sampling_params.top_k,
        }
        with torch.no_grad():
            for micro_batch in split_dict_to_chunk(flat_forward_inputs, num_chunks):
                micro_batch = put_tensor_device(micro_batch, self.device)
                with self.amp_context:
                    teacher_output = teacher_model(
                        forward_inputs=micro_batch,
                        compute_logprobs=True,
                        compute_entropy=False,
                        compute_values=False,
                        use_cache=False,
                        **kwargs,
                    )
                teacher_logprobs.append(teacher_output["logprobs"].detach().cpu())

        teacher_logprobs = torch.cat(teacher_logprobs, dim=0)
        expected_shape = (flat_batch_size, *prev_logprobs.shape[2:])
        assert teacher_logprobs.shape == expected_shape, (
            f"teacher_logprobs shape {teacher_logprobs.shape} must match "
            f"flattened student logprobs shape {expected_shape}."
        )
        self.rollout_batch["teacher_logprobs"] = teacher_logprobs.reshape(
            time_dim, batch_dim, *teacher_logprobs.shape[1:]
        )

        teacher_model.to("cpu")
        clear_memory()

    def _get_opd_teacher_model(self):
        if self._opd_teacher_model is None:
            teacher_model_config = build_expert_model_config(
                self.cfg, self.cfg.actor.model
            )
            teacher_model = get_model(teacher_model_config)
            if self.cfg.runner.get("expert_ckpt_path", None):
                teacher_model_dict = torch.load(
                    self.cfg.runner.expert_ckpt_path, map_location="cpu"
                )
                teacher_model.load_state_dict(teacher_model_dict)
            teacher_model.eval()
            teacher_model.requires_grad_(False)
            teacher_model.to("cpu")
            self._opd_teacher_model = teacher_model
        return self._opd_teacher_model

    def _build_sft_data_loader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            repo_id = resolve_lerobot_repo_id(self.cfg.actor.get("sft_data_path"))
            if repo_id is None:
                raise ValueError(
                    "actor.sft_data_path must be set to a local dataset path or "
                    "LeRobot repo id when enable_sft_co_train=True."
                )

            import openpi.training.data_loader as _data

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            if "config_name" not in self.cfg.actor:
                raise ValueError(
                    "config_name is required when enable_sft_co_train=True"
                )
            training_config_name = self.cfg.actor.config_name
            data_loader_config = get_openpi_config(
                training_config_name,
                model_path=self.cfg.actor.model.model_path,
                repo_id=repo_id,
                data_kwargs=getattr(self.cfg.actor.model, "openpi_data", None),
            )
            self.data_loader = _data.create_data_loader(
                data_loader_config, framework="pytorch", shuffle=True
            )
            self.sft_iterator = iter(self.data_loader)
            self.train_epoch = 0
            self.sft_loss_weight = self.cfg.actor.get("sft_loss_weight", 0.1)
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _train_sft_epoch(
        self, metrics_data: dict[str, torch.Tensor], loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Train one epoch of SFT.
        """
        metrics_data["ppo_loss"] = loss.clone().detach().item()

        # Get next data batch
        try:
            observation, actions = next(self.sft_iterator)
        except StopIteration:
            self.train_epoch += 1
            self.data_loader.set_epoch(self.train_epoch)
            self.sft_iterator = iter(self.data_loader)
            observation, actions = next(self.sft_iterator)

        sft_loss = self.model(
            data=(observation, actions),
            forward_type=ForwardType.SFT,
        )
        metrics_data["sft_loss"] = sft_loss.detach().item()
        total_loss = loss + self.sft_loss_weight * sft_loss
        loss = total_loss

        metrics_data["loss_ratio"] = (
            np.abs(metrics_data["sft_loss"]) / np.abs(metrics_data["ppo_loss"])
            if np.abs(metrics_data["ppo_loss"]) > 0
            else float("inf")
        )
        if metrics_data["loss_ratio"] > 1e5:
            self.logger.warning(
                "SFT/PPO loss imbalance detected: "
                f"ratio={metrics_data['loss_ratio']:.3e}, "
                f"sft_loss={metrics_data['sft_loss']:.6f}, "
                f"ppo_loss={metrics_data['ppo_loss']:.6f}, "
                f"sft_loss_weight={self.sft_loss_weight:.6f}"
            )
        return loss

    def _prepare_dvac_train_step(self) -> None:
        forward_inputs = self.rollout_batch.get("forward_inputs")
        if not isinstance(forward_inputs, dict):
            raise ValueError("DVAC gradient weighting requires forward_inputs.")
        variance_key = f"dvac_v_l{self.dvac_selected_l}"
        if variance_key not in forward_inputs:
            raise ValueError(f"Missing train-time DVAC tensor: {variance_key}")
        variance = forward_inputs.pop(variance_key)

        local_stats = local_log_v_sufficient_statistics(
            variance, log_eps=self.dvac_recent_stats.log_eps
        ).to(self.device)
        torch.distributed.all_reduce(local_stats, op=torch.distributed.ReduceOp.SUM)
        count, value_sum, value_sq_sum = local_stats.detach().cpu().tolist()
        current_stats = DVACStepStats(
            runner_step=int(self.version),
            count=int(round(count)),
            value_sum=float(value_sum),
            value_sq_sum=float(value_sq_sum),
        )

        weights, clipped_z, warmup, history = self.dvac_recent_stats.compute_weights(
            variance
        )
        forward_inputs["dvac_weights"] = weights
        self._dvac_pending_step = {
            "runner_step": int(self.version),
            "current_stats": current_stats,
            "history": history,
            "warmup": warmup,
            "variance": variance.detach().cpu().contiguous(),
            "weights": weights.detach().cpu().contiguous(),
            "clipped_z": clipped_z.detach().cpu().contiguous(),
            "advantages": self.rollout_batch["advantages"].detach().cpu().contiguous(),
            "rewards": self.rollout_batch["rewards"].detach().cpu().contiguous(),
            "loss_mask": (
                None
                if self.rollout_batch.get("loss_mask") is None
                else self.rollout_batch["loss_mask"].detach().cpu().contiguous()
            ),
            "metrics": {
                "actor/dvac_warmup": float(warmup),
                "actor/dvac_history_mean": float(history["history_mean"]),
                "actor/dvac_history_std": float(history["history_std"]),
                "actor/dvac_current_mean": current_stats.mean,
                "actor/dvac_current_std": current_stats.std,
                "actor/dvac_weight_mean": float(weights.float().mean().item()),
                "actor/dvac_weight_sq_mean": float(
                    weights.float().square().mean().item()
                ),
                "actor/dvac_z_low_clip_fraction": float(
                    (clipped_z <= -self.dvac_recent_stats.z_clip).float().mean().item()
                ),
                "actor/dvac_z_high_clip_fraction": float(
                    (clipped_z >= self.dvac_recent_stats.z_clip).float().mean().item()
                ),
            },
        }

    def _write_dvac_step_artifact(self, pending: dict) -> None:
        if not bool(self.dvac_train_cfg.get("save_step_tensors", True)):
            return
        path = self.dvac_output_dir / (
            f"runner_step_{int(pending['runner_step']):04d}.pt"
        )
        if path.exists():
            raise FileExistsError(f"DVAC step artifact already exists: {path}")
        temporary = path.with_suffix(".partial.pt")
        torch.save(
            {
                "schema_version": 1,
                "runner_step": int(pending["runner_step"]),
                "actor_rank": int(self._rank),
                "application": self.dvac_train_application,
                "advantage_reduction": (
                    "sum_valid_actions_then_mean_queries"
                    if self.dvac_train_application == "action_advantage"
                    else None
                ),
                "selected_l": int(self.dvac_selected_l),
                "warmup": bool(pending["warmup"]),
                "history": dict(pending["history"]),
                "current_stats": {
                    "count": pending["current_stats"].count,
                    "mean": pending["current_stats"].mean,
                    "std": pending["current_stats"].std,
                },
                "variance": pending["variance"],
                "weights": pending["weights"],
                "clipped_z": pending["clipped_z"],
                "advantages": pending["advantages"],
                "rewards": pending["rewards"],
                "loss_mask": pending["loss_mask"],
            },
            temporary,
        )
        os.replace(temporary, path)

    @Worker.timer("run_training")
    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        if self.cfg.algorithm.loss_type == "opd":
            target_steps = int(self.rollout_batch["advantages"].shape[0])
            for key in [
                "prev_logprobs",
                "forward_inputs",
                "loss_mask",
                "loss_mask_sum",
            ]:
                assert key in self.rollout_batch, f"OPD training requires {key}."
                self.rollout_batch[key] = trim_nested_tensor_time_dim(
                    self.rollout_batch[key], target_steps, (key,)
                )

        self.model.train()
        if self.dvac_train_enabled:
            self._prepare_dvac_train_step()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    self.train_micro_batch(
                        micro_batch=batch,
                        metrics=metrics,
                        is_last=(idx + 1) == self.gradient_accumulation,
                    )
                    # avoid gpu memory leak
                    train_micro_batch[idx] = None
                    del batch

                self.torch_platform.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        if self.dvac_train_enabled:
            append_to_dict(metrics, self._dvac_pending_step["metrics"])

        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        explained_variance_stats = pop_critic_explained_variance_stats(metrics)
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        if self.dvac_train_enabled:
            weight_mean = float(mean_metric_dict["actor/dvac_weight_mean"])
            weight_sq_mean = float(mean_metric_dict["actor/dvac_weight_sq_mean"])
            mean_metric_dict["actor/dvac_weight_ess_fraction"] = (
                weight_mean**2 / weight_sq_mean if weight_sq_mean > 0 else 0.0
            )
            pending = self._dvac_pending_step
            self._write_dvac_step_artifact(pending)
            self.dvac_recent_stats.push(pending["current_stats"])
            self._dvac_pending_step = None
        if explained_variance_stats:
            reduced_stats = all_reduce_dict(
                explained_variance_stats, op=torch.distributed.ReduceOp.SUM
            )
            mean_metric_dict[CRITIC_EXPLAINED_VARIANCE_KEY] = (
                compute_critic_explained_variance_from_stats(reduced_stats).item()
            )

        return mean_metric_dict

    def train_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        metrics: dict[str, list[float]],
        *,
        is_last: bool,
    ) -> None:
        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)
        advantages = micro_batch["advantages"]
        prev_logprobs = micro_batch["prev_logprobs"]
        returns = micro_batch.get("returns", None)
        prev_values = micro_batch.get("prev_values", None)
        loss_mask = micro_batch.get("loss_mask", None)
        loss_mask_sum = micro_batch.get("loss_mask_sum", None)
        forward_inputs = micro_batch.get("forward_inputs", None)

        dvac_weights = None
        model_forward_inputs = forward_inputs
        if self.dvac_train_enabled:
            if forward_inputs is None or "dvac_weights" not in forward_inputs:
                raise ValueError("Missing frozen per-h DVAC weights during replay.")
            dvac_weights = forward_inputs["dvac_weights"]
            model_forward_inputs = {
                key: value
                for key, value in forward_inputs.items()
                if key != "dvac_weights"
            }

        kwargs = {}
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            kwargs["temperature"] = self.cfg.rollout.sampling_params.temperature_train
            kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            kwargs["prev_logprobs"] = prev_logprobs

        compute_values = self.cfg.algorithm.adv_type == "gae"
        with self.amp_context:
            output_dict = self.model(
                forward_inputs=model_forward_inputs,
                compute_logprobs=True,
                compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                compute_values=compute_values,
                use_cache=False,
                **kwargs,
            )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            prev_logprobs = output_dict["prev_logprobs"]

        logprobs_for_loss = output_dict["logprobs"]
        if self.dvac_train_enabled and self.dvac_train_application == "logprob_st":
            logprobs_for_loss = straight_through_scale_logprobs(
                logprobs_for_loss, dvac_weights
            )

        loss_kwargs = {
            "loss_type": self.cfg.algorithm.loss_type,
            "logprob_type": self.cfg.algorithm.logprob_type,
            "reward_type": self.cfg.algorithm.reward_type,
            "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
            "logprobs": logprobs_for_loss,
            "values": output_dict.get("values", None),
            "old_logprobs": prev_logprobs,
            "advantages": advantages,
            "returns": returns,
            "prev_values": prev_values,
            "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
            "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
            "value_clip": self.cfg.algorithm.get("value_clip", None),
            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "max_episode_steps": self.cfg.env.train.max_episode_steps,
            "task_type": self.cfg.runner.task_type,
            "critic_warmup": self.optimizer_steps < self.critic_warmup_steps,
        }
        if (
            self.dvac_train_enabled
            and self.dvac_train_application == "action_advantage"
        ):
            loss_kwargs["dvac_advantage_weights"] = dvac_weights
            loss_kwargs["action_level_sum"] = True

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
        ]:
            loss_kwargs["clip_ratio_c"] = self.cfg.algorithm.get("clip_ratio_c", 3.0)
            if self.cfg.algorithm.get("clip_log_ratio_min") is not None:
                loss_kwargs["clip_log_ratio_min"] = (
                    self.cfg.algorithm.clip_log_ratio_min
                )
            if self.cfg.algorithm.get("clip_log_ratio_max") is not None:
                loss_kwargs["clip_log_ratio_max"] = (
                    self.cfg.algorithm.clip_log_ratio_max
                )

        loss, metrics_data = policy_loss(**loss_kwargs)
        entropy_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
        if self.cfg.algorithm.entropy_bonus > 0 and not loss_kwargs["critic_warmup"]:
            entropy = output_dict["entropy"]
            entropy = reshape_entropy(
                entropy,
                entropy_type=self.cfg.algorithm.entropy_type,
                action_dim=self.cfg.actor.model.get("action_dim", 7),
                batch_size=output_dict["logprobs"].shape[0],
            )
            entropy_loss = masked_mean(entropy, mask=loss_mask)
            loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
        metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()

        if self.enable_sft_co_train:
            loss = self._train_sft_epoch(metrics_data, loss)

        loss /= self.gradient_accumulation
        with backward_ctx:
            self.grad_scaler.scale(loss).backward()

        metrics_data["actor/total_loss"] = loss.detach().item()
        append_to_dict(metrics, metrics_data)

    def _dvac_sidecar_path(self, checkpoint_path: str) -> Path:
        return Path(checkpoint_path) / f"dvac_state_rank{self._rank:04d}.json"

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)
        if not self.dvac_train_enabled:
            return
        if self._dvac_pending_step is not None:
            raise RuntimeError(
                "Cannot checkpoint a partially applied DVAC runner step."
            )
        path = self._dvac_sidecar_path(save_path)
        temporary = path.with_suffix(".partial.json")
        payload = {
            "schema_version": 1,
            "runner_step": int(step),
            "actor_rank": int(self._rank),
            "actor_world_size": int(self._world_size),
            "mode": self.dvac_train_mode,
            "application": self.dvac_train_application,
            "selected_l": int(self.dvac_selected_l),
            "recent_stats": self.dvac_recent_stats.state_dict(),
        }
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def load_checkpoint(self, load_path: str) -> None:
        restored_stats = None
        if self.dvac_train_enabled:
            path = self._dvac_sidecar_path(load_path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Exact DVAC resume requires actor sidecar: {path}"
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            expected = {
                "schema_version": 1,
                "actor_rank": int(self._rank),
                "actor_world_size": int(self._world_size),
                "mode": self.dvac_train_mode,
                "application": self.dvac_train_application,
                "selected_l": int(self.dvac_selected_l),
            }
            for key, value in expected.items():
                checkpoint_value = payload.get(key)
                if key == "application" and checkpoint_value is None:
                    checkpoint_value = "logprob_st"
                if checkpoint_value != value:
                    raise ValueError(
                        f"DVAC resume mismatch for {key}: "
                        f"checkpoint={checkpoint_value!r}, current={value!r}"
                    )
            restored_stats = self._new_dvac_recent_stats()
            restored_stats.load_state_dict(payload["recent_stats"])

        super().load_checkpoint(load_path)
        if restored_stats is not None:
            self.dvac_recent_stats = restored_stats

    def set_global_step(self, global_step: int) -> None:
        """
        Set the global step for the model, if needed.
        """
        self.version = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def finish_global_batch(self, metrics: dict[str, list[float]]) -> None:
        self.torch_platform.empty_cache()
        grad_norm, lr_list = self.optimizer_step()
        self.optimizer.zero_grad()
        metric_data = {
            "actor/grad_norm": grad_norm,
            "actor/lr": lr_list[0],
        }
        if len(lr_list) > 1:
            metric_data["critic/lr"] = lr_list[1]
        append_to_dict(metrics, metric_data)
