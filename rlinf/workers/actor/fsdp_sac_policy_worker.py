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


import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from rlinf.config import SupportedModel
from rlinf.data.embodied_buffer_dataset import (
    PreloadReplayBufferDataset,
    ReplayBufferDataset,
    replay_buffer_collate_fn,
)
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import (
    DSRLTransitionReplayBuffer,
    TrajectoryReplayBuffer,
    project_dsrl_trajectory,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.scheduler import Channel, Worker
from rlinf.utils import drq
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory, collect_param_names_need_sync
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedSACFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # SAC-specific initialization
        self.replay_buffer = None
        self.target_model = None
        self.entropy_temp = None
        self.demo_buffer = None
        self.alpha_optimizer = None
        self.update_step = 0
        self.enable_drq = bool(getattr(self.cfg.actor, "enable_drq", False))
        self.use_dsrl_flat_replay = False
        self._local_new_transitions = 0

    def init_worker(self):
        self.setup_model_and_optimizer(initialize_target=True)
        self.setup_sac_components()
        if self.use_dsrl and self.cfg.actor.get("compile_model", False):
            raise ValueError(
                "DSRL target-shadow parameter names do not support "
                "actor.compile_model=True"
            )
        self.soft_update_target_model(tau=1.0)
        if self.use_dsrl:
            self._init_target_shadow()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(
                self.model, mode="default"
            )  # max-autotune-no-cudagraphs
            self.target_model = torch.compile(self.target_model, mode="default")

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        """Setup model, lr_scheduler, optimizer and grad_scaler."""
        """Add initializing target model logic."""
        module = self.model_provider_func()
        if initialize_target:
            target_module = self.model_provider_func()

        # Enable gradient checkpointing if configured
        if self.cfg.actor.model.get("gradient_checkpointing", False):
            self.logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
            if initialize_target:
                target_module.gradient_checkpointing_enable()
        else:
            self.logger.info("[FSDP] Gradient checkpointing is disabled")

        # Record the original trainable parameter names before FSDP wrapping.
        # Persistent buffer names are also recorded for selective weight syncing.
        self.param_names_need_sync = collect_param_names_need_sync(module)

        # build model, optimizer, lr_scheduler, grad_scaler
        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        # When precision is null (e.g. Pi0), detect actual dtype from wrapped model
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype
        if initialize_target:
            self.target_model = self._strategy.wrap_model(
                model=target_module, device_mesh=self._device_mesh
            )
            self.target_model.requires_grad_(False)
            self.target_model_initialized = True

        self.use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        use_dsrl = self.use_dsrl
        if use_dsrl:
            # DSRL: separate actor/critic encoders into different optimizer groups
            param_filters = {
                "critic": ["critic_image_encoder", "critic_state_encoder", "q_head"]
            }
        else:
            param_filters = {"critic": ["encoders", "encoder", "q_head", "state_proj"]}
        filtered_optim_config = {"critic": self.cfg.actor.critic_optim}
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters=param_filters,
            filtered_optim_config=filtered_optim_config,
        )
        self.optimizer = optimizers[0]
        self.qf_optimizer = optimizers[1]

        # SAC alpha
        # Initialize temperature parameter for automatic entropy tuning
        alpha_type = self.cfg.algorithm.entropy_tuning.get(
            "alpha_type", "softplus"
        )  # supported type: ["softplus","exp","fixed_alpha"]
        self.entropy_temp = EntropyTemperature(
            initial_alpha=self.cfg.algorithm.entropy_tuning.get("initial_alpha", 0.01),
            alpha_type=alpha_type,
            device=self.device,
            dtype=self.torch_dtype,
        )
        if alpha_type != "fixed_alpha":
            self.target_entropy = self.cfg.algorithm.entropy_tuning.get(
                "target_entropy",
                -self.cfg.actor.model.action_dim,
            )

            self.alpha_optimizer = torch.optim.Adam(
                self.entropy_temp.parameters(),
                lr=self.cfg.algorithm.entropy_tuning.optim.lr,
            )

        self.build_lr_schedulers()

        self.grad_scaler = self.build_grad_scaler(
            self.cfg.actor.fsdp_config.grad_scaler
        )

    def build_lr_schedulers(self):
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )
        if self.alpha_optimizer is not None:
            self.alpha_lr_scheduler = self.build_lr_scheduler(
                self.alpha_optimizer, self.cfg.algorithm.entropy_tuning.optim
            )

    def setup_sac_components(self):
        """Initialize SAC-specific components"""
        # Initialize replay buffer
        seed = self.cfg.actor.get("seed", 1234)
        replay_cfg = self.cfg.algorithm.replay_buffer
        replay_type = replay_cfg.get("type", "trajectory")
        self.use_dsrl_flat_replay = bool(
            self.use_dsrl and replay_type == "dsrl_transition"
        )
        if replay_type == "dsrl_transition" and not self.use_dsrl:
            raise ValueError("dsrl_transition replay requires openpi.use_dsrl=True")

        if self.use_dsrl_flat_replay:
            self.replay_buffer = DSRLTransitionReplayBuffer(
                capacity=replay_cfg.capacity,
                seed=seed,
                rank=self._rank,
                world_size=self._world_size,
                schema_version=replay_cfg.get("schema_version", 1),
            )
        else:
            auto_save_path = replay_cfg.get("auto_save_path", None)
            if auto_save_path is None:
                auto_save_path = os.path.join(
                    self.cfg.runner.logger.log_path,
                    f"replay_buffer/rank_{self._rank}",
                )
            else:
                auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
            self.replay_buffer = TrajectoryReplayBuffer(
                seed=seed,
                enable_cache=replay_cfg.enable_cache,
                cache_size=replay_cfg.cache_size,
                sample_window_size=replay_cfg.sample_window_size,
                auto_save=replay_cfg.get("auto_save", False),
                auto_save_path=auto_save_path,
                trajectory_format=replay_cfg.get("trajectory_format", "pt"),
            )

        min_demo_buffer_size = 0
        if self.cfg.algorithm.get("demo_buffer", None) is not None:
            if self.use_dsrl_flat_replay:
                raise ValueError("DSRL transition replay does not support demo_buffer")
            auto_save_path = self.cfg.algorithm.demo_buffer.get("auto_save_path", None)
            if auto_save_path is None:
                auto_save_path = os.path.join(
                    self.cfg.runner.logger.log_path, f"demo_buffer/rank_{self._rank}"
                )
            else:
                auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
            self.demo_buffer = TrajectoryReplayBuffer(
                seed=seed,
                enable_cache=self.cfg.algorithm.demo_buffer.enable_cache,
                cache_size=self.cfg.algorithm.demo_buffer.cache_size,
                sample_window_size=self.cfg.algorithm.demo_buffer.sample_window_size,
                auto_save=self.cfg.algorithm.demo_buffer.get("auto_save", False),
                auto_save_path=auto_save_path,
                trajectory_format="pt",
            )
            min_demo_buffer_size = self.cfg.algorithm.demo_buffer.min_buffer_size
            if self.cfg.algorithm.demo_buffer.get("load_path", None) is not None:
                self.demo_buffer.load_checkpoint(
                    self.cfg.algorithm.demo_buffer.load_path,
                    is_distributed=True,
                    local_rank=self._rank,
                    world_size=self._world_size,
                )

        if self.use_dsrl_flat_replay and replay_cfg.get("enable_preload", False):
            raise ValueError("DSRL transition replay does not support preload sampling")
        if replay_cfg.get("enable_preload", False):
            buffer_dataset_cls = PreloadReplayBufferDataset
        else:
            buffer_dataset_cls = ReplayBufferDataset
        min_replay_buffer_size = (
            1 if self.use_dsrl_flat_replay else replay_cfg.min_buffer_size
        )
        self.buffer_dataset = buffer_dataset_cls(
            replay_buffer=self.replay_buffer,
            demo_buffer=self.demo_buffer,
            batch_size=self.cfg.actor.global_batch_size // self._world_size,
            min_replay_buffer_size=min_replay_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
            prefetch_size=replay_cfg.get("prefetch_size", 10),
        )
        self.buffer_dataloader = DataLoader(
            self.buffer_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            collate_fn=replay_buffer_collate_fn,
        )
        self.buffer_dataloader_iter = iter(self.buffer_dataloader)

        self.critic_actor_ratio = self.cfg.algorithm.get("critic_actor_ratio", 1)
        self.critic_subsample_size = self.cfg.algorithm.get("critic_subsample_size", -1)
        self.critic_sample_generator = torch.Generator(self.device)
        self.critic_sample_generator.manual_seed(seed)

        self.target_update_type = self.cfg.algorithm.get("target_update_type", "all")
        assert self.target_update_type in ["all", "q_head_only"], (
            f"{self.target_update_type=} is not suppported!"
        )
        if self.use_dsrl_flat_replay:
            openpi_cfg = self.cfg.actor.model.openpi
            action_horizon = int(openpi_cfg.get("action_horizon", 50))
            num_action_chunks = int(self.cfg.actor.model.num_action_chunks)
            rollout_epoch = int(self.cfg.env.train.rollout_epoch)
            if not 0 < num_action_chunks <= action_horizon:
                raise ValueError(
                    "DSRL requires 0 < num_action_chunks <= action_horizon, "
                    f"got N={num_action_chunks}, H={action_horizon}"
                )
            if rollout_epoch != 1:
                raise ValueError(
                    "DSRL transition projection currently requires "
                    f"env.train.rollout_epoch=1, got {rollout_epoch}"
                )
            if bool(self.cfg.env.train.get("auto_reset", False)):
                raise ValueError(
                    "DSRL transition projection requires env.train.auto_reset=False"
                )
            if bool(self.cfg.env.train.get("ignore_terminations", False)):
                raise ValueError(
                    "DSRL transition projection requires "
                    "env.train.ignore_terminations=False"
                )
            if not 0.0 < float(self.cfg.algorithm.gamma) <= 1.0:
                raise ValueError("DSRL requires gamma in (0, 1]")
            if int(replay_cfg.capacity) < int(replay_cfg.warmup_size):
                raise ValueError(
                    "DSRL replay capacity must be at least warmup_size, "
                    f"got {replay_cfg.capacity} < {replay_cfg.warmup_size}"
                )
            if self.cfg.algorithm.get("bootstrap_type", "standard") != "standard":
                raise ValueError(
                    "DSRL transition replay requires bootstrap_type=standard"
                )
            if self.cfg.algorithm.get("utd_ratio", 0) <= 0:
                raise ValueError("DSRL transition replay requires utd_ratio > 0")
            if replay_cfg.get("warmup_size", 0) <= 0:
                raise ValueError("DSRL transition replay requires warmup_size > 0")
        self._local_new_transitions = 0

    @staticmethod
    def _is_dsrl_target_q_parameter(name: str) -> bool:
        parts = name.split(".")
        return any(
            component in parts
            for component in (
                "critic_image_encoder",
                "critic_state_encoder",
                "q_head",
            )
        )

    def _named_dsrl_target_q_parameters(self, model) -> dict[str, torch.nn.Parameter]:
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if self._is_dsrl_target_q_parameter(name)
        }
        components = {
            component
            for name in parameters
            for component in (
                "critic_image_encoder",
                "critic_state_encoder",
                "q_head",
            )
            if component in name.split(".")
        }
        expected = {"critic_image_encoder", "critic_state_encoder", "q_head"}
        if components != expected:
            raise RuntimeError(
                f"Incomplete DSRL target-Q allowlist: {sorted(components)}"
            )
        return parameters

    def _init_target_shadow(self):
        """Create persistent float32 shadow of target model parameters.

        bfloat16 has only 7 mantissa bits (ULP ~0.002 at magnitude 0.3).
        With tau=0.005, per-step EMA delta can be smaller than ULP/2, so
        storing back to bf16 each step rounds away the update. The shadow
        keeps the accumulated EMA state in float32 (ULP ~3.6e-8) across
        steps, preventing precision loss.
        """
        target_parameters = self._named_dsrl_target_q_parameters(self.target_model)
        self._target_shadow_f32 = {
            name: parameter.data.float().clone()
            for name, parameter in target_parameters.items()
        }

    def soft_update_target_model(self, tau: Optional[float] = None):
        """Soft update target model parameters.

        For DSRL (bfloat16 models), uses a persistent float32 shadow buffer
        to prevent EMA precision loss. For non-DSRL SAC, uses direct EMA
        on model parameters.
        """
        if tau is None:
            tau = self.cfg.algorithm.tau

        assert self.target_model_initialized

        with torch.no_grad():
            if self.use_dsrl:
                online_parameters = self._named_dsrl_target_q_parameters(self.model)
                target_parameters = self._named_dsrl_target_q_parameters(
                    self.target_model
                )
                if set(online_parameters) != set(target_parameters):
                    raise RuntimeError("Online/target DSRL target-Q names differ")
                if hasattr(self, "_target_shadow_f32") and set(
                    self._target_shadow_f32
                ) != set(target_parameters):
                    raise RuntimeError("DSRL FP32 target shadow names differ")

                for name, target_param in target_parameters.items():
                    online_param = online_parameters[name]
                    use_ema = (
                        self.target_update_type == "all" or "q_head" in name.split(".")
                    )
                    if hasattr(self, "_target_shadow_f32"):
                        shadow = self._target_shadow_f32[name]
                        if (
                            shadow.shape != target_param.shape
                            or shadow.dtype != torch.float32
                        ):
                            raise RuntimeError(
                                f"Invalid DSRL FP32 shadow for {name}: "
                                f"{tuple(shadow.shape)}/{shadow.dtype}"
                            )
                        if use_ema:
                            shadow.mul_(1.0 - tau).add_(
                                online_param.data.float(), alpha=tau
                            )
                        else:
                            shadow.copy_(online_param.data.float())
                        target_param.data.copy_(shadow.to(target_param.data.dtype))
                    elif use_ema:
                        target_param.data.mul_(1.0 - tau)
                        target_param.data.add_(online_param.data * tau)
                    else:
                        target_param.data.copy_(online_param.data)
                return

            if not hasattr(self, "_target_shadow_f32"):
                # Non-DSRL path (or before shadow init): direct EMA update
                for (name1, online_param), (name2, target_param) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    assert name1 == name2
                    if "q_head" not in name1:
                        if self.target_update_type == "all":
                            target_param.data.mul_(1.0 - tau)
                            target_param.data.add_(online_param.data * tau)
                        else:
                            target_param.data.mul_(0.0)
                            target_param.data.add_(online_param.data)
                    else:
                        target_param.data.mul_(1.0 - tau)
                        target_param.data.add_(online_param.data * tau)

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

        if self.use_dsrl_flat_replay:
            openpi_cfg = self.cfg.actor.model.openpi
            for trajectory in recv_list:
                projected = project_dsrl_trajectory(
                    trajectory,
                    action_horizon=openpi_cfg.get("action_horizon", 50),
                    latent_dim=openpi_cfg.dsrl_action_noise_dim,
                    state_dim=openpi_cfg.dsrl_state_dim,
                    num_action_chunks=self.cfg.actor.model.num_action_chunks,
                    gamma=self.cfg.algorithm.gamma,
                )
                self._local_new_transitions += self.replay_buffer.add_batch(projected)
        else:
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

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        if self.use_dsrl_flat_replay:
            discount = batch["discounts"].to(self.torch_dtype)
            rewards_for_bootstrap = batch["rewards"].to(self.torch_dtype)
            continuations = batch["continuations"].to(self.torch_dtype)
            terminations = batch["terminations"].to(self.torch_dtype)
        elif use_dsrl:
            num_action_chunks = self.cfg.actor.model.get("num_action_chunks", 1)
            discount = self.cfg.algorithm.gamma**num_action_chunks
            rewards_for_bootstrap = batch["rewards"][:, 0:1].to(self.torch_dtype)
            continuations = None
            terminations = batch["terminations"].to(self.torch_dtype)
        else:
            discount = self.cfg.algorithm.gamma
            rewards_for_bootstrap = (
                batch["rewards"].sum(dim=-1, keepdim=True).to(self.torch_dtype)
            )
            continuations = None
            terminations = batch["terminations"].to(self.torch_dtype)

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]

        with torch.no_grad():
            kwargs = {}
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                kwargs["temperature"] = (
                    self.cfg.rollout.sampling_params.temperature_train
                )
            if use_dsrl:
                kwargs["train"] = True
            next_state_actions, next_state_log_pi, shared_feature = self.model(
                forward_type=ForwardType.SAC, obs=next_obs, **kwargs
            )
            if next_state_log_pi.ndim == 1:
                next_state_log_pi = next_state_log_pi.unsqueeze(-1)
            next_state_log_pi = next_state_log_pi.sum(dim=-1, keepdim=True)
            if not use_crossq:
                dsrl_kwargs = {"train": True} if use_dsrl else {}
                all_qf_next_target = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_obs,
                    actions=next_state_actions,
                    shared_feature=None,
                    **dsrl_kwargs,
                )
                if self.critic_subsample_size > 0:
                    sample_idx = torch.randint(
                        0,
                        all_qf_next_target.shape[-1],
                        (self.critic_subsample_size,),
                        generator=self.critic_sample_generator,
                        device=self.device,
                    )
                    all_qf_next_target = all_qf_next_target.index_select(
                        dim=-1, index=sample_idx
                    )

                if agg_q == "min":
                    qf_next_target, _ = torch.min(
                        all_qf_next_target, dim=1, keepdim=True
                    )
                elif agg_q == "mean":
                    qf_next_target = torch.mean(all_qf_next_target, dim=1, keepdim=True)

                if self.cfg.algorithm.get("backup_entropy", True):
                    qf_next_target = (
                        qf_next_target - self.entropy_temp.alpha * next_state_log_pi
                    )
                    qf_next_target = qf_next_target.to(dtype=self.torch_dtype)
                if self.use_dsrl_flat_replay:
                    target_q_values = (
                        rewards_for_bootstrap
                        + continuations * discount * qf_next_target
                    )
                elif bootstrap_type == "always":
                    target_q_values = (
                        rewards_for_bootstrap + discount * qf_next_target
                    )  # [bsz, 1]
                elif bootstrap_type == "standard":
                    target_q_values = (
                        rewards_for_bootstrap
                        + (~(terminations.any(dim=-1, keepdim=True)))
                        * discount
                        * qf_next_target
                    )  # [bsz, 1]
                else:
                    raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        if not use_crossq:
            dsrl_kwargs = {"train": True} if use_dsrl else {}
            all_data_q_values = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions,
                **dsrl_kwargs,
            )
        else:
            all_data_q_values, all_qf_next = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=actions,
                next_obs=next_obs,
                next_actions=next_state_actions,
            )

            all_qf_next = all_qf_next.detach()
            if agg_q == "min":
                qf_next, _ = torch.min(all_qf_next, dim=1, keepdim=True)
            elif agg_q == "mean":
                qf_next = torch.mean(all_qf_next, dim=1, keepdim=True)
            if self.cfg.algorithm.get("backup_entropy", True):
                qf_next = qf_next - self.entropy_temp.alpha * next_state_log_pi
                qf_next = qf_next.to(dtype=self.torch_dtype)

            if self.use_dsrl_flat_replay:
                target_q_values = (
                    rewards_for_bootstrap + continuations * discount * qf_next
                )
            elif bootstrap_type == "always":
                target_q_values = rewards_for_bootstrap + discount * qf_next  # [bsz, 1]
            elif bootstrap_type == "standard":
                target_q_values = (
                    rewards_for_bootstrap
                    + (~(terminations.any(dim=-1, keepdim=True))) * discount * qf_next
                )  # [bsz, 1]
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        # Align dtype: bool ops with Python floats promote to float32,
        # which can mismatch with bfloat16 model outputs.
        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        return critic_loss, {"q_data": all_data_q_values.mean().item()}

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        if "actor_agg_q" in self.cfg.algorithm:
            agg_q = self.cfg.algorithm["actor_agg_q"]
        else:
            agg_q = self.cfg.algorithm.get("agg_q", "min")

        curr_obs = batch["curr_obs"]
        kwargs = {}
        if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
            kwargs["temperature"] = self.cfg.rollout.sampling_params.temperature_train
        if self.use_dsrl:
            kwargs["train"] = True
        pi, log_pi, shared_feature = self.model(
            forward_type=ForwardType.SAC, obs=curr_obs, **kwargs
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)  # sum over the chunk dimension
        if not use_crossq:
            dsrl_kwargs = {"train": True} if self.use_dsrl else {}
            all_qf_pi = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=pi,
                shared_feature=None,
                detach_encoder=True,
                **dsrl_kwargs,
            )
        else:
            all_qf_pi, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=pi,
                next_obs=None,
                next_actions=None,
                shared_feature=None,
                detach_encoder=True,
            )
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(all_qf_pi.shape[-1])
        }
        if agg_q == "min":
            qf_pi, _ = torch.min(all_qf_pi, dim=1, keepdim=True)
        elif agg_q == "mean":
            qf_pi = torch.mean(all_qf_pi, dim=1, keepdim=True)
        metrics["q_pi"] = qf_pi.mean().item()
        actor_loss = ((self.entropy_temp.alpha * log_pi) - qf_pi).mean()

        entropy = -log_pi.mean()
        return actor_loss, entropy, metrics

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        curr_obs = batch["curr_obs"]
        with torch.no_grad():
            kwargs = {}
            if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
                kwargs["temperature"] = (
                    self.cfg.rollout.sampling_params.temperature_train
                )
            if self.use_dsrl:
                kwargs["train"] = True
            _, log_pi, _ = self.model(
                forward_type=ForwardType.SAC, obs=curr_obs, **kwargs
            )
            if log_pi.ndim == 1:
                log_pi = log_pi.unsqueeze(-1)
            log_pi = log_pi.sum(dim=-1, keepdim=True)

        alpha = self.entropy_temp.compute_alpha()
        alpha_loss = -alpha * (log_pi.mean() + self.target_entropy)
        return alpha_loss

    def _clear_dsrl_critic_grads_before_actor_clip(self):
        if self.use_dsrl:
            self.qf_optimizer.zero_grad(set_to_none=True)

    def _clear_dsrl_actor_grads_before_critic_clip(self):
        if self.use_dsrl:
            self.optimizer.zero_grad(set_to_none=True)

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self, train_actor: bool = True):
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )

        with self.worker_timer("sample"):
            global_batch = next(self.buffer_dataloader_iter)

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )

        # Move micro-batches to device and apply DRQ for all SAC passes.
        for i, batch in enumerate(train_micro_batch_list):
            batch = put_tensor_device(batch, device=self.device)
            if self.enable_drq:
                drq.apply_drq(batch["curr_obs"], pad=4)
                drq.apply_drq(batch["next_obs"], pad=4)
            train_micro_batch_list[i] = batch

        self.qf_optimizer.zero_grad()
        self._clear_dsrl_actor_grads_before_critic_clip()
        gbs_critic_loss = []
        all_critic_metrics = {}
        for batch in train_micro_batch_list:
            critic_loss, critic_metrics = self.forward_critic(batch)
            critic_loss = critic_loss / self.gradient_accumulation
            critic_loss.backward()
            gbs_critic_loss.append(critic_loss.item() * self.gradient_accumulation)
            append_to_dict(all_critic_metrics, critic_metrics)
        all_critic_metrics = {
            f"critic/{key}": np.mean(value) for key, value in all_critic_metrics.items()
        }
        qf_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.critic_optim.clip_grad
        )

        self.qf_optimizer.step()
        self.qf_lr_scheduler.step()

        metrics_data = {
            "sac/critic_loss": np.mean(gbs_critic_loss),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": qf_grad_norm,
            **all_critic_metrics,
        }

        if self.update_step % self.critic_actor_ratio == 0 and train_actor:
            self.optimizer.zero_grad()
            gbs_actor_loss = []
            gbs_entropy = []
            all_actor_metrics = {}
            for batch in train_micro_batch_list:
                actor_loss, entropy, q_metrics = self.forward_actor(batch)
                actor_loss = actor_loss / self.gradient_accumulation
                actor_loss.backward()
                gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)
                gbs_entropy.append(entropy.item())
                append_to_dict(all_actor_metrics, q_metrics)
            # The DSRL actor needs dQ/da through q_head, but q_head itself
            # belongs to qf_optimizer. Remove those incidental parameter
            # gradients before the FSDP-wide actor grad norm.
            self._clear_dsrl_critic_grads_before_actor_clip()
            all_actor_metrics = {
                f"actor/{key}": np.mean(value)
                for key, value in all_actor_metrics.items()
            }
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.lr_scheduler.step()

            # Update temperature parameter if using automatic entropy tuning
            gbs_alpha_loss = [0]
            alpha_grad_norm = 0
            if self.alpha_optimizer is not None:
                self.alpha_optimizer.zero_grad()
                gbs_alpha_loss = []
                for batch in train_micro_batch_list:
                    alpha_loss = self.forward_alpha(batch) / self.gradient_accumulation
                    alpha_loss.backward()
                    gbs_alpha_loss.append(
                        alpha_loss.item() * self.gradient_accumulation
                    )
                torch.distributed.all_reduce(
                    self.entropy_temp.base_alpha.grad,
                    op=torch.distributed.ReduceOp.AVG,
                )
                alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.entropy_temp.base_alpha,
                    self.cfg.algorithm.entropy_tuning.optim.clip_grad,
                )
                self.alpha_optimizer.step()
                self.alpha_lr_scheduler.step()

            # Collect metrics
            metrics_data.update(
                {
                    "sac/actor_loss": np.mean(gbs_actor_loss),
                    "sac/alpha_loss": np.mean(gbs_alpha_loss),
                    "sac/alpha": self.entropy_temp.alpha,
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                    "actor/entropy": np.mean(gbs_entropy),
                    "alpha/grad_norm": alpha_grad_norm,
                    **all_actor_metrics,
                }
            )
        # Soft update target network
        if (
            self.target_model_initialized
            and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0
        ):
            self.soft_update_target_model()

        return metrics_data

    def process_train_metrics(self, metrics):
        replay_buffer_stats = self.replay_buffer.get_stats()
        replay_buffer_stats = {
            f"replay_buffer/{key}": value for key, value in replay_buffer_stats.items()
        }
        append_to_dict(metrics, replay_buffer_stats)

        if self.demo_buffer is not None:
            demo_buffer_stats = self.demo_buffer.get_stats()
            demo_buffer_stats = {
                f"demo_buffer/{key}": value for key, value in demo_buffer_stats.items()
            }
            append_to_dict(metrics, demo_buffer_stats)
        # Average metrics across updates
        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0:
                # Convert tensor values to CPU and detach before computing mean
                cpu_values = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        cpu_values.append(v.detach().cpu().item())
                    else:
                        cpu_values.append(v)
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                # Handle single values
                if isinstance(value, torch.Tensor):
                    mean_metric_dict[key] = value.detach().cpu().item()
                else:
                    mean_metric_dict[key] = value

        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict

    @staticmethod
    def _phase_buffers(model) -> list[torch.Tensor]:
        return [
            buffer
            for name, buffer in model.named_buffers()
            if name.split(".")[-1] == "dsrl_policy_phase"
        ]

    def _get_dsrl_policy_phase(self) -> int:
        return self._get_dsrl_policy_phase_from_model(self.model, "online")

    def _get_dsrl_policy_phase_from_model(self, model, model_name: str) -> int:
        buffers = self._phase_buffers(model)
        if len(buffers) != 1:
            raise RuntimeError(
                f"Expected one {model_name} DSRL phase buffer, found {len(buffers)}"
            )
        return int(buffers[0].item())

    def _set_dsrl_policy_phase(self, phase: int):
        if phase not in (0, 1):
            raise ValueError(f"Invalid DSRL policy phase: {phase}")
        for model_name, model in (
            ("online", self.model),
            ("target", self.target_model),
        ):
            buffers = self._phase_buffers(model)
            if len(buffers) != 1:
                raise RuntimeError(
                    f"Expected one {model_name} DSRL phase buffer, found {len(buffers)}"
                )
            buffers[0].fill_(phase)

    @staticmethod
    def _set_legacy_phase_load(model, enabled: bool):
        found = 0
        for module in model.modules():
            if hasattr(module, "_load_missing_dsrl_phase_as_learned"):
                module._load_missing_dsrl_phase_as_learned = enabled
                found += 1
        if found != 1:
            raise RuntimeError(
                f"Expected one OpenPI DSRL phase compatibility module, found {found}"
            )

    def _consume_global_dsrl_replay_counts(self) -> tuple[int, int, int]:
        counts = torch.tensor(
            [self._local_new_transitions, len(self.replay_buffer)],
            dtype=torch.long,
            device=self.device,
        )
        min_resident = torch.tensor(
            len(self.replay_buffer), dtype=torch.long, device=self.device
        )
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(min_resident, op=torch.distributed.ReduceOp.MIN)
        self._local_new_transitions = 0
        return (
            int(counts[0].item()),
            int(counts[1].item()),
            int(min_resident.item()),
        )

    @Worker.timer("run_training")
    def run_training(self):
        """SAC training using replay buffer"""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        dsrl_counts = None
        if self.use_dsrl_flat_replay:
            global_new, global_resident, min_local_resident = (
                self._consume_global_dsrl_replay_counts()
            )
            dsrl_counts = {
                "sac/global_new_transitions": [global_new],
                "sac/global_resident_transitions": [global_resident],
            }
            warmup_size = self.cfg.algorithm.replay_buffer.warmup_size
            if global_resident < warmup_size:
                self.log_on_first_rank(
                    "DSRL replay warm-up: "
                    f"{global_resident} < {warmup_size} global transitions"
                )
                return self.process_train_metrics(dsrl_counts)
            if min_local_resident <= 0:
                raise RuntimeError(
                    "Global DSRL replay is warm but at least one actor rank is empty"
                )
            self._set_dsrl_policy_phase(1)
            update_epoch = int(self.cfg.algorithm.utd_ratio) * global_new
            dsrl_counts["sac/planned_optimizer_updates"] = [update_epoch]
            if update_epoch <= 0:
                return self.process_train_metrics(dsrl_counts)
            train_actor = True
        else:
            # Check if the legacy replay buffer has enough trajectories.
            min_buffer_size = self.cfg.algorithm.replay_buffer.get(
                "min_buffer_size", 100
            )
            if not self.replay_buffer.is_ready(min_buffer_size):
                self.log_on_first_rank(
                    f"Replay buffer size {len(self.replay_buffer)} "
                    f"< {min_buffer_size}, skipping training"
                )
                return {}

            # Delay actor training until the legacy buffer has enough trajectories.
            train_actor_steps = self.cfg.algorithm.get("train_actor_steps", 0)
            train_actor_steps = max(min_buffer_size, train_actor_steps)
            train_actor = self.replay_buffer.is_ready(train_actor_steps)
            update_epoch = self.cfg.algorithm.get("update_epoch", 1)

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
        metrics = dsrl_counts or {}

        for _ in range(update_epoch):
            metrics_data = self.update_one_epoch(train_actor=train_actor)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        mean_metric_dict = self.process_train_metrics(metrics)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self):
        """
        SAC doesn't compute advantages/returns like PPO.
        This method is kept for compatibility but returns empty metrics.
        """
        return {}

    def _dsrl_trainer_state_path(self, base_path: str) -> str:
        return os.path.join(
            base_path,
            "sac_components",
            f"dsrl_trainer_state_rank_{self._rank}.pt",
        )

    def _save_dsrl_trainer_state(self, save_base_path: str):
        target_parameters = self._named_dsrl_target_q_parameters(self.target_model)
        if not hasattr(self, "_target_shadow_f32"):
            raise RuntimeError("DSRL checkpoint requires an initialized FP32 shadow")
        if set(self._target_shadow_f32) != set(target_parameters):
            raise RuntimeError("DSRL checkpoint shadow names do not match target-Q")

        shadow_cpu = {}
        for name, target_parameter in target_parameters.items():
            shadow = self._target_shadow_f32[name]
            if shadow.shape != target_parameter.shape or shadow.dtype != torch.float32:
                raise RuntimeError(
                    f"Invalid DSRL checkpoint shadow for {name}: "
                    f"{tuple(shadow.shape)}/{shadow.dtype}"
                )
            if not torch.equal(
                shadow.to(target_parameter.dtype), target_parameter.data
            ):
                raise RuntimeError(
                    f"DSRL FP32 shadow no longer rounds to target parameter {name}"
                )
            shadow_cpu[name] = shadow.detach().cpu().clone()

        state = {
            "schema_version": 1,
            "rank": self._rank,
            "world_size": self._world_size,
            "update_step": self.update_step,
            "policy_phase": self._get_dsrl_policy_phase(),
            "pending_local_new_transitions": self._local_new_transitions,
            "flat_replay": self.use_dsrl_flat_replay,
            "target_shadow_f32": shadow_cpu,
        }
        target_path = self._dsrl_trainer_state_path(save_base_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        temp_path = f"{target_path}.tmp"
        torch.save(state, temp_path)
        os.replace(temp_path, target_path)

    def _load_dsrl_trainer_state(self, load_base_path: str):
        state_path = self._dsrl_trainer_state_path(load_base_path)
        if not os.path.isfile(state_path):
            self._init_target_shadow()
            self.update_step = 0
            self._local_new_transitions = 0
            self._set_dsrl_policy_phase(1)
            self.logger.warning(
                "Legacy DSRL checkpoint has no FP32 target shadow/trainer state. "
                "Rebuilt critic-only shadow from the loaded BF16 target and "
                "restored learned-policy phase; resume is compatible but not "
                "bitwise continuous."
            )
            return

        state = torch.load(state_path, map_location="cpu", weights_only=True)
        expected_layout = {
            "schema_version": 1,
            "rank": self._rank,
            "world_size": self._world_size,
            "flat_replay": self.use_dsrl_flat_replay,
        }
        mismatches = {
            key: (state.get(key), expected)
            for key, expected in expected_layout.items()
            if state.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"DSRL trainer-state layout mismatch: {mismatches}")

        target_parameters = self._named_dsrl_target_q_parameters(self.target_model)
        saved_shadow = state.get("target_shadow_f32", {})
        if set(saved_shadow) != set(target_parameters):
            missing = sorted(set(target_parameters) - set(saved_shadow))
            extra = sorted(set(saved_shadow) - set(target_parameters))
            raise ValueError(
                f"DSRL trainer-state shadow names mismatch: {missing=}, {extra=}"
            )
        restored_shadow = {}
        for name, target_parameter in target_parameters.items():
            shadow = saved_shadow[name]
            if shadow.dtype != torch.float32 or shadow.shape != target_parameter.shape:
                raise ValueError(
                    f"Invalid saved DSRL shadow for {name}: "
                    f"{tuple(shadow.shape)}/{shadow.dtype}"
                )
            shadow = shadow.to(device=target_parameter.device).contiguous()
            if not torch.equal(
                shadow.to(target_parameter.dtype), target_parameter.data
            ):
                raise ValueError(
                    f"Saved DSRL shadow does not match loaded target {name}"
                )
            restored_shadow[name] = shadow
        self._target_shadow_f32 = restored_shadow

        phase = int(state["policy_phase"])
        loaded_online_phase = self._get_dsrl_policy_phase()
        loaded_target_phase = self._get_dsrl_policy_phase_from_model(
            self.target_model, "target"
        )
        if loaded_online_phase != phase or loaded_target_phase != phase:
            raise ValueError(
                "DSRL model/trainer phase mismatch: "
                f"online={loaded_online_phase}, target={loaded_target_phase}, "
                f"trainer={phase}"
            )
        update_step = int(state["update_step"])
        pending_new = int(state.get("pending_local_new_transitions", 0))
        if update_step < 0 or pending_new < 0:
            raise ValueError(
                f"Invalid DSRL trainer counters: {update_step=}, {pending_new=}"
            )
        self.update_step = update_step
        self._local_new_transitions = pending_new
        self._set_dsrl_policy_phase(phase)

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        # Save model
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            save_full_model_weights=self.cfg.actor.fsdp_config.get(
                "save_full_model_weights", True
            ),
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # Save sac components
        # save alpha
        if self.alpha_optimizer is not None:
            alpha_save_path = os.path.join(save_base_path, "sac_components/alpha")
            self._strategy.save_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                save_path=alpha_save_path,
                save_full_model_weights=False,
            )

        # save target model
        target_model_save_path = os.path.join(
            save_base_path, "sac_components/target_model"
        )
        os.makedirs(target_model_save_path, exist_ok=True)
        target_model_state_dict = self._strategy.get_model_state_dict(
            self.target_model, cpu_offload=False, full_state_dict=True
        )
        torch.save(
            target_model_state_dict,
            os.path.join(target_model_save_path, f"checkpoint_rank_{self._rank}.pt"),
        )
        if self.use_dsrl:
            self._save_dsrl_trainer_state(save_base_path)

        # save replay buffer
        buffer_save_path = os.path.join(
            save_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.save_checkpoint(buffer_save_path)

    def load_checkpoint(self, load_base_path):
        legacy_dsrl_checkpoint = self.use_dsrl and not os.path.isfile(
            self._dsrl_trainer_state_path(load_base_path)
        )
        # load model
        if legacy_dsrl_checkpoint:
            self._set_legacy_phase_load(self.model, True)
        try:
            self._strategy.load_checkpoint(
                model=self.model,
                optimizers=[self.optimizer, self.qf_optimizer],
                lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
                load_path=load_base_path,
                checkpoint_format="local_shard"
                if self.cfg.actor.fsdp_config.use_orig_params
                else "dcp",
            )
        finally:
            if legacy_dsrl_checkpoint:
                self._set_legacy_phase_load(self.model, False)

        # load alpha
        if self.alpha_optimizer is not None:
            alpha_load_path = os.path.join(load_base_path, "sac_components/alpha")
            self._strategy.load_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                load_path=alpha_load_path,
            )

        # load target model
        target_model_load_path = os.path.join(
            load_base_path, "sac_components/target_model"
        )
        target_model_state_dict = torch.load(
            os.path.join(target_model_load_path, f"checkpoint_rank_{self._rank}.pt"),
            map_location="cpu",
            weights_only=True,
        )
        if legacy_dsrl_checkpoint:
            self._set_legacy_phase_load(self.target_model, True)
        try:
            self._strategy.load_model_with_state_dict(
                self.target_model,
                target_model_state_dict,
                cpu_offload=False,
                full_state_dict=True,
            )
        finally:
            if legacy_dsrl_checkpoint:
                self._set_legacy_phase_load(self.target_model, False)
        if self.use_dsrl:
            self._load_dsrl_trainer_state(load_base_path)

        # load replay buffer
        buffer_load_path = os.path.join(
            load_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.load_checkpoint(buffer_load_path)
