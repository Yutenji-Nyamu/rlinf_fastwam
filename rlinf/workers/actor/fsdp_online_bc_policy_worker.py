# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Online success BC using RLinf's existing supervised FSDP update machinery."""

from dataclasses import replace
from pathlib import Path

import torch

from rlinf.data.online_bc import SuccessReplay
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Worker
from rlinf.utils.online_bc_resume_audit import audit_online_bc_resume
from rlinf.utils.metric_utils import compute_split_num
from rlinf.workers.actor.fsdp_dagger_policy_worker import EmbodiedDAGGERFSDPPolicy


class EmbodiedOnlineBCFSDPPolicy(EmbodiedDAGGERFSDPPolicy):
    def init_worker(self):
        super().init_worker()
        trainable = [
            (name, p.numel())
            for name, p in self.model.named_parameters()
            if p.requires_grad
        ]
        if self.cfg.actor.model.openpi.train_expert_only and any(
            "paligemma_with_expert.paligemma." in name for name, _ in trainable
        ):
            raise RuntimeError(
                "Expert-only BC unexpectedly has trainable VLM parameters."
            )
        self.log_info(
            f"Online BC trainable parameters: {sum(n for _, n in trainable):,}; "
            f"expert_only={self.cfg.actor.model.openpi.train_expert_only}"
        )

    def setup_dagger_components(self):
        # Only the generic supervised update loop is inherited; no teacher,
        # intervention extraction, LeRobot collector or DAgger loss is used.
        bc = self.cfg.algorithm.online_bc
        self.checkpoint_format = self.cfg.actor.fsdp_config.get(
            "checkpoint_format", "local_shard"
        )
        if self.checkpoint_format not in ("local_shard", "dcp"):
            raise ValueError("Online BC checkpoint_format must be local_shard or dcp.")
        self.demo_weight = float(bc.demo_weight)
        if self.demo_weight < 0:
            raise ValueError("online_bc.demo_weight must be non-negative.")
        self.replay_buffer = SuccessReplay(
            seed=self.cfg.actor.seed + self._rank,
            archive_path=str(Path(bc.data_path) / f"rank_{self._rank}"),
            max_success_chunks=bc.get("max_success_chunks"),
        )
        self.dvac = None
        self.dvac_metrics = {}
        self.dvac_batch_metrics = {}
        dvac_cfg = bc.get("dvac", {})
        self.dvac_normalization = dvac_cfg.get("normalization", "recent")
        if self.dvac_normalization not in ("recent", "two_level_batch"):
            raise ValueError("Unknown online BC DVAC normalization.")
        if self.dvac_normalization == "two_level_batch":
            if not dvac_cfg.get("enabled", False) or self._world_size != 1:
                raise ValueError("Two-level BC DVAC requires enabled DVAC and one actor rank.")
            if any(key in dvac_cfg for key in ("window", "mapping", "alpha", "z_clip", "weight_min", "weight_max", "std_floor")):
                raise ValueError("Two-level BC DVAC cannot include legacy calibration settings.")
            self.dvac_new_settings = {
                "alpha_local": float(dvac_cfg.get("alpha_local", 1.0)),
                "alpha_chunk": float(dvac_cfg.get("alpha_chunk", 1.0)),
                "variance_eps": float(dvac_cfg.get("log_eps", 1e-12)),
                "range_eps": float(dvac_cfg.get("range_eps", 1e-6)),
            }
            from rlinf.algorithms.online_bc_dvac_two_level import compute_two_level_bc_weights

            compute_two_level_bc_weights(torch.ones(1, 1), torch.ones(1, 1, 1), **self.dvac_new_settings)
            self.dvac_debug_batches = int(dvac_cfg.get("debug_batches", 0))
            if not 0 <= self.dvac_debug_batches <= 4:
                raise ValueError("DVAC debug_batches must be between 0 and 4.")
        elif dvac_cfg.get("enabled", False):
            from rlinf.algorithms.online_bc_dvac import OnlineBCDvac

            self.dvac = OnlineBCDvac(
                **{
                    key: dvac_cfg[key]
                    for key in (
                        "window", "alpha", "z_clip", "log_eps", "std_floor",
                        "mapping", "weight_min", "weight_max",
                    )
                    if key in dvac_cfg
                }
            )
        if self.demo_weight:
            self._build_demo_loader()

    def _build_demo_loader(self):
        import openpi.training.data_loader as data

        from rlinf.data.storage.lerobot import resolve_lerobot_repo_id
        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

        path = self.cfg.algorithm.online_bc.demo_data_path
        if not path or not Path(path).is_dir():
            raise ValueError(
                "demo_weight>0 requires an existing local LeRobot demo_data_path."
            )
        config = get_openpi_config(
            self.cfg.actor.model.openpi.config_name,
            model_path=self.cfg.actor.model.model_path,
            repo_id=resolve_lerobot_repo_id(path),
            data_kwargs=getattr(self.cfg.actor.model, "openpi_data", None),
        )
        config = replace(
            config,
            batch_size=self.cfg.actor.micro_batch_size * self._world_size,
            num_workers=0,
        )
        self.demo_loader = data.create_data_loader(
            config, framework="pytorch", shuffle=True
        )
        self.demo_iterator = iter(self.demo_loader)

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        new_episodes = []
        moments = torch.zeros(3, dtype=torch.float64)
        for _ in range(compute_split_num(send_num, recv_num)):
            # Every env stage sends its exact split count, including empty lists.
            packet = await input_channel.get(async_op=True).async_wait()
            if getattr(self, "dvac_normalization", "recent") == "two_level_batch":
                episodes = packet["episodes"]
                for episode in episodes:
                    for record in episode:
                        self.validate_dvac_record(record)
                        if "action_weights" in record:
                            raise ValueError("New BC DVAC replay must store V, not frozen weights.")
                self.replay_buffer.add_episodes(episodes)
            elif self.dvac is None:
                self.replay_buffer.add_episodes(packet)
            else:
                new_episodes.extend(packet["episodes"])
                moments += packet["dvac_moments"].cpu()
        if self.dvac is not None:
            moments = moments.to(self.device)
            torch.distributed.all_reduce(moments)
            self.dvac_metrics = self.dvac.annotate(new_episodes, moments.cpu())
            self.replay_buffer.add_episodes(new_episodes)
            self.log_info(f"Online BC DVAC: {self.dvac_metrics}")

    @staticmethod
    def validate_dvac_record(record):
        """Reject malformed online signal before admission or model restoration."""
        if "dvac_v" not in record:
            raise ValueError("Two-level BC DVAC online record is missing dvac_v.")
        variance, mask = record["dvac_v"], record["action_valid_mask"]
        if mask.ndim != 2 or variance.shape != mask.shape[:1]:
            raise ValueError("BC DVAC V must align with the command mask [H,D].")
        if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("BC command mask must be binary and finite.")
        valid = mask.bool().any(-1)
        if not valid.any() or not torch.isfinite(variance[valid]).all() or (variance[valid] < 0).any():
            raise ValueError("BC DVAC requires valid nonnegative V and nonempty query targets.")

    def prepare_replay_batch(self, batch):
        if getattr(self, "dvac_normalization", "recent") != "two_level_batch":
            return batch
        from rlinf.algorithms.online_bc_dvac_two_level import compute_two_level_bc_weights

        inputs = batch["forward_inputs"]
        if "dvac_v" not in inputs:
            raise ValueError("Two-level BC DVAC sampled batch is missing dvac_v.")
        mask = inputs["action_valid_mask"]
        if mask.ndim != 3 or mask.shape[0] != self.cfg.actor.global_batch_size:
            raise ValueError("Two-level BC DVAC must see the complete optimizer batch.")
        if (mask.sum(dim=(1, 2)) == 0).any():
            raise ValueError("BC FM loss requires nonempty targets in every query.")
        weights, diagnostics = compute_two_level_bc_weights(
            inputs["dvac_v"], mask, **self.dvac_new_settings
        )
        # Replace any previous mapping, without mutating the replay records.
        inputs["action_weights"] = weights
        self.dvac_batch_metrics = {f"dvac_new/{key}": value for key, value in diagnostics.items()}
        if self.update_step < self.dvac_debug_batches:
            target = Path(self.cfg.runner.logger.log_path) / "dvac_new_debug"
            target.mkdir(parents=True, exist_ok=True)
            with (target / f"update_{self.update_step:06d}.pt").open("xb") as stream:
                torch.save({
                    "variance": inputs["dvac_v"].cpu(), "mask": mask.cpu(),
                    "weights": weights.cpu(), "settings": self.dvac_new_settings,
                    "update_step": self.update_step, "diagnostics": diagnostics,
                }, stream)
        return batch

    def update_buffer_one_epoch(self):
        metrics = super().update_buffer_one_epoch()
        return metrics | getattr(self, "dvac_batch_metrics", {})

    def dvac_new_state(self):
        return {"normalization": "two_level_batch", "version": 1, "settings": self.dvac_new_settings}

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        prepared = self.model.prepare_dagger_sft_batch(batch)
        online_loss = self.model(
            forward_type=ForwardType.SFT,
            data=prepared,
            use_action_chunk_loss=True,
            action_valid_mask=batch["action_valid_mask"],
            action_weights=batch.get("action_weights"),
        )
        if not self.demo_weight:
            return online_loss
        try:
            demo = next(self.demo_iterator)
        except StopIteration:
            self.demo_iterator = iter(self.demo_loader)
            demo = next(self.demo_iterator)
        demo_loss = self.model(
            forward_type=ForwardType.SFT, data=demo, use_action_chunk_loss=True
        )
        # Explicit loss mixture; this is not a fraction of interaction episodes.
        return (online_loss + self.demo_weight * demo_loss) / (1 + self.demo_weight)

    @Worker.timer("run_training")
    def run_training(self):
        ready = torch.tensor(
            [
                int(
                    self.replay_buffer.is_ready(
                        self.cfg.algorithm.replay_buffer.min_buffer_size
                    )
                )
            ],
            device=self.device,
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
        if not ready.item():
            return {"bc/skipped_empty_rank": 1.0, **self.dvac_metrics}
        metrics = super().run_training()
        return {
            k.replace("dagger/", "bc/"): v for k, v in metrics.items()
        } | self.dvac_metrics

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer],
            lr_schedulers=[self.lr_scheduler],
            save_path=save_base_path,
            checkpoint_format=self.checkpoint_format,
        )
        target = Path(save_base_path) / "online_bc" / f"rank_{self._rank}"
        self.replay_buffer.save_checkpoint(target)
        torch.save({"update_step": self.update_step}, target / "learner.pt")
        if self.dvac is not None:
            torch.save(self.dvac.state_dict(), target / "dvac.pt")
        if getattr(self, "dvac_normalization", "recent") == "two_level_batch":
            torch.save(self.dvac_new_state(), target / "dvac_new.pt")

    def load_checkpoint(self, load_base_path):
        # FSDP restore requires parameters and optimizer on their active device.
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False
        target = Path(load_base_path) / "online_bc" / f"rank_{self._rank}"
        if getattr(self, "dvac_normalization", "recent") == "two_level_batch":
            state_path = target / "dvac_new.pt"
            if not state_path.is_file() or (target / "dvac.pt").exists():
                raise ValueError("BC DVAC new requires its own checkpoint; legacy resume is unsupported.")
            if torch.load(state_path, weights_only=True) != self.dvac_new_state():
                raise ValueError("BC DVAC new checkpoint settings/version mismatch.")
            self.replay_buffer.load_checkpoint(target)
            for record in self.replay_buffer.records:
                self.validate_dvac_record(record)
                if "action_weights" in record:
                    raise ValueError("BC DVAC new checkpoint contains frozen replay weights.")
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer],
            lr_schedulers=[self.lr_scheduler],
            load_path=load_base_path,
            checkpoint_format=self.checkpoint_format,
        )
        if getattr(self, "dvac_normalization", "recent") != "two_level_batch":
            self.replay_buffer.load_checkpoint(target)
        self.update_step = torch.load(target / "learner.pt", weights_only=True)[
            "update_step"
        ]
        if self.dvac is not None:
            self.dvac.load_state_dict(torch.load(target / "dvac.pt", weights_only=True))
        audit_online_bc_resume(self, load_base_path)
