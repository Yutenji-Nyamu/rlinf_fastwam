# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Online success BC using RLinf's existing supervised FSDP update machinery."""

from dataclasses import replace
from pathlib import Path

import torch

from rlinf.data.online_bc import SuccessReplay
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.workers.actor.fsdp_dagger_policy_worker import EmbodiedDAGGERFSDPPolicy
from rlinf.algorithms.online_bc_rabc import RABCConfig, RABCStats, raw_weights, normalized_batch_weights
from rlinf.utils.rynnvalue_client import RynnValueClient


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
        self.rabc_enabled = bool(bc.get("rabc", {}).get("enabled", False))
        self.rabc_metrics = {}
        if self.rabc_enabled:
            if self._world_size != 1 or self.demo_weight or self.enable_drq:
                raise ValueError("Initial RA-BC requires one actor rank, no demo mixture and no DrQ.")
            if bc.get("dvac", {}).get("enabled", False) or bc.get("attena", {}).get("enabled", False):
                raise ValueError("RA-BC is an independent clean-BC method.")
            rc = bc.rabc
            self.rabc_config = RABCConfig(float(rc.kappa_seconds), float(rc.get("epsilon_signal", 1e-6)),
                                          float(rc.get("epsilon_weight", 1e-6)))
            self.rabc_stats = RABCStats()
            self.rabc_seen = set()
            self.rabc_client = RynnValueClient(rc.endpoint, Path(bc.data_path) / "value_cache",
                                              rc.model_revision, rc.robot_description, rc.camera_description,
                                              rc.get("timeout_seconds", 900))
            self.rabc_debug_remaining = int(rc.get("debug_batches", 0))
            self.rabc_debug_index = 0
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
        for _ in range(compute_split_num(send_num, recv_num)):
            # Every env stage sends its exact split count, including empty lists.
            episodes = await input_channel.get(async_op=True).async_wait()
            if self.rabc_enabled:
                episodes = self.score_admissions(episodes)
            self.replay_buffer.add_episodes(episodes)

    def score_admissions(self, episodes):
        admitted = []
        for episode in episodes:
            limit = self.replay_buffer.max_success_chunks
            if limit is not None and len(episode) > limit:
                # Pass through for the unchanged replay admission counter.
                admitted.append(episode)
                continue
            key = bytes(episode[0]["rabc_episode_key"].tolist()).hex()
            if key in self.rabc_seen:
                continue
            scored, deltas, key = self.rabc_client.score_episode(episode)
            self.rabc_stats.update(deltas)
            self.rabc_seen.add(key)
            admitted.append(scored)
        if self.rabc_stats.count:
            self.rabc_metrics.update({"rabc/unique_queries": self.rabc_stats.count,
                                     "rabc/progress_mean_seconds": self.rabc_stats.raw_mean,
                                     "rabc/progress_std_seconds": self.rabc_stats.population_std})
        return admitted

    def rabc_identity(self):
        return {"version": 1, "scorer": self.rabc_client.identity,
                "kappa_seconds": self.rabc_config.kappa_seconds,
                "epsilon_signal": self.rabc_config.epsilon_signal,
                "epsilon_weight": self.rabc_config.epsilon_weight,
                "stats_scope": "all_admitted_unique_queries", "loss_scope": "complete_optimizer_batch",
                "max_success_chunks": self.replay_buffer.max_success_chunks}

    def validate_rabc_record(self, record):
        if bytes(record["rabc_identity"].tolist()).hex() != self.rabc_client.identity_hash:
            raise ValueError("RA-BC replay scorer identity mismatch.")
        delta = record["rabc_delta_seconds"]
        if delta.ndim != 0 or not torch.isfinite(delta):
            raise ValueError("RA-BC replay has invalid scalar progress.")
        before, after = record["rabc_value_before"], record["rabc_value_after"]
        if any(v.ndim != 0 or not torch.isfinite(v) or v < 0 or v > 512 for v in (before, after)):
            raise ValueError("RA-BC replay has invalid remaining seconds.")
        if not torch.allclose(before - after, delta, atol=1e-6, rtol=1e-6):
            raise ValueError("RA-BC replay progress and boundary values differ.")
        if "sample_weights" in record:
            raise ValueError("RA-BC replay must store raw progress, not frozen training weights.")

    def prepare_replay_batch(self, batch):
        if not self.rabc_enabled:
            return batch
        data = batch["forward_inputs"]
        expected = torch.tensor(list(bytes.fromhex(self.rabc_client.identity_hash)), dtype=torch.uint8)
        if not torch.equal(data["rabc_identity"], expected.expand_as(data["rabc_identity"])):
            raise ValueError("Sampled RA-BC scorer identity mismatch.")
        deltas = data["rabc_delta_seconds"]
        if deltas.numel() != self.cfg.actor.global_batch_size:
            raise ValueError("RA-BC must normalize the complete optimizer batch.")
        raw = raw_weights(deltas, self.rabc_stats, self.rabc_config)
        weights, diagnostics = normalized_batch_weights(raw, self.rabc_config)
        self.rabc_metrics.update({"rabc/" + key: float(value) for key, value in diagnostics.items()})
        if self.rabc_debug_remaining > 0:
            target = Path(self.cfg.algorithm.online_bc.data_path) / "weight_debug"
            target.mkdir(parents=True, exist_ok=True)
            with (target / f"batch_{self.rabc_debug_index:06d}.pt").open("xb") as stream:
                torch.save({"delta_seconds": deltas, "raw_weights": raw, "sample_weights": weights,
                            "stats": self.rabc_stats.state_dict(), "identity": self.rabc_identity(),
                            "diagnostics": diagnostics}, stream)
            self.rabc_debug_remaining -= 1
            self.rabc_debug_index += 1
        if diagnostics["skip_update"]:
            return None
        data["sample_weights"] = weights.to(dtype=torch.float32)
        return batch

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        prepared = self.model.prepare_dagger_sft_batch(batch)
        online_loss = self.model(
            forward_type=ForwardType.SFT,
            data=prepared,
            use_action_chunk_loss=True,
            action_valid_mask=batch["action_valid_mask"],
            sample_weights=batch.get("sample_weights"),
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
            return {"bc/skipped_empty_rank": 1.0}
        metrics = super().run_training()
        return {k.replace("dagger/", "bc/"): v for k, v in metrics.items()} | self.rabc_metrics

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
        if self.rabc_enabled:
            torch.save({"identity": self.rabc_identity(), "stats": self.rabc_stats.state_dict(),
                        "seen": sorted(self.rabc_seen)}, target / "rabc.pt")

    def load_checkpoint(self, load_base_path):
        target = Path(load_base_path) / "online_bc" / f"rank_{self._rank}"
        if self.rabc_enabled:
            state = torch.load(target / "rabc.pt", weights_only=True)
            if state["identity"] != self.rabc_identity():
                raise ValueError("RA-BC resume scorer/method identity mismatch.")
            stats = RABCStats()
            stats.load_state_dict(state["stats"])
            self.replay_buffer.load_checkpoint(target)
            for record in self.replay_buffer.records:
                self.validate_rabc_record(record)
            if stats.count != len(self.replay_buffer.records):
                raise ValueError("RA-BC statistics and replay unique-query counts differ.")
            recomputed = RABCStats()
            if self.replay_buffer.records:
                recomputed.update(torch.stack([record["rabc_delta_seconds"] for record in self.replay_buffer.records]))
            if not torch.allclose(torch.tensor([stats.raw_mean, stats.m2], dtype=torch.float64),
                                  torch.tensor([recomputed.raw_mean, recomputed.m2], dtype=torch.float64),
                                  atol=1e-10, rtol=1e-10):
                raise ValueError("RA-BC checkpoint moments do not match replay progress.")
            actual_keys = {bytes(record["rabc_episode_key"].tolist()).hex() for record in self.replay_buffer.records}
            if actual_keys != set(state["seen"]):
                raise ValueError("RA-BC checkpoint episode identities differ.")
            self.rabc_stats, self.rabc_seen = stats, actual_keys
            self.rabc_debug_remaining = 0  # Existing debug receipts remain immutable on resume.
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer],
            lr_schedulers=[self.lr_scheduler],
            load_path=load_base_path,
            checkpoint_format=self.checkpoint_format,
        )
        target = Path(load_base_path) / "online_bc" / f"rank_{self._rank}"
        if not self.rabc_enabled:
            self.replay_buffer.load_checkpoint(target)
        self.update_step = torch.load(target / "learner.pt", weights_only=True)[
            "update_step"
        ]
