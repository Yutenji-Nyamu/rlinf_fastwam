# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-round success-BC warmup followed by online pixel-IQL weighted FM."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import torch
from omegaconf import OmegaConf

from rlinf.data.online_iql import IQLRynnValueClient, TransitionReplay
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.online_iql_critic import PixelIQLLearner, stack_pixels
from rlinf.scheduler import Channel, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.workers.actor.fsdp_online_bc_policy_worker import EmbodiedOnlineBCFSDPPolicy


FORMAT_VERSION = 1


def phase_for_round(completed_rounds: int, warmup_rounds: int) -> str:
    if type(completed_rounds) is not int or completed_rounds < 0:
        raise ValueError("completed_rounds must be a nonnegative integer.")
    if type(warmup_rounds) is not int or warmup_rounds < 0:
        raise ValueError("warmup_rounds must be a nonnegative integer.")
    return "success_bc" if completed_rounds < warmup_rounds else "iql"


def _plain(config):
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return dict(config)


def _identity_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _finite_scalar(value, name: str) -> float:
    scalar = float(torch.as_tensor(value).detach().cpu())
    if not math.isfinite(scalar):
        raise FloatingPointError(f"Non-finite {name}: {scalar}")
    return scalar


class EmbodiedOnlineIQLFSDPPolicy(EmbodiedOnlineBCFSDPPolicy):
    """Single-rank IQL controller; the VLA's supervised optimizer is unchanged."""

    def setup_dagger_components(self):
        super().setup_dagger_components()
        config = self.cfg.algorithm.online_iql
        if not bool(config.get("enabled", False)):
            raise ValueError("The IQL worker requires online_iql.enabled=true.")
        if self._world_size != 1 or self.demo_weight or self.enable_drq:
            raise ValueError("Online IQL supports one actor rank, no demos and no actor DRQ.")
        if self.enable_online_lerobot or self.cfg.runner.get("use_training_pipeline", False):
            raise ValueError("Online IQL requires synchronous replay-based rounds.")
        if self.cfg.actor.get("compile_model", False):
            raise ValueError("The initial online IQL implementation does not compile the actor.")
        self.warmup_rounds = config.get("warmup_rounds", 10)
        phase_for_round(0, self.warmup_rounds)
        self.critic_batch_size = int(config.get("critic_batch_size", 64))
        self.critic_updates_per_slot = int(config.get("critic_updates_per_slot", 1))
        self.expected_episodes_per_round = int(config.get("episodes_per_round", 4))
        if self.critic_batch_size < 1 or self.critic_updates_per_slot != 1:
            raise ValueError("First-version IQL requires a positive critic batch and K=1.")
        if float(config.get("diagnostic_holdout_fraction", 0.0)) != 0.0:
            raise ValueError("Fixed two-stage IQL uses the full transition replay (holdout=0).")
        if int(self.cfg.actor.global_batch_size) != 1024 or int(self.cfg.actor.micro_batch_size) != 32:
            raise ValueError("Online IQL retains the clean actor batch1024/micro32 contract.")
        self._last_completed_round = 0
        self._completed_before = None
        self._round_open = False
        self._round_received_episodes = 0
        self._phase = "success_bc"
        self.actor_skip_count = 0
        self.critic_skip_count = 0
        self.iql_slots = 0
        self._micro_unweighted = []
        self._last_weight_metrics = {}

        # UUID uses system entropy, never either training sampler or torch RNG.
        self.attempt_id = uuid.uuid4().hex
        self.replay_buffer.archive_path = (
            self.replay_buffer.archive_path / f"attempt_{self.attempt_id}"
        )
        scorer = config.scorer
        self.iql_scorer = IQLRynnValueClient(
            endpoint=str(scorer.endpoint),
            cache_path=str(scorer.cache_path),
            expected_revision=str(scorer.expected_revision),
            robot_description=str(scorer.robot_description),
            camera_description=str(scorer.camera_description),
            timeout_seconds=float(scorer.get("timeout_seconds", 900)),
            gamma=float(config.get("gamma", 0.99)),
            potential_scale=float(config.get("potential_scale", 0.1)),
        )
        critic_config = _plain(config.critic)
        if float(critic_config.get("discount", 0.99)) != float(config.get("gamma", 0.99)):
            raise ValueError("Critic and reward must use the same macro-query discount.")
        critic_config.setdefault("seed", int(self.cfg.actor.seed) + 10001)
        self.iql = PixelIQLLearner(critic_config, self.device)
        source_root = Path(__file__).resolve().parents[2]
        source_files = (
            "workers/actor/fsdp_online_iql_policy_worker.py",
            "workers/actor/fsdp_dagger_policy_worker.py",
            "workers/env/env_worker.py",
            "models/embodiment/openpi/openpi_action_model.py",
            "models/embodiment/online_iql_critic.py",
            "algorithms/online_iql.py",
            "data/online_iql.py",
            "data/online_bc.py",
        )
        source_hashes = {
            relative: hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
            for relative in source_files
        }
        scorer_identity = self.iql_scorer.identity
        self.iql_contract = {
            "format_version": FORMAT_VERSION,
            "method": "pi05-online-iql-fixed-two-stage-v1",
            "warmup_rounds": self.warmup_rounds,
            "actor_updates_per_round": int(self.cfg.algorithm.update_epoch),
            "critic_updates_per_slot": self.critic_updates_per_slot,
            "critic_batch_size": self.critic_batch_size,
            "episodes_per_round": self.expected_episodes_per_round,
            "critic": critic_config,
            "gamma": float(config.get("gamma", 0.99)),
            "potential_scale": float(config.get("potential_scale", 0.1)),
            "terminal_contract": "finite-200-command-success-or-budget-absorbing-v1",
            "action_contract": "full-proposal-50x14-clean-mask-nullable-physical-length-v1",
            "scorer": scorer_identity,
            "actor_seed": int(self.cfg.actor.seed),
            "actor_batch": int(self.cfg.actor.global_batch_size),
            "actor_micro_batch": int(self.cfg.actor.micro_batch_size),
            "actor_optimizer": _plain(self.cfg.actor.optim),
            "actor_model_path": str(self.cfg.actor.model.model_path),
            "success_max_chunks": self.replay_buffer.max_success_chunks,
            "sources": source_hashes,
        }
        self.iql_contract_hash = _identity_hash(self.iql_contract)
        self.transition_replay = TransitionReplay(
            seed=int(self.cfg.actor.seed) + 20001,
            actor_seed=int(self.cfg.actor.seed) + 30001,
            archive_path=str(
                Path(config.data_path) / f"rank_{self._rank}" / f"attempt_{self.attempt_id}"
            ),
            contract=self.iql_contract,
        )
        self.log_info(
            f"Online IQL initialized: warmup_rounds={self.warmup_rounds}, "
            f"critic_batch={self.critic_batch_size}, K=1, identity={self.iql_contract_hash}"
        )

    def set_global_step(self, global_step: int) -> None:
        phase = phase_for_round(global_step, self.warmup_rounds)
        if self._round_open or global_step != self._last_completed_round:
            raise ValueError(
                f"Round boundary mismatch: runner={global_step}, "
                f"completed={self._last_completed_round}, open={self._round_open}"
            )
        super().set_global_step(global_step)
        self._completed_before = global_step
        self._phase = phase
        self._round_open = True
        self._round_received_episodes = 0

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel):
        if not self._round_open:
            raise RuntimeError("IQL rollout receipt requires a declared collection round.")
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        for _ in range(compute_split_num(send_num, recv_num)):
            payload = await input_channel.get(async_op=True).async_wait()
            if not isinstance(payload, dict) or not {
                "episodes", "transition_episodes"
            }.issubset(payload):
                raise ValueError("IQL requires success and complete-transition payloads.")
            scored = []
            for episode in payload["transition_episodes"]:
                if not episode or any(
                    int(row["iql_collection_round"]) != self._completed_before + 1
                    for row in episode
                ):
                    raise ValueError("IQL received an empty or wrong-round transition episode.")
                records, _values, _key = self.iql_scorer.score_episode(episode)
                scored.append(records)
            # Do not admit actor successes when scoring this payload failed.
            self.transition_replay.add_episodes(scored)
            self.replay_buffer.add_episodes(payload["episodes"])
            self._round_received_episodes += len(scored)

    def _normalized_critic_actions(self, data):
        # This is the original deterministic OpenPI action transform, not raw
        # physical commands and not a second independently fitted normalizer.
        with torch.no_grad():
            transform_data = self._actor_fields(data)
            transform_data["action"] = data["action"].clone()
            prepared = self.model.prepare_dagger_sft_batch(transform_data)
            actions = prepared["actions"][:, :50, :14].detach().to(torch.float32).contiguous()
        if tuple(actions.shape[1:]) != (50, 14) or not torch.isfinite(actions).all():
            raise ValueError("IQL requires finite normalized full50x14 proposals.")
        return actions

    @staticmethod
    def _actor_fields(data):
        keys = {
            "action", "action_valid_mask", "model_action", "tokenized_prompt",
            "tokenized_prompt_mask", "sample_weights",
        }
        return {key: value for key, value in data.items() if key.startswith("observation/") or key in keys}

    def _critic_inputs(self, batch):
        data = batch["forward_inputs"]
        pixels = stack_pixels(data["observation/image"], data["observation/wrist_image"])
        next_pixels = stack_pixels(data["iql_next_image"], data["iql_next_wrist_image"])
        actions = self._normalized_critic_actions(data)
        return pixels, actions, next_pixels, data["iql_reward"], data["iql_bootstrap_mask"]

    def _update_critic_once(self):
        if len(self.transition_replay) == 0:
            self.critic_skip_count += 1
            return {"iql/critic_skipped_empty": 1.0}
        batch = self.transition_replay.sample_critic(self.critic_batch_size)
        inputs = self._critic_inputs(batch)
        metrics = self.iql.update(*inputs)
        result = {"iql/" + key: _finite_scalar(value, key) for key, value in metrics.items()}
        result["iql/critic_batch_actual"] = float(inputs[1].shape[0])
        return result

    def _weight_actor_batch(self, global_batch):
        data = global_batch["forward_inputs"]
        count = data["action"].shape[0]
        if count != int(self.cfg.actor.global_batch_size):
            raise ValueError("IQL weights must cover the complete actor optimizer batch.")
        weights, advantages = [], []
        # CPU replay may contain large images; never materialize the entire
        # 1024-image actor observation on GPU just to compute critic weights.
        for start in range(0, count, self.critic_batch_size):
            part = {key: value[start:start + self.critic_batch_size] for key, value in data.items()}
            pixels = stack_pixels(part["observation/image"], part["observation/wrist_image"])
            actions = self._normalized_critic_actions(part)
            out = self.iql.advantages(pixels, actions, batch_size=self.critic_batch_size)
            weights.append(out["weights"].detach().to(device="cpu", dtype=torch.float32))
            advantages.append(out["advantage"].detach().to(device="cpu", dtype=torch.float32))
        weight = torch.cat(weights)
        advantage = torch.cat(advantages)
        if weight.shape != (count,) or not torch.isfinite(weight).all() or (weight < 0).any():
            raise FloatingPointError("Invalid IQL actor weights.")
        if not torch.isfinite(advantage).all():
            raise FloatingPointError("Non-finite IQL actor advantage.")
        data["sample_weights"] = weight
        total = weight.double().sum()
        square = weight.double().square().sum()
        self._last_weight_metrics = {
            "iql/weight_mean": float(weight.mean()),
            "iql/weight_min": float(weight.min()),
            "iql/weight_max": float(weight.max()),
            "iql/weight_p50": float(weight.quantile(0.5)),
            "iql/weight_p95": float(weight.quantile(0.95)),
            "iql/weight_cap_fraction": float((weight >= 100).float().mean()),
            "iql/weight_zero_fraction": float((weight == 0).float().mean()),
            "iql/weight_ess": float(total.square() / square) if square > 0 else 0.0,
            "iql/advantage_mean": float(advantage.mean()),
            "iql/advantage_positive_fraction": float((advantage > 0).float().mean()),
        }
        # The generic BC loop stages all microbatches on GPU. Do not carry
        # critic next-images, scoring metadata or replay IDs into that loop.
        return {"forward_inputs": self._actor_fields(data)}

    def forward_actor(self, batch):
        if self._phase == "success_bc":
            if "sample_weights" in batch:
                raise ValueError("Warmup BC cannot receive advantage weights.")
            loss = super().forward_actor(batch)
            self._micro_unweighted.append(_finite_scalar(loss, "warmup FM loss"))
            return loss
        if "sample_weights" not in batch:
            raise ValueError("The IQL phase requires frozen full-batch sample weights.")
        prepared = self.model.prepare_dagger_sft_batch(batch)
        out = self.model(
            forward_type=ForwardType.SFT,
            data=prepared,
            use_action_chunk_loss=True,
            action_valid_mask=batch["action_valid_mask"],
            sample_weights=batch["sample_weights"],
            return_sft_metrics=True,
        )
        _finite_scalar(out["loss"], "weighted FM loss")
        self._micro_unweighted.append(_finite_scalar(out["unweighted_loss"], "ordinary FM loss"))
        return out["loss"]

    def validate_replay_gradients(self, actor_grad_norm):
        _finite_scalar(actor_grad_norm, "actor gradient norm")

    def _run_slot(self):
        metrics = self._update_critic_once()
        self.iql_slots += 1
        if self._phase == "success_bc":
            if len(self.replay_buffer) == 0:
                self.actor_skip_count += 1
                return {**metrics, "iql/actor_skipped_empty_success": 1.0}
            batch = self.replay_buffer.sample(int(self.cfg.actor.global_batch_size))
        else:
            if len(self.transition_replay) == 0:
                raise RuntimeError("The IQL phase has no complete transitions after collection.")
            batch = self._weight_actor_batch(
                self.transition_replay.sample_actor(int(self.cfg.actor.global_batch_size))
            )
            metrics.update(self._last_weight_metrics)
        self._micro_unweighted = []
        actor_metrics = self.update_sampled_buffer_batch(batch)
        self.update_step += 1
        metrics.update({key.replace("dagger/", "bc/"): value for key, value in actor_metrics.items()})
        if self._micro_unweighted:
            metrics["iql/unweighted_fm_loss"] = sum(self._micro_unweighted) / len(self._micro_unweighted)
        return metrics

    def _ensure_actor_loaded(self):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

    @Worker.timer("run_training")
    def run_training(self):
        if not self._round_open:
            raise RuntimeError("IQL training must run once per declared collection round.")
        if self._round_received_episodes != self.expected_episodes_per_round:
            raise ValueError(
                f"IQL round received {self._round_received_episodes} complete attempts; "
                f"expected {self.expected_episodes_per_round}."
            )
        self._ensure_actor_loaded()
        self.gradient_accumulation = int(self.cfg.actor.global_batch_size) // int(self.cfg.actor.micro_batch_size)
        self.model.train()
        values = {}
        for _slot in range(int(self.cfg.algorithm.update_epoch)):
            for key, value in self._run_slot().items():
                values.setdefault(key, []).append(_finite_scalar(value, key))
        result = {key: sum(items) / len(items) for key, items in values.items()}
        self._last_completed_round = self._completed_before + 1
        self._round_open = False
        result.update({
            "iql/phase": float(self._phase == "iql"),
            "iql/completed_rounds": float(self._last_completed_round),
            "iql/actor_updates": float(self.update_step),
            "iql/critic_updates": float(self.iql.update_count),
            "iql/slots": float(self.iql_slots),
            "iql/collected_episodes": float(self._round_received_episodes),
            "iql/actor_skips_total": float(self.actor_skip_count),
            "iql/critic_skips_total": float(self.critic_skip_count),
        })
        for name, pool in (("success", self.replay_buffer), ("transition", self.transition_replay)):
            result.update({f"iql/{name}_{key}": float(value) for key, value in pool.get_stats().items()})
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return result

    def save_checkpoint(self, save_base_path, step):
        if self._round_open or step != self._last_completed_round:
            raise ValueError("Only a completely finished IQL round can be saved.")
        self._ensure_actor_loaded()
        target = Path(save_base_path) / "online_iql" / f"rank_{self._rank}"
        target.mkdir(parents=True, exist_ok=True)
        marker = target / "COMPLETED.json"
        if marker.exists():
            raise FileExistsError(f"Refusing to overwrite a committed checkpoint: {marker}")
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer],
            lr_schedulers=[self.lr_scheduler],
            save_path=save_base_path,
            checkpoint_format=self.checkpoint_format,
        )
        self.replay_buffer.save_checkpoint(target / "success")
        self.transition_replay.save_checkpoint(target / "transition")
        state = {
            "format_version": FORMAT_VERSION,
            "contract": self.iql_contract,
            "contract_hash": self.iql_contract_hash,
            "completed_rounds": int(step),
            "next_phase": phase_for_round(int(step), self.warmup_rounds),
            "actor_update_step": self.update_step,
            "actor_skip_count": self.actor_skip_count,
            "critic_skip_count": self.critic_skip_count,
            "slots": self.iql_slots,
            "critic": self.iql.state_dict(),
            "grad_scaler": self.grad_scaler.state_dict(),
            "actor_torch_rng": torch.get_rng_state(),
            "actor_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        torch.save(state, target / "learner.pt")
        files = {
            str(path.relative_to(Path(save_base_path))): path.stat().st_size
            for path in Path(save_base_path).rglob("*") if path.is_file()
        }
        receipt = {
            "format_version": FORMAT_VERSION,
            "completed_rounds": int(step),
            "contract_hash": self.iql_contract_hash,
            "files": files,
        }
        # Written last: a partial actor/replay/critic save is never resumable.
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())

    def load_checkpoint(self, load_base_path):
        base = Path(load_base_path)
        target = base / "online_iql" / f"rank_{self._rank}"
        marker = target / "COMPLETED.json"
        if not marker.is_file():
            raise ValueError("IQL checkpoint has no complete-round commit marker.")
        receipt = json.loads(marker.read_text(encoding="utf-8"))
        if receipt.get("format_version") != FORMAT_VERSION or receipt.get("contract_hash") != self.iql_contract_hash:
            raise ValueError("IQL checkpoint method/source contract mismatch.")
        for relative, size in receipt.get("files", {}).items():
            path = (base / relative).resolve()
            if not path.is_relative_to(base.resolve()) or not path.is_file() or path.stat().st_size != size:
                raise ValueError(f"Incomplete or changed IQL checkpoint member: {relative}")
        state = torch.load(target / "learner.pt", map_location="cpu", weights_only=True)
        if state.get("contract") != self.iql_contract or state.get("contract_hash") != self.iql_contract_hash:
            raise ValueError("IQL learner contract mismatch.")
        completed = state["completed_rounds"]
        expected_phase = phase_for_round(completed, self.warmup_rounds)
        if completed != receipt["completed_rounds"] or state["next_phase"] != expected_phase:
            raise ValueError("IQL checkpoint round/phase mismatch.")
        if base.parent.name.startswith("global_step_"):
            if int(base.parent.name.removeprefix("global_step_")) != completed:
                raise ValueError("IQL checkpoint directory and completed rounds differ.")
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer],
            lr_schedulers=[self.lr_scheduler],
            load_path=load_base_path,
            checkpoint_format=self.checkpoint_format,
        )
        self.replay_buffer.load_checkpoint(target / "success")
        self.transition_replay.load_checkpoint(target / "transition")
        self.iql.load_state_dict(state["critic"])
        self.grad_scaler.load_state_dict(state["grad_scaler"])
        self.update_step = int(state["actor_update_step"])
        self.actor_skip_count = int(state["actor_skip_count"])
        self.critic_skip_count = int(state["critic_skip_count"])
        self.iql_slots = int(state["slots"])
        self._last_completed_round = completed
        self._completed_before = None
        self._phase = expected_phase
        self._round_open = False
        self._round_received_episodes = 0
        torch.set_rng_state(state["actor_torch_rng"])
        if state["actor_cuda_rng"]:
            torch.cuda.set_rng_state_all(state["actor_cuda_rng"])
