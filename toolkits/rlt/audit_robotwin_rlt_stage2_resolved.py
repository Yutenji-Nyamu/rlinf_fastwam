#!/usr/bin/env python3
"""Audit bound RoboTwin RLT Stage 2 formal/fresh/resume configs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf


def require_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}.")


def load_config(path: Path) -> dict:
    config = OmegaConf.load(path)
    return OmegaConf.to_container(config, resolve=True)


def select(config: dict, dotted_path: str):
    value = config
    for key in dotted_path.split("."):
        value = value[key]
    return value


def reject_unresolved(label: str, config: dict) -> None:
    rendered = json.dumps(config, sort_keys=True)
    if "UNRESOLVED" in rendered or "PLACEHOLDER" in rendered or "/path/to/" in rendered:
        raise ValueError(f"{label} config still contains an unresolved value.")


def audit_common(config: dict, args: argparse.Namespace, label: str) -> None:
    reject_unresolved(label, config)
    expected = {
        "runner.weight_sync_interval": 1,
        "algorithm.loss_type": "rlt_ac",
        "algorithm.agg_q": "min",
        "algorithm.actor_agg_q": "q1",
        "algorithm.bootstrap_type": "standard",
        "algorithm.gamma": 0.99,
        "algorithm.tau": 0.005,
        "algorithm.update_epoch": 5,
        "algorithm.critic_actor_ratio": 2,
        "algorithm.reference_dropout_prob": 0.5,
        "algorithm.entropy_tuning.alpha_type": "fixed_alpha",
        "algorithm.entropy_tuning.initial_alpha": 0.0,
        "algorithm.rlt_route.type": "full_task",
        "algorithm.rlt_transition_replay.enable": True,
        "algorithm.rlt_transition_replay.compact": True,
        "algorithm.rlt_transition_replay.bootstrap_on_truncation": True,
        "actor.micro_batch_size": 128,
        "actor.global_batch_size": 512,
        "actor.model.precision": "fp32",
        "actor.model.z_dim": 2048,
        "actor.model.proprio_dim": 14,
        "actor.model.action_dim": 14,
        "actor.model.num_action_chunks": 10,
        "actor.model.ref_num_action_chunks": 10,
        "actor.model.fixed_std": 0.002,
        "actor.optim.lr": 0.0001,
        "actor.critic_optim.lr": 0.0001,
        "rollout.rlt_feature_model.model_path": args.stage1_model_path,
        "rollout.rlt_feature_model.openpi.action_horizon": 50,
        "rollout.rlt_feature_model.openpi.action_chunk": 10,
        "rollout.rlt_feature_model.openpi.action_env_dim": 14,
        "rollout.rlt_feature_model.openpi.rlt_embed_dim": 2048,
        "rollout.rlt_feature_model.openpi.rlt_prefix_seq_len": 768,
        "rollout.rlt_feature_model.openpi.rlt_input_dim": 2048,
        "rollout.rlt_feature_model.openpi.rlt_action_adapter": (
            "robotwin_aloha_canonical_v1"
        ),
        "rollout.rlt_feature_model.openpi_data.norm_stats_path": (
            args.norm_stats_path
        ),
        "algorithm.rlt_resume.contract.stage1_manifest_path": args.manifest_path,
        "algorithm.rlt_resume.contract.stage1_manifest_id": args.manifest_id,
        "algorithm.rlt_resume.contract.stage1_manifest_sha256": (
            args.manifest_sha256
        ),
        "algorithm.rlt_resume.contract.norm_stats_sha256": args.norm_stats_sha256,
        "algorithm.rlt_resume.contract.canonical_adapter_version": (
            "robotwin_aloha_canonical_v1"
        ),
        "env.train.total_num_envs": 4,
        "env.eval.total_num_envs": 4,
        "weight_syncer.patch.init_sync.enabled": True,
    }
    for dotted_path, expected_value in expected.items():
        require_equal(
            f"{label}:{dotted_path}",
            select(config, dotted_path),
            expected_value,
        )

    placement = select(config, "cluster.component_placement")
    if not isinstance(placement, dict):
        raise ValueError(f"{label}: component_placement must resolve to a mapping.")
    require_equal(
        f"{label}:component placement",
        placement,
        {"actor, env, rollout": "0-1"},
    )

    world_size = 2
    gradient_accumulation = (
        int(select(config, "actor.global_batch_size"))
        // int(select(config, "actor.micro_batch_size"))
        // world_size
    )
    require_equal(f"{label}:gradient accumulation", gradient_accumulation, 2)


def audit(args: argparse.Namespace) -> dict:
    formal_path = Path(args.formal).resolve(strict=True)
    fresh_path = Path(args.fresh).resolve(strict=True)
    resume_path = Path(args.resume).resolve(strict=True)
    configs = {
        "formal": load_config(formal_path),
        "fresh": load_config(fresh_path),
        "resume": load_config(resume_path),
    }
    for label, config in configs.items():
        audit_common(config, args, label)

    formal = configs["formal"]
    require_equal("formal:max_steps fail-closed", select(formal, "runner.max_steps"), 0)
    require_equal(
        "formal:max_epochs candidate ceiling",
        select(formal, "runner.max_epochs"),
        1000,
    )

    fresh = configs["fresh"]
    fresh_expected = {
        "runner.max_steps": 1,
        "runner.val_check_interval": 1,
        "runner.save_interval": 1,
        "runner.resume_dir": None,
        "algorithm.rlt_schedule.max_updates_per_train_step": 20,
        "algorithm.rlt_schedule.warmup_min_size": 2,
        "algorithm.rlt_schedule.warmup_post_collect_updates": 8,
        "algorithm.rlt_schedule.train_every_transitions": 1,
        "algorithm.actor_weight_schedule.warmup_updates": 4,
        "algorithm.actor_weight_schedule.ramp_updates": 8,
        "algorithm.replay_buffer.cache_size": 64,
        "algorithm.replay_buffer.sample_window_size": 64,
        "env.train.max_episode_steps": 20,
        "env.train.max_steps_per_rollout_epoch": 20,
        "env.eval.max_episode_steps": 20,
        "env.eval.max_steps_per_rollout_epoch": 20,
    }
    for dotted_path, expected_value in fresh_expected.items():
        require_equal(
            f"fresh:{dotted_path}",
            select(fresh, dotted_path),
            expected_value,
        )

    resume = configs["resume"]
    require_equal("resume:max_steps", select(resume, "runner.max_steps"), 2)
    require_equal(
        "resume:resume_dir",
        select(resume, "runner.resume_dir"),
        args.resume_dir,
    )
    require_equal(
        "resume:experiment_name",
        select(resume, "runner.logger.experiment_name"),
        args.resume_experiment_name,
    )

    train_envs = int(select(fresh, "env.train.total_num_envs"))
    primitive_steps = int(select(fresh, "env.train.max_steps_per_rollout_epoch"))
    chunk = int(select(fresh, "actor.model.num_action_chunks"))
    full_length_global_rows = train_envs * primitive_steps // chunk
    warmup_updates = int(
        select(fresh, "algorithm.rlt_schedule.warmup_post_collect_updates")
    )
    update_epoch = int(select(fresh, "algorithm.update_epoch"))
    cap = int(select(fresh, "algorithm.rlt_schedule.max_updates_per_train_step"))
    ratio = int(select(fresh, "algorithm.critic_actor_ratio"))
    fresh_critic_updates = min(warmup_updates, cap)
    fresh_actor_updates = len(range(0, fresh_critic_updates, ratio))
    resume_desired_updates = warmup_updates + full_length_global_rows * update_epoch
    resume_pending_before_cap = resume_desired_updates - fresh_critic_updates
    resume_critic_updates = min(resume_pending_before_cap, cap)
    resume_actor_updates = len(
        range(
            fresh_critic_updates,
            fresh_critic_updates + resume_critic_updates,
            ratio,
        )
    )
    resume_final_update_step = fresh_critic_updates + resume_critic_updates
    resume_pending_after = resume_desired_updates - resume_final_update_step

    derived = {
        "gradient_accumulation": 2,
        "fresh_full_length_global_rows": full_length_global_rows,
        "fresh_full_length_rows_per_rank": full_length_global_rows // 2,
        "fresh_expected_critic_updates": fresh_critic_updates,
        "fresh_expected_actor_updates": fresh_actor_updates,
        "resume_full_length_expected_critic_updates": resume_critic_updates,
        "resume_full_length_expected_actor_updates": resume_actor_updates,
        "resume_full_length_final_update_step": resume_final_update_step,
        "resume_full_length_pending_after": resume_pending_after,
    }
    require_equal("fresh full-length global rows", full_length_global_rows, 8)
    require_equal("fresh critic updates", fresh_critic_updates, 8)
    require_equal("fresh actor updates", fresh_actor_updates, 4)
    require_equal("resume critic updates", resume_critic_updates, 20)
    require_equal("resume actor updates", resume_actor_updates, 10)
    require_equal("resume final update step", resume_final_update_step, 28)
    require_equal("resume pending after", resume_pending_after, 20)

    return {
        "passed": True,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "configs": {
            label: str(path)
            for label, path in {
                "formal": formal_path,
                "fresh": fresh_path,
                "resume": resume_path,
            }.items()
        },
        "derived_smoke_contract": derived,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", required=True)
    parser.add_argument("--fresh", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--stage1-model-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--norm-stats-path", required=True)
    parser.add_argument("--norm-stats-sha256", required=True)
    parser.add_argument("--resume-dir", required=True)
    parser.add_argument("--resume-experiment-name", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite resolved audit: {output}")
    payload = audit(args)
    output.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
