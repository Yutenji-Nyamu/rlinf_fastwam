"""Configure a new RLT run to recover its BC actor and schedule Q in rounds."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch


def configure(
    config_path, resume_checkpoint, policy_path, bc_rounds, ramp_rounds, q_weight
):
    cfg = OmegaConf.load(config_path)
    resume_checkpoint = Path(resume_checkpoint).resolve()
    policy_path = Path(policy_path).resolve()
    state_path = (
        resume_checkpoint
        / "actor/sac_components/rlt_trainer_state/checkpoint_rank_0.pt"
    )
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    anchor = int(state["update_step"])
    env = cfg.env.train
    transitions = (
        int(env.total_num_envs)
        * int(env.max_steps_per_rollout_epoch)
        * int(env.get("rollout_epoch", 1))
        // int(cfg.actor.model.num_action_chunks)
    )
    updates_per_round = transitions * int(cfg.algorithm.update_epoch)
    if int(cfg.algorithm.rlt_schedule.train_every_transitions) != 1:
        raise ValueError("Expected transition-based RLT schedule.")
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)
    cfg.runner.resume_dir = str(resume_checkpoint)
    cfg.algorithm.rlt_actor_recovery = {
        "enable": True,
        "resume_checkpoint": str(resume_checkpoint),
        "policy_path": str(policy_path),
    }
    cfg.algorithm.actor_weight_schedule.update(
        enable=True,
        warmup_updates=anchor + bc_rounds * updates_per_round,
        ramp_updates=ramp_rounds * updates_per_round,
        warmup_bc_weight=1.0,
        warmup_q_weight=0.0,
        online_bc_weight=1.0,
        online_q_weight=q_weight,
    )
    cfg.algorithm.reference_dropout_prob = 0.0
    OmegaConf.save(cfg, config_path)
    return {
        "resume_checkpoint": str(resume_checkpoint),
        "policy_path": str(policy_path),
        "update_anchor": anchor,
        "transitions_per_round": transitions,
        "updates_per_round": updates_per_round,
        "bc_rounds": bc_rounds,
        "ramp_rounds": ramp_rounds,
        "q_weight": q_weight,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-checkpoint", required=True)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--bc-rounds", type=int, default=2)
    parser.add_argument("--ramp-rounds", type=int, default=10)
    parser.add_argument("--q-weight", type=float, default=0.005)
    args = parser.parse_args()
    print(
        json.dumps(
            configure(
                args.config,
                args.resume_checkpoint,
                args.policy_path,
                args.bc_rounds,
                args.ramp_rounds,
                args.q_weight,
            ),
            indent=2,
        )
    )
