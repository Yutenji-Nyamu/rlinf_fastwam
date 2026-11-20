"""Apply the 256-episode Clean GRPO recipe to the native Dojo task."""
from omegaconf import OmegaConf


def configure(template, plan, directory, *, final=False):
    c = OmegaConf.load(template)
    cards = plan["gpus"]
    world = len(cards)
    concurrent = world * plan["native_envs_per_card"]
    assert 256 % concurrent == 0 and plan["native_envs_per_card"] % 8 == 0
    for mode in ("train", "eval"):
        env = c.env[mode]
        env.task_config.task_name = plan["task"]
        env.physical_gpus = cards
        env.rpc_dir = str(directory / "rpc")
        env.reset_timeout = 3600
        env.total_num_envs = concurrent if mode == "train" else 32
        env.rollout_epoch = 256 // concurrent if mode == "train" else (4 if final else 1)
        env.group_size = 8 if mode == "train" else 1
        env.max_episode_steps = env.max_steps_per_rollout_epoch = 400
        assert env.seed == 42 and env.is_eval == (mode == "eval")
    for side in (c.actor.model, c.rollout.model):
        side.num_steps = side.openpi.num_steps = 10
        side.openpi.noise_level = 0.3
        assert side.openpi.train_expert_only and side.openpi.action_chunk == 50
    c.actor.global_batch_size = 1024
    c.actor.micro_batch_size = 32
    assert c.algorithm.update_epoch == 2 and c.algorithm.group_size == 8
    assert c.actor.optim.lr == 5e-6 and c.algorithm.clip_ratio_high == c.algorithm.clip_ratio_low == 0.2
    assert c.algorithm.adv_type == "grpo" and c.algorithm.dvac_gradient_weighting.mode == "off"
    c.cluster.component_placement["actor, env, rollout"] = ",".join(map(str, cards))
    c.runner.logger.log_path = plan["run"]
    c.runner.logger.experiment_name = "pour-clean-256"
    c.runner.max_steps = plan["rounds"]
    c.runner.val_check_interval = 5
    c.runner.save_interval = 5
    c.runner.resume_dir = str(directory_checkpoint(plan)) if final else None
    c.algorithm.dvac_gradient_weighting.output_dir = str(directory / "diagnostics")
    return c


def directory_checkpoint(plan):
    from pathlib import Path
    return Path(plan["run"]) / "pour-clean-256/checkpoints" / f"global_step_{plan['rounds']}"

