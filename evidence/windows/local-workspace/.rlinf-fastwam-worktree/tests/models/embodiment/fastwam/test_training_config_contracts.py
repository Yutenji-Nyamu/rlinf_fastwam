"""Static contracts for Fast-WAM GRPO/PPO training configurations.

These tests deliberately avoid importing Fast-WAM or loading its checkpoint. They
protect the experiment arithmetic and actor/rollout critic schema before a GPU
smoke is attempted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = REPO_ROOT / "examples" / "embodiment" / "config"


def _load_yaml(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _different_paths(
    left: Any, right: Any, prefix: tuple[str, ...] = ()
) -> set[tuple[str, ...]]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: set[tuple[str, ...]] = set()
        for key in left.keys() | right.keys():
            paths |= _different_paths(left.get(key), right.get(key), (*prefix, key))
        return paths
    if left != right:
        return {prefix}
    return set()


def test_pi0_aligned_grpo_changes_only_the_four_approved_fields() -> None:
    baseline = _load_yaml("robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu.yaml")
    aligned = _load_yaml(
        "robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu_pi0_aligned.yaml"
    )

    assert _different_paths(baseline, aligned) == {
        ("runner", "logger", "experiment_name"),
        ("algorithm", "update_epoch"),
        ("env", "train", "rollout_epoch"),
        ("actor", "global_batch_size"),
    }
    assert aligned["env"]["train"]["total_num_envs"] == 4
    assert aligned["env"]["train"]["rollout_epoch"] == 32
    assert aligned["algorithm"]["group_size"] == 4
    assert aligned["algorithm"]["normalize_advantages"] is True
    assert aligned["algorithm"]["update_epoch"] == 2
    assert aligned["actor"]["global_batch_size"] == 512
    assert aligned["actor"]["micro_batch_size"] == 2
    assert aligned["actor"]["optim"]["lr"] == 5.0e-6

    # H/N=192/24=8 transitions per trajectory. This matches the successful
    # pi0 recipe at 1024 unique transitions, 2048 sample presentations and four
    # optimizer updates per RL step.
    transitions = 4 * 32 * 8
    assert transitions == 1024
    assert transitions * aligned["algorithm"]["update_epoch"] == 2048
    assert (
        transitions
        // aligned["actor"]["global_batch_size"]
        * aligned["algorithm"]["update_epoch"]
        == 4
    )


def test_fastwam_model_defaults_remain_critic_free() -> None:
    model_cfg = _load_yaml("model/fastwam_robotwin.yaml")
    assert model_cfg["add_value_head"] is False
    assert model_cfg["value_head"] == {
        "feature_source": "video_cache_last_v_mean",
        "input_dim": 3072,
        "hidden_sizes": [1024, 512, 256],
        "activation": "relu",
        "bias_last": True,
        "detach_input": True,
    }


def _assert_ppo_contract(cfg: dict[str, Any]) -> None:
    assert cfg["algorithm"]["adv_type"] == "gae"
    assert cfg["algorithm"]["loss_type"] == "actor_critic"
    assert cfg["algorithm"]["group_size"] == 1
    assert cfg["algorithm"]["normalize_advantages"] is True
    assert cfg["algorithm"]["filter_rewards"] is False
    assert cfg["algorithm"]["gamma"] == 0.99
    assert cfg["algorithm"]["gae_lambda"] == 0.95
    assert cfg["algorithm"]["value_clip"] == 0.2
    assert cfg["algorithm"]["huber_delta"] == 10.0
    assert cfg["critic"]["use_critic_model"] is False

    # Fresh official-base initialization: the top-level PPO config does not
    # provide either a DCP resume path or a replacement policy checkpoint.
    assert cfg["runner"]["resume_dir"] is None
    assert cfg["runner"]["ckpt_path"] is None
    assert "model_path" not in cfg["actor"]["model"]

    actor_model = cfg["actor"]["model"]
    value_cfg = actor_model["value_head"]
    assert actor_model["add_value_head"] is True
    assert actor_model["model_forward_batch_size"] == 2
    assert actor_model["rl"]["noise_level"] == 0.3
    assert value_cfg == {
        "feature_source": "video_cache_last_v_mean",
        "input_dim": 3072,
        "hidden_sizes": [1024, 512, 256],
        "activation": "relu",
        "bias_last": True,
        "detach_input": True,
    }

    rollout_model = cfg["rollout"]["model"]
    assert rollout_model["add_value_head"] == "${actor.model.add_value_head}"
    assert rollout_model["model_forward_batch_size"] == (
        "${actor.model.model_forward_batch_size}"
    )
    for key in value_cfg:
        assert rollout_model["value_head"][key] == (
            f"${{actor.model.value_head.{key}}}"
        )

    assert cfg["rollout"]["collect_prev_infos"] is True
    assert cfg["rollout"]["enable_offload"] is True
    assert cfg["env"]["train"]["enable_offload"] is True
    assert cfg["actor"]["enable_offload"] is False
    assert cfg["actor"]["micro_batch_size"] == 2
    assert cfg["actor"]["optim"]["lr"] == 5.0e-6
    assert cfg["actor"]["optim"]["value_lr"] == 1.1e-4


def test_fastwam_ppo_smoke_is_exactly_one_real_optimizer_step() -> None:
    cfg = _load_yaml("robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu_smoke.yaml")
    _assert_ppo_contract(cfg)

    transitions = (
        cfg["env"]["train"]["total_num_envs"] * cfg["env"]["train"]["rollout_epoch"] * 8
    )
    assert cfg["env"]["train"]["total_num_envs"] == 4
    assert cfg["env"]["train"]["rollout_epoch"] == 1
    assert transitions == 32
    assert cfg["actor"]["global_batch_size"] == 32
    assert cfg["algorithm"]["update_epoch"] == 1
    assert transitions // cfg["actor"]["global_batch_size"] == 1
    assert cfg["runner"]["max_steps"] == 1
    assert cfg["runner"]["save_interval"] == 1


def test_fastwam_ppo_formal_matches_pi0_aligned_grpo_training_budget() -> None:
    cfg = _load_yaml("robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu.yaml")
    grpo = _load_yaml(
        "robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu_pi0_aligned.yaml"
    )
    _assert_ppo_contract(cfg)

    # PPO differs algorithmically (GAE/actor-critic/group1/value head), while
    # its sampling depth, effective actor batch and sample reuse deliberately
    # match the approved pi0-aligned Fast-WAM GRPO recipe.
    assert cfg["env"]["train"]["total_num_envs"] == 4
    assert cfg["env"]["train"]["rollout_epoch"] == 32
    assert cfg["env"]["train"]["total_num_envs"] == grpo["env"]["train"][
        "total_num_envs"
    ]
    assert cfg["env"]["train"]["rollout_epoch"] == grpo["env"]["train"][
        "rollout_epoch"
    ]
    assert cfg["actor"]["global_batch_size"] == 512
    assert cfg["actor"]["global_batch_size"] == grpo["actor"]["global_batch_size"]
    assert cfg["actor"]["micro_batch_size"] == grpo["actor"]["micro_batch_size"]
    assert cfg["algorithm"]["update_epoch"] == grpo["algorithm"]["update_epoch"]

    transitions = (
        cfg["env"]["train"]["total_num_envs"] * cfg["env"]["train"]["rollout_epoch"] * 8
    )
    assert transitions == 1024
    assert cfg["algorithm"]["update_epoch"] == 2
    assert (
        transitions
        // cfg["actor"]["global_batch_size"]
        * cfg["algorithm"]["update_epoch"]
        == 4
    )
    assert transitions * cfg["algorithm"]["update_epoch"] == 2048
    assert cfg["runner"]["max_steps"] == 100
    assert cfg["runner"]["save_interval"] == 10


def test_fastwam_ppo_formal_differs_from_aligned_grpo_only_by_ppo_contract() -> None:
    grpo = _load_yaml(
        "robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu_pi0_aligned.yaml"
    )
    ppo = _load_yaml("robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu.yaml")

    assert _different_paths(grpo, ppo) == {
        ("runner", "logger", "experiment_name"),
        ("algorithm", "adv_type"),
        ("algorithm", "filter_rewards"),
        ("algorithm", "group_size"),
        ("algorithm", "loss_type"),
        ("actor", "model", "add_value_head"),
        ("actor", "model", "model_forward_batch_size"),
        ("actor", "model", "value_head"),
        ("actor", "optim", "value_lr"),
        ("rollout", "model", "add_value_head"),
        ("rollout", "model", "model_forward_batch_size"),
        ("rollout", "model", "value_head"),
    }


def test_fastwam_ppo_smoke_only_reduces_serial_budget_and_lifecycle() -> None:
    smoke = _load_yaml("robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu_smoke.yaml")
    formal = _load_yaml("robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu.yaml")

    assert _different_paths(smoke, formal) == {
        ("runner", "logger", "experiment_name"),
        ("runner", "max_steps"),
        ("runner", "save_interval"),
        ("algorithm", "update_epoch"),
        ("env", "train", "rollout_epoch"),
        ("actor", "global_batch_size"),
    }


def test_ppo_launcher_selects_only_ppo_configs() -> None:
    launcher = (
        REPO_ROOT / "examples" / "embodiment" / "run_fastwam_robotwin_ppo.sh"
    ).read_text(encoding="utf-8")
    assert "robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu_smoke" in launcher
    assert "robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu" in launcher
    assert "robotwin_move_stapler_pad_grpo" not in launcher
    assert "monitor_resources.py" in launcher
