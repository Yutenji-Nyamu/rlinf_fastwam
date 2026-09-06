from __future__ import annotations

import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


worktree = Path(
    "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/"
    "rlt-pi0-robotwin-ar-7d07a421"
)
os.environ.setdefault("REPO_PATH", str(worktree))
os.environ.setdefault("ROBOTWIN_RLT_CLEAN50_PATH", "/tmp/compose-only-clean50")
os.environ.setdefault("RLT_STAGE1_MODEL_PATH", "/tmp/compose-only-stage1-model")

cases = [
    (
        worktree / "examples/sft/config",
        "robotwin_rlt_stage1_sft_openpi_current_ar",
    ),
    (
        worktree / "examples/embodiment/config",
        "robotwin_adjust_bottle_rlt_stage2_ac_mlp_current",
    ),
]

for index, (config_dir, config_name) in enumerate(cases):
    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
        job_name=f"rlt-compose-{index}",
    ):
        cfg = compose(config_name=config_name)
    OmegaConf.to_container(cfg, resolve=True)
    print(
        "COMPOSE_OK",
        config_name,
        f"placement={OmegaConf.to_container(cfg.cluster.component_placement)}",
        f"max_steps={cfg.runner.max_steps}",
    )
    if config_name.startswith("robotwin_rlt_stage1"):
        print(
            "STAGE1_KEYS",
            f"model={cfg.actor.model.model_type}",
            f"H={cfg.actor.model.openpi.action_horizon}",
            f"C={cfg.actor.model.openpi.action_chunk}",
            f"D={cfg.actor.model.openpi.action_env_dim}",
            f"AR=current",
            f"train_vla={cfg.actor.model.openpi.rlt_train_vla}",
            f"rtc={cfg.actor.model.openpi.rtc_enabled}",
        )
    else:
        print(
            "STAGE2_KEYS",
            f"route={cfg.algorithm.rlt_route.type}",
            f"compact={cfg.algorithm.rlt_transition_replay.compact}",
            "bootstrap="
            f"{cfg.algorithm.rlt_transition_replay.bootstrap_on_truncation}",
            f"rtc={cfg.rollout.rlt_feature_model.openpi.rtc_enabled}",
            f"initial_sync={cfg.weight_syncer.patch.init_sync.enabled}",
        )
