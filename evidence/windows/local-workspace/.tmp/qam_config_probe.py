from __future__ import annotations

import hashlib
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.config import _validate_embodied_qam_contract, validate_cfg


CONFIG_DIR = Path(
    "/root/autodl-tmp/RLinf_qam_pi0_robotwin/examples/embodiment/config"
)


with initialize_config_dir(
    version_base="1.1",
    config_dir=str(CONFIG_DIR),
):
    qam = compose(config_name="robotwin_adjust_bottle_qam_openpi")

qam = validate_cfg(qam)
resolved = OmegaConf.to_container(qam, resolve=True)
assert resolved["runner"]["max_steps"] == 0
assert resolved["algorithm"]["loss_type"] == "embodied_qam"
assert resolved["algorithm"]["qam"]["phase"] == "collect"
assert resolved["algorithm"]["qam"]["inv_temp"] == 0.0
assert resolved["actor"]["model"]["openpi"]["use_qam"] is True
assert resolved["actor"]["model"]["openpi"]["use_dsrl"] is False
assert resolved["actor"]["model"]["openpi"]["use_rlt"] is False
assert resolved["rollout"]["collect_transitions"] is True
assert resolved["actor"]["global_batch_size"] == 64
assert resolved["actor"]["micro_batch_size"] == 32
assert resolved["algorithm"]["qam"]["replay_capacity"] == 4096
source_resolved_path = Path(
    "/root/autodl-tmp/qam_source_resolved_20260731_v1.yaml"
)
OmegaConf.save(config=qam, f=source_resolved_path, resolve=True)
source_resolved_sha256 = hashlib.sha256(
    source_resolved_path.read_bytes()
).hexdigest()
print(
    "QAM_COMPOSE_OK",
    {
        "path": str(source_resolved_path),
        "sha256": source_resolved_sha256,
        "max_steps": resolved["runner"]["max_steps"],
        "phase": resolved["algorithm"]["qam"]["phase"],
        "global_batch": resolved["actor"]["global_batch_size"],
        "micro_batch": resolved["actor"]["micro_batch_size"],
        "envs": resolved["env"]["train"]["total_num_envs"],
        "use_qam": resolved["actor"]["model"]["openpi"]["use_qam"],
        "use_dsrl": resolved["actor"]["model"]["openpi"]["use_dsrl"],
        "use_rlt": resolved["actor"]["model"]["openpi"]["use_rlt"],
    },
)

with initialize_config_dir(
    version_base="1.1",
    config_dir=str(CONFIG_DIR),
):
    smoke = compose(
        config_name="robotwin_adjust_bottle_qam_openpi",
        overrides=[
            "runner.logger.log_path=/root/autodl-tmp/experiments/qam_qonly_smoke_20260731_v1",
            "runner.logger.experiment_name=robotwin_adjust_bottle_qam_qonly_smoke_20260731_v1",
            "runner.max_steps=1",
            "runner.save_interval=1",
            "algorithm.qam.phase=q_only",
            "algorithm.qam.warmup_global_inserts=2",
            "algorithm.qam.min_replay_per_rank=1",
            "algorithm.qam.max_updates_per_step=2",
            "actor.global_batch_size=2",
            "actor.micro_batch_size=1",
            "+actor.fsdp_config.save_full_model_weights=false",
        ],
    )

smoke = validate_cfg(smoke)
smoke_resolved = OmegaConf.to_container(smoke, resolve=True)
assert smoke_resolved["runner"]["max_steps"] == 1
assert smoke_resolved["runner"]["save_interval"] == 1
assert smoke_resolved["runner"]["resume_dir"] is None
assert smoke_resolved["algorithm"]["qam"]["phase"] == "q_only"
assert smoke_resolved["algorithm"]["qam"]["inv_temp"] == 0.0
assert smoke_resolved["algorithm"]["qam"]["am_evidence_passed"] is False
assert smoke_resolved["algorithm"]["qam"]["warmup_global_inserts"] == 2
assert smoke_resolved["algorithm"]["qam"]["min_replay_per_rank"] == 1
assert smoke_resolved["algorithm"]["qam"]["max_updates_per_step"] == 2
assert smoke_resolved["actor"]["global_batch_size"] == 2
assert smoke_resolved["actor"]["micro_batch_size"] == 1
assert smoke_resolved["algorithm"]["qam"]["replay_capacity"] == 4096
assert smoke_resolved["actor"]["fsdp_config"]["save_full_model_weights"] is False
resolved_path = Path(
    "/root/autodl-tmp/qam_qonly_smoke_resolved_20260731_v1.yaml"
)
OmegaConf.save(config=smoke, f=resolved_path, resolve=True)
resolved_sha256 = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
print(
    "QAM_QONLY_SMOKE_COMPOSE_OK",
    {
        "path": str(resolved_path),
        "sha256": resolved_sha256,
        "max_steps": smoke_resolved["runner"]["max_steps"],
        "phase": smoke_resolved["algorithm"]["qam"]["phase"],
        "global_batch": smoke_resolved["actor"]["global_batch_size"],
        "micro_batch": smoke_resolved["actor"]["micro_batch_size"],
        "max_updates": smoke_resolved["algorithm"]["qam"][
            "max_updates_per_step"
        ],
        "save_full_model_weights": smoke_resolved["actor"]["fsdp_config"][
            "save_full_model_weights"
        ],
    },
)


with initialize_config_dir(
    version_base="1.1",
    config_dir=str(CONFIG_DIR),
):
    legacy = compose(config_name="robotwin_adjust_bottle_ppo_openpi")

_validate_embodied_qam_contract(legacy, only_eval=False)
assert legacy.algorithm.loss_type == "actor_critic"
assert not bool(
    OmegaConf.select(
        legacy,
        "actor.model.openpi.use_qam",
        default=False,
    )
)
print(
    "LEGACY_QAM_OFF_OK",
    {
        "loss_type": legacy.algorithm.loss_type,
        "use_qam": bool(
            OmegaConf.select(
                legacy,
                "actor.model.openpi.use_qam",
                default=False,
            )
        ),
    },
)
