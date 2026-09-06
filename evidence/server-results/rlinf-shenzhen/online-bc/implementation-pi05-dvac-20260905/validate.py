"""Resolved same-budget formal comparison; no model/environment rollout."""
import hashlib
import json
import os
from pathlib import Path

import ray
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.config import validate_cfg

root = Path.cwd()
base = root.with_name("pi05-online-bc")
control_run = Path("/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc32x1-b1024-u10-m10-eval8x4-gpu6-formal100-20260905-v1")
packet = Path("/data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-pi05-dvac-20260905")
overrides = ["+online_bc_model=pi05_sidney", "+bc_dvac=bounded_half", "runner.max_epochs=100", "runner.val_check_interval=5", "runner.save_interval=10", "actor.optim.total_training_steps=1000", "runner.logger.experiment_name=pi05-pillbottle-bc-dvac-u10-w05to15-formal100-gpu7"]
with initialize_config_dir(version_base="1.1", config_dir=str(root / "examples/embodiment/config")):
    cfg = compose(config_name="robotwin_adjust_bottle_online_bc_openpi", overrides=overrides)
    try:
        cfg = validate_cfg(cfg)
    finally:
        ray.shutdown()

def flat(value, prefix=""):
    if isinstance(value, dict):
        return {k: v for name, item in value.items() for k, v in flat(item, f"{prefix}.{name}" if prefix else name).items()}
    return {prefix: value}

old = flat(OmegaConf.to_container(OmegaConf.load(control_run / "runtime/resolved.yaml"), resolve=True))
new = flat(OmegaConf.to_container(cfg, resolve=True))
changes = {k: {"control": old.get(k), "dvac": new.get(k)} for k in sorted(old.keys() | new.keys()) if old.get(k) != new.get(k)}
allowed = {"cluster.component_placement.actor,env,rollout", "runner.logger.experiment_name", "runner.logger.log_path", "algorithm.online_bc.data_path", "env.train.task_config.save_path", "env.eval.task_config.save_path", "env.train.video_cfg.video_base_dir", "env.eval.video_cfg.video_base_dir", "env.train.seeds_path", "env.eval.seeds_path"}
assert all(k.startswith("algorithm.online_bc.dvac.") or k in allowed for k in changes), changes
assert len(changes) == 17, changes
assert cfg.runner.resume_dir is None and cfg.runner.ckpt_path is None
assert cfg.actor.model.num_steps == cfg.actor.model.openpi.num_steps == 10
assert cfg.algorithm.online_bc.dvac.alpha == 0.125 and cfg.algorithm.online_bc.dvac.tail_steps == 3
assert cfg.actor.global_batch_size == 1024 and cfg.actor.micro_batch_size == 32 and cfg.algorithm.update_epoch == 10
assert cfg.runner.max_epochs == 100 and cfg.runner.val_check_interval == 5 and cfg.runner.save_interval == 10
assert cfg.env.train.total_num_envs == 32 and cfg.env.train.rollout_epoch == 1
assert cfg.env.eval.total_num_envs == 8 and cfg.env.eval.rollout_epoch == cfg.env.eval.fixed_reset_batch_count == 4
assert cfg.actor.model.openpi.train_expert_only and not cfg.actor.model.openpi.image_augmentation
assert cfg.actor.model.model_path == old["actor.model.model_path"]
assert not Path(os.environ["ONLINE_BC_RUN_DIR"]).exists()
# Seed path relocation must not change the actual fixed states or training seeds.
identical = {}
for name in ("rlinf/envs/robotwin/seeds/train_seeds.json", "rlinf/envs/robotwin/seeds/eval_sidney_fixed32.json", "rlinf/models/embodiment/openpi/dataconfig/__init__.py", "rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py", "examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml", "examples/embodiment/config/online_bc_model/pi05_sidney.yaml"):
    value = (root / name).read_bytes()
    assert value == (base / name).read_bytes(), name
    identical[name] = hashlib.sha256(value).hexdigest()
(packet / "resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
result = {"passed": True, "overrides": overrides, "changes": changes, "unchanged_files": identical, "no_smoke": True, "weight_bounds": [0.5, 1.5], "alpha": 0.125, "train_attempts": 3200, "max_optimizer_updates": 1000, "eval_episodes": 640, "checkpoints": 10}
(packet / "validation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("PI05_DVAC_UNIT_AND_FORMAL_CONFIG_VALIDATION_PASSED; no smoke or training launched")
