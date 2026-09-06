#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
python_bin=/root/autodl-tmp/RLinf/.venv/bin/python
resolved_ogpo=/tmp/ogpo_resolved_config_20260807_final.yaml
resolved_ppo=/tmp/ppo_resolved_config_20260807_regression.yaml

test "$(git -C "$repo" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git -C "$repo" branch --show-current)" = codex/ogpo-pi0-robotwin
test -x "$python_bin"

cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export REPO_PATH="$repo"
export EMBODIED_PATH="$repo/examples/embodiment"

printf 'FINAL_TEST_RUNTIME '
"$python_bin" -c \
  'import sys, torch; print(f"python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.cuda.is_available()}")'

"$python_bin" - <<'PY'
from pathlib import Path

files = (
    "examples/embodiment/train_embodied_agent.py",
    "rlinf/algorithms/ogpo/__init__.py",
    "rlinf/algorithms/ogpo/core.py",
    "rlinf/config.py",
    "rlinf/data/ogpo_replay.py",
    "rlinf/models/embodiment/base_policy.py",
    "rlinf/models/embodiment/modules/ogpo_critic.py",
    "rlinf/models/embodiment/modules/ogpo_modules.py",
    "rlinf/models/embodiment/openpi/__init__.py",
    "rlinf/models/embodiment/openpi/openpi_action_model.py",
    "rlinf/models/embodiment/openpi/openpi_ogpo.py",
    "rlinf/runners/embodied_runner.py",
    "rlinf/workers/actor/fsdp_ogpo_policy_worker.py",
    "rlinf/workers/env/env_worker.py",
    "rlinf/workers/rollout/hf/huggingface_worker.py",
    "tests/algorithms/test_ogpo_core.py",
    "tests/data/test_ogpo_replay.py",
    "tests/embodiment/test_ogpo_critic.py",
    "tests/embodiment/test_openpi_ogpo_adapter.py",
    "tests/embodiment/ogpo_fsdp_ema_fixture.py",
    "tests/embodiment/ogpo_real_fsdp_update_probe.py",
    "tests/embodiment/ogpo_real_fsdp_ema_probe.py",
    "tests/workers/test_ogpo_checkpoint_sidecar.py",
    "tests/workers/test_ogpo_env_trace.py",
    "tests/workers/test_ogpo_row_schedule.py",
)
for name in files:
    compile(Path(name).read_text(encoding="utf-8"), name, "exec")
print(f"SYNTAX_COMPILE_OK files={len(files)}")
PY

"$python_bin" -m pytest -q -p no:cacheprovider \
  tests/algorithms/test_ogpo_core.py \
  tests/data/test_ogpo_replay.py \
  tests/embodiment/test_ogpo_critic.py \
  tests/embodiment/test_openpi_ogpo_adapter.py \
  tests/workers/test_ogpo_checkpoint_sidecar.py \
  tests/workers/test_ogpo_env_trace.py \
  tests/workers/test_ogpo_row_schedule.py

"$python_bin" - <<'PY' > "$resolved_ogpo"
import os

import hydra
from omegaconf import OmegaConf

from rlinf.config import validate_cfg

config_dir = os.path.join(os.environ["EMBODIED_PATH"], "config")
with hydra.initialize_config_dir(version_base="1.1", config_dir=config_dir):
    cfg = hydra.compose(config_name="robotwin_adjust_bottle_ogpo_openpi")
cfg = validate_cfg(cfg)
print(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
PY

"$python_bin" - <<'PY' > "$resolved_ppo"
import os

import hydra
from omegaconf import OmegaConf

from rlinf.config import validate_cfg

config_dir = os.path.join(os.environ["EMBODIED_PATH"], "config")
with hydra.initialize_config_dir(version_base="1.1", config_dir=config_dir):
    cfg = hydra.compose(config_name="robotwin_adjust_bottle_ppo_openpi")
# The official PPO example targets eight GPUs.  Adapt only the in-memory
# placement so its unchanged legacy route can be validated on this 2-GPU host.
cfg.cluster.component_placement["actor, env, rollout"] = "0-1"
cfg = validate_cfg(cfg)
print(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
PY

"$python_bin" - <<'PY'
import os

import hydra
from omegaconf import OmegaConf

from rlinf.config import validate_cfg

bad_values = {
    "variant": "ogpo_plus",
    "model_horizon": 49,
    "model_action_dim": 31,
    "gaussian_clip": 2.0,
    "normalize_denoising_horizon": False,
    "normalize_act_space_dimension": False,
    "offline_ratio": 0.5,
    "use_success_buffer_q": True,
    "best_of_n": 2,
}
config_dir = os.path.join(os.environ["EMBODIED_PATH"], "config")
with hydra.initialize_config_dir(version_base="1.1", config_dir=config_dir):
    for name, value in bad_values.items():
        cfg = hydra.compose(config_name="robotwin_adjust_bottle_ogpo_openpi")
        OmegaConf.update(cfg, f"algorithm.ogpo.{name}", value)
        try:
            validate_cfg(cfg)
        except ValueError as error:
            if "unsupported locked values" not in str(error):
                raise
        else:
            raise AssertionError(f"locked OGPO field was silently accepted: {name}")
print(f"LOCKED_CONFIG_GUARDS_OK fields={len(bad_values)}")
PY

printf 'CONFIG_COMPOSE_OK ogpo_path=%s lines=%s sha256=%s\n' \
  "$resolved_ogpo" \
  "$(wc -l < "$resolved_ogpo")" \
  "$(sha256sum "$resolved_ogpo" | awk '{print $1}')"
printf 'LEGACY_PPO_COMPOSE_OK path=%s lines=%s sha256=%s\n' \
  "$resolved_ppo" \
  "$(wc -l < "$resolved_ppo")" \
  "$(sha256sum "$resolved_ppo" | awk '{print $1}')"

if "$python_bin" -m ruff --version >/dev/null 2>&1; then
  "$python_bin" -m ruff check --no-cache \
    examples/embodiment/train_embodied_agent.py \
    rlinf/algorithms/ogpo \
    rlinf/config.py \
    rlinf/data/ogpo_replay.py \
    rlinf/models/embodiment/base_policy.py \
    rlinf/models/embodiment/modules/ogpo_critic.py \
    rlinf/models/embodiment/modules/ogpo_modules.py \
    rlinf/models/embodiment/openpi/__init__.py \
    rlinf/models/embodiment/openpi/openpi_action_model.py \
    rlinf/models/embodiment/openpi/openpi_ogpo.py \
    rlinf/runners/embodied_runner.py \
    rlinf/workers/actor/fsdp_ogpo_policy_worker.py \
    rlinf/workers/env/env_worker.py \
    rlinf/workers/rollout/hf/huggingface_worker.py \
    tests/algorithms/test_ogpo_core.py \
    tests/data/test_ogpo_replay.py \
    tests/embodiment/test_ogpo_critic.py \
    tests/embodiment/test_openpi_ogpo_adapter.py \
    tests/embodiment/ogpo_real_fsdp_update_probe.py \
    tests/workers/test_ogpo_checkpoint_sidecar.py \
    tests/workers/test_ogpo_env_trace.py \
    tests/workers/test_ogpo_row_schedule.py
  printf 'RUFF_OK\n'
else
  printf 'RUFF_UNAVAILABLE\n'
fi

git diff --check
printf 'FINAL_DIFF_CHECK_OK\n'
printf 'FINAL_STATUS\n'
git status --short
printf 'FINAL_GPU_PROCESSES\n'
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits || true
printf 'FINAL_RAY_PROCESSES\n'
pgrep -af '[r]aylet|[g]cs_server|[r]ay::' || true
