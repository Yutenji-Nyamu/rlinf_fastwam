set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
pycache=$(mktemp -d /tmp/rlinf-dsrl-pycache.XXXXXX)
trap 'rm -rf "$pycache"' EXIT

cd "$worktree"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$worktree"
export PYTHONPYCACHEPREFIX="$pycache"
export REPO_PATH="$worktree"
export EMBODIED_PATH="$worktree/examples/embodiment"

python_bin="$venv/bin/python"
ruff_bin="$venv/bin/ruff"

"$python_bin" -m py_compile \
  rlinf/data/storage/replay/dsrl_transition.py \
  rlinf/data/storage/replay/__init__.py \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/workers/actor/fsdp_sac_policy_worker.py \
  tests/unit_tests/test_dsrl_transition_replay.py \
  tests/unit_tests/test_dsrl_target_shadow_resume.py
echo DSRL_COMPILE_OK

if test -x "$ruff_bin"; then
  "$ruff_bin" check \
    rlinf/data/storage/replay/dsrl_transition.py \
    rlinf/data/storage/replay/__init__.py \
    rlinf/models/embodiment/openpi/openpi_action_model.py \
    rlinf/workers/actor/fsdp_sac_policy_worker.py \
    tests/unit_tests/test_dsrl_transition_replay.py \
    tests/unit_tests/test_dsrl_target_shadow_resume.py
else
  "$python_bin" -m ruff check \
    rlinf/data/storage/replay/dsrl_transition.py \
    rlinf/data/storage/replay/__init__.py \
    rlinf/models/embodiment/openpi/openpi_action_model.py \
    rlinf/workers/actor/fsdp_sac_policy_worker.py \
    tests/unit_tests/test_dsrl_transition_replay.py \
    tests/unit_tests/test_dsrl_target_shadow_resume.py
fi
echo DSRL_RUFF_OK

"$python_bin" -m pytest -q \
  tests/unit_tests/test_dsrl_transition_replay.py \
  tests/unit_tests/test_dsrl_target_shadow_resume.py
echo DSRL_FOCUSED_TESTS_OK

"$python_bin" - <<'PY'
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

config_dir = str(Path("examples/embodiment/config").resolve())
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(
        config_name="robotwin_adjust_bottle_dsrl_openpi"
    )
OmegaConf.resolve(cfg)
assert cfg.algorithm.replay_buffer.type == "dsrl_transition"
assert cfg.algorithm.utd_ratio == 20
assert cfg.actor.model.openpi.use_dsrl is True
assert cfg.actor.model.openpi.rtc_enabled is False
assert cfg.actor.model.openpi.dsrl_gaussian_warmup is True
assert cfg.actor.model.openpi.dsrl_eval_deterministic is False
assert cfg.actor.model.num_action_chunks == 20
assert cfg.actor.model.openpi.action_horizon == 50
assert cfg.actor.fsdp_config.save_full_model_weights is False
print("DSRL_HYDRA_COMPOSE_OK")
print(OmegaConf.to_yaml(cfg, resolve=True))
PY

git diff --check
git status --short
