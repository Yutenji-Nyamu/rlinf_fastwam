#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
python_bin=/root/autodl-tmp/RLinf/.venv/bin/python
resolved=/tmp/ogpo_resolved_config_20260807.yaml

test "$(git -C "$repo" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git -C "$repo" branch --show-current)" = codex/ogpo-pi0-robotwin
test -x "$python_bin"

cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export REPO_PATH="$repo"
export EMBODIED_PATH="$repo/examples/embodiment"

printf 'TEST_RUNTIME '
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
    "tests/embodiment/ogpo_real_fsdp_ema_probe.py",
    "tests/workers/test_ogpo_checkpoint_sidecar.py",
)
for name in files:
    source = Path(name).read_text(encoding="utf-8")
    compile(source, name, "exec")
print(f"SYNTAX_COMPILE_OK files={len(files)}")
PY

"$python_bin" -m pytest -q -p no:cacheprovider \
  tests/algorithms/test_ogpo_core.py \
  tests/data/test_ogpo_replay.py \
  tests/embodiment/test_ogpo_critic.py \
  tests/embodiment/test_openpi_ogpo_adapter.py \
  tests/workers/test_ogpo_checkpoint_sidecar.py

"$python_bin" - <<'PY' > "$resolved"
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

printf 'CONFIG_COMPOSE_OK path=%s lines=%s sha256=%s\n' \
  "$resolved" \
  "$(wc -l < "$resolved")" \
  "$(sha256sum "$resolved" | awk '{print $1}')"

git diff --check
printf 'POST_TEST_STATUS\n'
git status --short
