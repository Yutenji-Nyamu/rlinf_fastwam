#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
OUT=/data/chenyiteng/results/rlinf-shenzhen/grpo/pretest-current-dvac-global-z-20260824-v3

printf 'MARKER=SZ_CURRENT_DVAC_GRPO_FOCUSED_CHECKS_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
test "$(git -C "$WT" branch --show-current)" = codex/sz-current-pi0-dvac-grpo
git -C "$WT" diff --cached --check
test ! -e "$OUT"
install -d -m 755 "$OUT"

source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES=""
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1

"$VENV/bin/python" -m pytest -q \
  "$WT/tests/unit_tests/test_dvac_train_weighting.py" \
  "$WT/tests/unit_tests/test_dvac_telemetry.py" | tee "$OUT/pytest.txt"

OVERRIDES=(
  "runner.logger.log_path=$OUT/not-launched"
  "env.train.assets_path=$ROBOTWIN"
  "env.eval.assets_path=$ROBOTWIN"
  "actor.model.model_path=$MODEL"
)
"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  --config-path "$WT/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_grpo_openpi "${OVERRIDES[@]}" \
  --cfg job --resolve > "$OUT/base.resolved.yaml"
"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  --config-path "$WT/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_grpo_openpi "${OVERRIDES[@]}" \
  algorithm.dvac_gradient_weighting.mode=apply \
  --cfg job --resolve > "$OUT/dvac.resolved.yaml"

"$VENV/bin/python" - "$OUT/base.resolved.yaml" "$OUT/dvac.resolved.yaml" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

base = OmegaConf.load(sys.argv[1])
dvac = OmegaConf.load(sys.argv[2])
base_weighting = OmegaConf.to_container(base.algorithm.pop("dvac_gradient_weighting"), resolve=True)
weighting = OmegaConf.to_container(dvac.algorithm.pop("dvac_gradient_weighting"), resolve=True)
assert OmegaConf.to_container(base, resolve=True) == OmegaConf.to_container(dvac, resolve=True)
assert base_weighting["mode"] == "off"
assert weighting == {
    "mode": "apply",
    "selected_l": 3,
    "warmup_steps": 1,
    "window_steps": 5,
    "log_eps": 1e-12,
    "std_floor": 1e-6,
    "z_clip": 2.0,
    "strength": 0.5,
    "save_step_tensors": True,
    "output_dir": str(Path(base.runner.logger.log_path) / base.runner.logger.experiment_name / "dvac_train"),
}
print("BASE_PARITY_AND_DVAC_CONFIG_OK")
PY

git -C "$WT" diff --cached --stat > "$OUT/diff.stat.txt"
sha256sum "$OUT/base.resolved.yaml" "$OUT/dvac.resolved.yaml"
du -ah "$OUT" | sort -h
printf 'MARKER=SZ_CURRENT_DVAC_GRPO_FOCUSED_CHECKS_OK\n'
