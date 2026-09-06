#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1/fresh_runtime
ckpt=/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1/robotwin_adjust_bottle_rlt_stage2_smoke_fresh_v1/checkpoints/global_step_1
completion="${ckpt}/actor/sac_components/rlt_trainer_state/rlt_trainer_state_complete.json"

cd "${repo}"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test -z "$(git status --short)"
read -r left right < <(git rev-list --left-right --count '@{upstream}...HEAD')
test "${left}" = 0
test "${right}" = 0
test "$(tr -d '\n' < "${runtime}/exit_code.txt")" = 0
test -d "${ckpt}"

mapfile -t process_rows < <(
  pgrep -af 'train_embodied_agent.py|rlt_stage2_smoke|raylet|gcs_server' \
    || true
)
process_count="${#process_rows[@]}"
test "${process_count}" = 0

formal_max_steps="$(
  /root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
from omegaconf import OmegaConf
cfg = OmegaConf.load(
    "examples/embodiment/config/"
    "robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml"
)
print(cfg.runner.max_steps)
PY
)"
test "${formal_max_steps}" = 0

saved_update_step="$(
  /root/autodl-tmp/RLinf/.venv/bin/python -B - "${completion}" <<'PY'
import json
import sys
from pathlib import Path
obj = json.loads(Path(sys.argv[1]).read_text())
assert obj["complete"] is True
print(obj["update_step"])
PY
)"
test "${saved_update_step}" = 8

printf 'audit_time\t%s\n' "$(date --iso-8601=seconds)"
printf 'branch\t%s\n' "$(git branch --show-current)"
printf 'head\t%s\n' "$(git rev-parse HEAD)"
printf 'left_right\t%s/%s\n' "${left}" "${right}"
printf 'dirty_count\t0\n'
printf 'rlt_ray_process_count\t%s\n' "${process_count}"
printf 'gpu_rows\t%s\n' "$(
  nvidia-smi --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    | paste -sd '|' -
)"
printf 'fresh_exit\t0\n'
printf 'saved_update_step\t%s\n' "${saved_update_step}"
printf 'checkpoint_bytes\t%s\n' "$(du -sb "${ckpt}" | cut -f1)"
printf 'formal_max_steps\t%s\n' "${formal_max_steps}"
printf 'host_available_kib\t%s\n' "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
printf 'disk_available_bytes\t%s\n' "$(df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' ')"
printf '%s\n' RLT_STAGE2_FRESH_FINAL_CLOSEOUT_AUDIT_OK
