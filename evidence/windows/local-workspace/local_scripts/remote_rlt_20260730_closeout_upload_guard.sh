#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1/fresh_runtime
completion=/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1/robotwin_adjust_bottle_rlt_stage2_smoke_fresh_v1/checkpoints/global_step_1/actor/sac_components/rlt_trainer_state/rlt_trainer_state_complete.json
stage=/root/autodl-tmp/tmp/rlt_stage2_fresh_closeout_20260730_v1
expected_head=6fd3ee7106fb82f06eda82603c41a09767151709

cd "${repo}"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git rev-parse HEAD)" = "${expected_head}"
test -z "$(git status --short)"
read -r left right < <(git rev-list --left-right --count '@{upstream}...HEAD')
test "${left}" = 0
test "${right}" = 0
test ! -e "${stage}"
test "$(tr -d '\n' < "${runtime}/exit_code.txt")" = 0
test -f "${completion}"

if pgrep -af \
  'train_embodied_agent.py|rlt_stage2_smoke|raylet|gcs_server' \
  | grep -vE 'pgrep -af|closeout_upload_guard' >/dev/null; then
  printf '%s\n' 'unexpected RLT/Ray process'
  exit 1
fi

mapfile -t gpu_rows < <(
  nvidia-smi --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits
)
test "${#gpu_rows[@]}" = 2
for row in "${gpu_rows[@]}"; do
  used="${row%%,*}"
  util="${row##*,}"
  used="${used// /}"
  util="${util// /}"
  test "${used}" -le 16
  test "${util}" -le 5
done

/root/autodl-tmp/RLinf/.venv/bin/python -B - "${completion}" <<'PY'
import json
import sys
from pathlib import Path

obj = json.loads(Path(sys.argv[1]).read_text())
assert obj["complete"] is True
assert obj["actor_world_size"] == 2
assert obj["update_step"] == 8
assert [entry["rank"] for entry in obj["files"]] == [0, 1]
assert all(entry["update_step"] == 8 for entry in obj["files"])
PY

printf 'guard_time\t%s\n' "$(date --iso-8601=seconds)"
printf 'branch\t%s\n' "$(git branch --show-current)"
printf 'head\t%s\n' "$(git rev-parse HEAD)"
printf 'left_right\t%s/%s\n' "${left}" "${right}"
printf 'gpu_rows\t%s | %s\n' "${gpu_rows[0]}" "${gpu_rows[1]}"
printf 'host_available_kib\t%s\n' "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
printf 'disk_available_bytes\t%s\n' "$(df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' ')"
printf 'smoke_exit\t0\n'
printf 'saved_update_step\t8\n'
printf '%s\n' RLT_STAGE2_FRESH_CLOSEOUT_UPLOAD_GUARD_OK
