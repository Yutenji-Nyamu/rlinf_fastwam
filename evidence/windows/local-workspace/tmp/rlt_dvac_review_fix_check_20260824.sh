#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
python=/root/autodl-tmp/RLinf/.venv/bin/python
ruff_bin=/root/autodl-tmp/RLinf/.venv/bin/ruff

printf 'NOW=%s\n' "$(date --iso-8601=seconds)"
"$ruff_bin" format \
  "$repo/rlinf/algorithms/rlt/dvac_weighting.py" \
  "$repo/rlinf/algorithms/rlt/rollout.py" \
  "$repo/rlinf/algorithms/rlt/transition.py" \
  "$repo/rlinf/models/embodiment/openpi/openpi_action_model.py" \
  "$repo/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py" \
  "$repo/tests/unit_tests/test_rlt_dvac_weighting.py"
"$ruff_bin" check \
  "$repo/rlinf/algorithms/rlt/dvac_weighting.py" \
  "$repo/rlinf/algorithms/rlt/rollout.py" \
  "$repo/rlinf/algorithms/rlt/transition.py" \
  "$repo/rlinf/models/embodiment/openpi/openpi_action_model.py" \
  "$repo/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py" \
  "$repo/tests/unit_tests/test_rlt_dvac_weighting.py"

git -C "$repo" diff --check
"$python" - <<'PY'
from pathlib import Path

root = Path("/root/autodl-tmp/RLinf_rlt_teacher_dvac")
for relative in (
    "rlinf/algorithms/rlt/dvac_weighting.py",
    "rlinf/algorithms/rlt/rollout.py",
    "rlinf/algorithms/rlt/transition.py",
    "rlinf/models/embodiment/openpi/openpi_action_model.py",
    "rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py",
    "tests/unit_tests/test_rlt_dvac_weighting.py",
):
    path = root / relative
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print("SYNTAX_OK", relative)
PY

printf 'GIT_STATUS\n'
git -C "$repo" status --short --branch
printf 'DIFF_STAT\n'
git -C "$repo" diff --stat

printf 'ACTIVE_TRAINING\n'
run=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823
for name in wrapper driver observer; do
  pid=$(cat "$run/$name.pid" 2>/dev/null || true)
  printf '%s=%s alive=%s\n' "$name" "$pid" "$([ -n "$pid" ] && [ -d "/proc/$pid" ] && echo 1 || echo 0)"
done
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
grep -E '^oom |^oom_kill ' /sys/fs/cgroup/memory.events
