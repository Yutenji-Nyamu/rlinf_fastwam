#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
python=/root/autodl-tmp/RLinf/.venv/bin/python

printf 'NOW=%s\n' "$(date --iso-8601=seconds)"
printf 'GIT_STATUS\n'
git -C "$repo" status --short --branch
printf 'DIFF_STAT\n'
git -C "$repo" diff --stat
printf 'DIFF_CHECK\n'
git -C "$repo" diff --check

printf 'PYTHON_SYNTAX\n'
"$python" - <<'PY'
from pathlib import Path

root = Path("/root/autodl-tmp/RLinf_rlt_teacher_dvac")
paths = (
    "rlinf/algorithms/rlt/dvac_weighting.py",
    "rlinf/algorithms/rlt/rollout.py",
    "rlinf/algorithms/rlt/transition.py",
    "rlinf/models/embodiment/openpi/openpi_action_model.py",
    "rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py",
    "tests/unit_tests/test_rlt_dvac_weighting.py",
)
for relative in paths:
    path = root / relative
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print(f"SYNTAX_OK {relative}")
PY

printf 'YAML_PARSE\n'
"$python" - <<'PY'
from pathlib import Path
import yaml

path = Path("/root/autodl-tmp/RLinf_rlt_teacher_dvac/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250_teacher_dvac_w0to2.yaml")
payload = yaml.safe_load(path.read_text(encoding="utf-8"))
assert payload["algorithm"]["rlt_dvac"]["strength"] == 0.5
assert payload["algorithm"]["rlt_dvac"]["z_clip"] == 2.0
print("YAML_OK", path.name)
PY

if command -v ruff >/dev/null 2>&1; then
  printf 'RUFF\n'
  ruff check \
    "$repo/rlinf/algorithms/rlt/dvac_weighting.py" \
    "$repo/rlinf/algorithms/rlt/rollout.py" \
    "$repo/rlinf/algorithms/rlt/transition.py" \
    "$repo/rlinf/models/embodiment/openpi/openpi_action_model.py" \
    "$repo/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py" \
    "$repo/tests/unit_tests/test_rlt_dvac_weighting.py"
else
  printf 'RUFF_NOT_INSTALLED\n'
fi

printf 'RUNNING_TRAINING\n'
for name in wrapper driver observer; do
  pid=$(cat "/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823/$name.pid" 2>/dev/null || true)
  printf '%s=%s alive=%s\n' "$name" "$pid" "$([ -n "$pid" ] && [ -d "/proc/$pid" ] && echo 1 || echo 0)"
done
printf 'RESOURCE\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
grep -E '^oom |^oom_kill ' /sys/fs/cgroup/memory.events

