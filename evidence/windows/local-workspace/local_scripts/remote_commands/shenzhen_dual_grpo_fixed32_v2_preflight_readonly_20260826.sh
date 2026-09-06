#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo
HEAD=0e28ac6f09f821ea12e7d54eba7118ce0000ca86
CONTROL=grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2
DVAC=dvac-global-z-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v2
RAY_ADDRESS=172.17.0.1:6389

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --short)"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null
test ! -e "$ROOT/runs/$CONTROL"
test ! -e "$ROOT/runs/$DVAC"
test ! -e "$ROOT/packets/$CONTROL"
test ! -e "$ROOT/packets/$DVAC"
if nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo GPU_4_7_NOT_IDLE
  exit 20
fi
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_fixed32_v2_preflight", logging_level="ERROR")
rows = ray.util.list_named_actors(all_namespaces=True)
owned = [r for r in rows if r.get("namespace") in {"RLinf", "RLinf_1"}]
print(f"old_namespace_actor_count={len(owned)}")
assert not owned, owned
ray.shutdown()
PY
TZ=Asia/Shanghai date --iso-8601=seconds
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -h | sed -n '1,3p'
echo SZ_DUAL_GRPO_FIXED32_V2_PREFLIGHT_OK
