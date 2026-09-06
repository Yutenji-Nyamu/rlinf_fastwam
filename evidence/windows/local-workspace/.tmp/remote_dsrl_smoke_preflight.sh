set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
ROBOTWIN=/root/autodl-tmp/RoboTwin_RLinf
PY=/root/autodl-tmp/RLinf/.venv/bin/python
CFG=robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke
RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
MODEL=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle

echo "== identity/time =="
date --iso-8601=seconds
id
hostname

echo "== git =="
git -C "$REPO" rev-parse --show-toplevel
git -C "$REPO" rev-parse --git-dir
git -C "$REPO" branch --show-current
git -C "$REPO" rev-parse HEAD
git -C "$REPO" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'
git -C "$REPO" status --short --branch
git -C "$REPO" remote -v
git -C "$REPO" ls-remote personal refs/heads/codex/dsrl-pi0-robotwin

echo "== required paths =="
test -x "$PY"
test -f "$REPO/examples/embodiment/train_embodied_agent.py"
test -f "$REPO/examples/embodiment/monitor_resources.py"
test -f "$REPO/examples/embodiment/config/${CFG}.yaml"
test -d "$ROBOTWIN"
test -d "$MODEL"
test -f "$REPO/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FRESH_SMOKE_VALIDATED_RESOLVED_20260728.yaml"
test -f "$REPO/docs/rlinf-robotwin-pi0-traditional-rl/evidence/RESUME_SMOKE_VALIDATED_RESOLVED_20260728.yaml"
sha256sum \
  "$REPO/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FRESH_SMOKE_VALIDATED_RESOLVED_20260728.yaml" \
  "$REPO/docs/rlinf-robotwin-pi0-traditional-rl/evidence/RESUME_SMOKE_VALIDATED_RESOLVED_20260728.yaml"
if test -e "$RUN_ROOT"; then
  echo "RUN_ROOT_EXISTS=$RUN_ROOT"
  find "$RUN_ROOT" -maxdepth 2 -mindepth 1 -printf '%y %p\n' | head -n 40
  exit 31
else
  echo "RUN_ROOT_ABSENT=1"
fi

echo "== runtime =="
export PYTHONDONTWRITEBYTECODE=1
"$PY" -B -c 'import sys, torch, ray, hydra; print(sys.version); print("torch", torch.__version__, "cuda", torch.version.cuda, "ray", ray.__version__, "hydra", hydra.__version__)'

echo "== processes =="
ps -eo pid=,ppid=,comm=,lstart=,args= --sort=pid |
  awk '$3 ~ /^(python|python3|raylet|gcs_server)$/ && $0 ~ /(train_embodied_agent|ray::|raylet|gcs_server|monitor_resources.py)/ {print}'

echo "== gpu =="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits

echo "== cgroup memory =="
printf 'memory.current='
cat /sys/fs/cgroup/memory.current
printf 'memory.max='
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
awk '$1 ~ /^(anon|file|shmem|file_mapped|inactive_file|active_file|kernel_stack|pagetables|slab)$/ {print}' /sys/fs/cgroup/memory.stat

echo "== disk =="
df -h /root/autodl-tmp
du -xhd1 /root/autodl-tmp 2>/dev/null | sort -h

echo "PREFLIGHT_OK=1"
