set -euo pipefail

CANON=/data/chenyiteng/projects/rlinf-shenzhen/RLinf
RLT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
TARGET=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-dvac-pure-single-gpu-7d07a421
BRANCH=codex/sz-rlt-dvac-pure-single-gpu

date '+time=%F %T %Z'
hostname
id
printf 'mem_available_kib='
awk '/MemAvailable:/ {print $2}' /proc/meminfo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
df -h / /home /data

printf 'canonical_head='
git -C "$CANON" rev-parse HEAD
printf 'rlt_head='
git -C "$RLT" rev-parse HEAD
printf 'rlt_branch='
git -C "$RLT" branch --show-current
git -C "$RLT" status --short --branch
git -C "$RLT" remote -v
git -C "$CANON" worktree list --porcelain

if git -C "$CANON" show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo 'target_branch_exists=yes'
else
  echo 'target_branch_exists=no'
fi
if [ -e "$TARGET" ]; then
  echo 'target_path_exists=yes'
else
  echo 'target_path_exists=no'
fi

ps -eo user,pid,ppid,pgid,etimes,rss,cmd --sort=pid | grep -E 'gcs_server|raylet|formal100.*(action|st)|1413907|1416016' | grep -v grep || true

if command -v ray >/dev/null 2>&1; then
  ray list actors --address 172.17.0.1:6389 --format json 2>/dev/null \
    | python -c 'import json,sys,collections; x=json.load(sys.stdin); print("actor_namespaces="+repr(dict(collections.Counter(a.get("ray_namespace") for a in x if a.get("state") == "ALIVE"))))' \
    || true
fi
