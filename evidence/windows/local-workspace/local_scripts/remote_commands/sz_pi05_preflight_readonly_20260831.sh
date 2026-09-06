set -u

echo '=== identity/time ==='
id
date -Is
hostname
uptime

echo '=== memory/disks ==='
free -h
df -h / /home /data

echo '=== gpu summary ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader

echo '=== gpu processes ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader || true
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -nu); do
  ps -o user=,pid=,ppid=,etimes=,rss=,args= -p "$pid" || true
done

echo '=== rlinf/ray processes ==='
pgrep -af 'run_embodiment|gcs_server|raylet|RLinf|formal|smoke' | tail -n 100 || true
ss -lnt 2>/dev/null | grep ':6389' || true

echo '=== candidate runtime paths ==='
for p in \
  /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin \
  /data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-dvac-action-adv-fix \
  /data/chenyiteng/models/rlinf; do
  if [ -e "$p" ]; then
    du -sh "$p" 2>/dev/null || true
    stat -c '%A %U:%G %n' "$p" || true
  else
    echo "MISSING $p"
  fi
done

echo '=== existing pi05 models ==='
find /data/chenyiteng/models -maxdepth 4 -iname '*pi05*' -printf '%y %p\n' 2>/dev/null | sort | head -n 100 || true

echo '=== relevant git state ==='
repo=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-dvac-action-adv-fix
if [ -d "$repo/.git" ] || [ -f "$repo/.git" ]; then
  git -C "$repo" rev-parse HEAD
  git -C "$repo" status --short
  git -C "$repo" branch --show-current
  git -C "$repo" remote -v | head -n 8
fi

echo '=== recent resolved configs ==='
find /data/chenyiteng/results/rlinf-shenzhen -type f -name resolved.yaml -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | head -n 20 || true
