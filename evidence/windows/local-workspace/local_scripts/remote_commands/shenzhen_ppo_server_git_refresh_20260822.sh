#!/usr/bin/env bash
set -uo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
CANON=/data/chenyiteng/projects/rlinf-shenzhen/RLinf
PPO=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin

section() { printf '\n===== %s =====\n' "$1"; }

section identity_time
date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds
hostname
id
uptime
cat /proc/loadavg

section gpu_and_compute_owners
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits
printf '%s\n' '-- compute applications --'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>&1
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -nu); do
  test -r "/proc/$pid/status" || continue
  printf 'pid=%s owner=%s comm=%s rss_kib=%s cgroup=' \
    "$pid" "$(stat -c '%U' "/proc/$pid" 2>/dev/null)" \
    "$(cat "/proc/$pid/comm" 2>/dev/null)" \
    "$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null)"
  awk -F: '$1=="0" {print $3}' "/proc/$pid/cgroup" 2>/dev/null
done

section ppo_live_progress
if test -r "$RUN/driver.pid"; then DRIVER=$(cat "$RUN/driver.pid"); else DRIVER=''; fi
printf 'driver_pid=%s\n' "$DRIVER"
if test -n "$DRIVER" && kill -0 "$DRIVER" 2>/dev/null; then
  printf 'driver_alive=yes\n'
  ps -p "$DRIVER" -o user=,pid=,ppid=,lstart=,etime=,stat=,%cpu=,%mem=,rss=,vsz=,comm=,args=
else
  printf 'driver_alive=no\n'
fi
printf '%s\n' '-- ray core processes --'
ps -u chenyiteng -o pid=,ppid=,stat=,%cpu=,rss=,etimes=,comm= | awk '$7 ~ /^(raylet|gcs_server|ray::)/ {print}' | sort -k5,5nr | head -n 60
if test -r "$RUN/driver.log"; then
  stat -c 'driver_log_bytes=%s mtime=%y' "$RUN/driver.log"
  printf 'fatal_count='; grep -aiEc 'Traceback|CUDA out of memory|OutOfMemory|WorkerCrashed|RayActorError|SIGKILL|Killed process|No space left|NCCL.*(error|failed)' "$RUN/driver.log" || true
  printf '%s\n' '-- latest progress markers --'
  grep -aE 'Global Step:|Generating Rollout Epochs:|Evaluating Rollout Epochs:' "$RUN/driver.log" | tail -n 30 || true
  printf '%s\n' '-- latest fatal matches --'
  grep -ainE 'Traceback|CUDA out of memory|OutOfMemory|WorkerCrashed|RayActorError|SIGKILL|Killed process|No space left|NCCL.*(error|failed)' "$RUN/driver.log" | tail -n 10 || true
fi

section tensorboard_high_value_scalars
PYTHONDONTWRITEBYTECODE=1 "$VENV/bin/python" -B - "$RUN" <<'PY'
import json
import math
import pathlib
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = pathlib.Path(sys.argv[1])
paths = sorted(run.rglob("events.out.tfevents.*"))
print("event_files=" + json.dumps([{"path": str(p), "bytes": p.stat().st_size, "mtime": p.stat().st_mtime} for p in paths]))
for path in paths:
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    tags = sorted(acc.Tags().get("scalars", []))
    print("scalar_tags=" + json.dumps(tags))
    wanted = (
        "env/success_once", "eval/success_once", "env/num_trajectories", "eval/num_trajectories",
        "train/actor/approx_kl", "train/actor/clip_fraction", "train/actor/grad_norm",
        "train/actor/policy_loss", "train/actor/total_loss", "train/actor/ratio",
        "train/critic/value_loss", "train/critic/explained_variance", "train/critic/value_clip_ratio",
        "train/advantages/mean", "train/returns/mean",
        "time/generate_rollouts", "time/actor_training", "time/sync_weights", "time/global_step",
    )
    for tag in wanted:
        if tag not in tags:
            continue
        vals = acc.Scalars(tag)
        records = [{"step": int(v.step), "value": float(v.value), "wall_time": float(v.wall_time)} for v in vals]
        finite = all(math.isfinite(v["value"]) for v in records)
        print("SCALAR=" + json.dumps({"tag": tag, "count": len(records), "finite": finite, "tail": records[-6:]}, allow_nan=False))
PY

section ppo_artifacts
du -sh "$RUN" 2>/dev/null
du -h --max-depth=2 "$RUN" 2>/dev/null | sort -h | tail -n 30
printf '%s\n' '-- checkpoints --'
find "$RUN" -type d -path '*/checkpoints/global_step_*' -printf '%T@\t%p\n' 2>/dev/null | sort -n | while IFS=$'\t' read -r _ ckpt; do
  printf 'checkpoint=%s\t' "$ckpt"
  du -sh "$ckpt" | awk '{printf "size=%s\t",$1}'
  printf 'files=%s\tdistcp=%s\tmetadata=%s\n' \
    "$(find "$ckpt" -type f | wc -l)" \
    "$(find "$ckpt" -type f -name '*.distcp' | wc -l)" \
    "$(find "$ckpt" -type f -name '.metadata' | wc -l)"
done
printf '%s\n' '-- video groups --'
for d in "$RUN"/video/train "$RUN"/video/eval; do
  test -d "$d" || continue
  printf 'video_dir=%s count=%s bytes=%s\n' "$d" \
    "$(find "$d" -type f -name '*.mp4' | wc -l)" \
    "$(find "$d" -type f -name '*.mp4' -printf '%s\n' | awk '{s+=$1} END {print s+0}')"
  find "$d" -type f -name '*.mp4' -printf '%T@\t%s\t%p\n' | sort -n | tail -n 6
done
printf '%s\n' '-- principal manifests configs logs and events --'
find "$RUN" -maxdepth 4 -type f \( -name '*.log' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' -o -name '*.txt' -o -name 'events.out.tfevents.*' -o -name '*.pid' \) \
  -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' 2>/dev/null | sort -k3,3 | head -n 160

section memory_and_envworkers
grep -E '^(MemTotal|MemAvailable|MemFree|Cached|SReclaimable|Shmem|SwapTotal|SwapFree|AnonPages|Mapped|PageTables):' /proc/meminfo
free -h
if test -n "$DRIVER" && test -r "/proc/$DRIVER/cgroup"; then
  CGREL=$(awk -F: '$1=="0" {print $3}' "/proc/$DRIVER/cgroup")
  CG=/sys/fs/cgroup${CGREL}
  printf 'driver_cgroup=%s\n' "$CG"
  for f in memory.current memory.peak memory.high memory.max memory.swap.current memory.events; do
    test -r "$CG/$f" && { printf -- '-- %s --\n' "$f"; cat "$CG/$f"; }
  done
fi
for proccomm in /proc/[0-9]*/comm; do
  comm=$(cat "$proccomm" 2>/dev/null) || continue
  test "$comm" = 'ray::EnvWorker' || continue
  pid=${proccomm#/proc/}; pid=${pid%/comm}
  test "$(stat -c '%U' "/proc/$pid" 2>/dev/null)" = chenyiteng || continue
  printf -- '--- envworker pid=%s ---\n' "$pid"
  grep -E '^(Name|State|Threads|VmPeak|VmSize|VmRSS|RssAnon|RssFile|RssShmem|VmData|VmSwap):' "/proc/$pid/status" 2>/dev/null
  if test -r "/proc/$pid/smaps_rollup"; then
    grep -E '^(Rss|Pss|Pss_Anon|Pss_File|Pss_Shmem|Private_Clean|Private_Dirty|Anonymous|AnonHugePages|Swap):' "/proc/$pid/smaps_rollup"
  fi
done

section storage_inodes_and_owned_footprint
df -hT / /home /data
df -ih / /home /data
du -sh /home/chenyiteng /home/chenyiteng/venvs /home/chenyiteng/cache /data/chenyiteng /data/chenyiteng/projects /data/chenyiteng/models /data/chenyiteng/results 2>/dev/null

section proxy_small_probe
systemctl is-active mihomo.service 2>&1
systemctl is-enabled mihomo.service 2>&1
ss -lnt | awk '$4 ~ /127\.0\.0\.1:(7890|9090)$/ {print}'
source /etc/profile.d/mihomo-proxy.sh 2>/dev/null
curl -sSIL --max-time 12 --connect-timeout 6 -o /dev/null -w 'github http=%{http_code} connect=%{time_connect} total=%{time_total}\n' https://github.com/ 2>&1
curl -sSIL --max-time 12 --connect-timeout 6 -o /dev/null -w 'huggingface http=%{http_code} connect=%{time_connect} total=%{time_total}\n' https://huggingface.co/ 2>&1

section git_and_ssh_readonly
for repo in "$CANON" "$PPO"; do
  printf -- '--- repo=%s ---\n' "$repo"
  git -C "$repo" rev-parse HEAD
  git -C "$repo" branch --show-current
  git -C "$repo" status --short --branch
  git -C "$repo" remote -v
done
printf '%s\n' '-- canonical worktree list --'
git -C "$CANON" worktree list --porcelain
printf '%s\n' '-- ~/.ssh names and modes only --'
if test -d "$HOME/.ssh"; then
  find "$HOME/.ssh" -mindepth 1 -maxdepth 1 -printf '%f\t%M\t%u:%g\t%s\n' | sort
  if test -r "$HOME/.ssh/known_hosts"; then
    printf 'known_hosts_lines=%s\n' "$(wc -l < "$HOME/.ssh/known_hosts")"
    printf 'known_hosts_plain_github_entries=%s\n' "$(grep -Ec '(^|,)github\.com([, ]|$)' "$HOME/.ssh/known_hosts" || true)"
  fi
else
  printf 'ssh_dir_absent=yes\n'
fi

section end
TZ=Asia/Shanghai date --iso-8601=seconds
exit 0
