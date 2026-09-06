#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
DRIVER_PID="$(cat "$RUN/driver.pid")"

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
printf 'driver_pid=%s alive=' "$DRIVER_PID"
if kill -0 "$DRIVER_PID" 2>/dev/null; then printf 'yes\n'; else printf 'no\n'; fi

printf '%s\n' '=== memory ==='
free -h
grep -E '^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SReclaimable|Shmem|SwapTotal|SwapFree|CommitLimit|Committed_AS):' /proc/meminfo
vmstat 1 3

printf '%s\n' '=== relevant process RSS ==='
ps -u chenyiteng -o pid=,ppid=,pgid=,rss=,vsz=,stat=,comm=,args= --sort=-rss | sed -n '1,45p'
ps -u chenyiteng -o rss= | awk '{s+=$1} END {printf "chenyiteng_total_rss_kib=%d\n", s}'

printf '%s\n' '=== gpu ==='
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits

printf '%s\n' '=== artifacts ==='
du -sh "$RUN"
find "$RUN" -maxdepth 6 -type f -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' | sort -k2,2 | tail -n 120
printf 'checkpoint_dirs='; find "$RUN" -path '*/checkpoints/global_step_*' -type d -printf '%f\n' | sort -Vu | tr '\n' ' '; printf '\n'
printf 'checkpoint_bytes='; du -sb "$RUN/robotwin_ppo_openpi/checkpoints" 2>/dev/null | awk '{print $1}' || true
printf 'video_count='; find "$RUN/video" -type f -name '*.mp4' 2>/dev/null | wc -l
printf 'video_bytes='; find "$RUN/video" -type f -name '*.mp4' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}'

printf '%s\n' '=== tensorboard scalars json ==='
"$VENV/bin/python" - "$RUN" <<'PY'
import json
import pathlib
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = pathlib.Path(sys.argv[1])
event_files = sorted(run.rglob('events.out.tfevents.*'))
print(json.dumps({'event_files': [str(p) for p in event_files]}))
for path in event_files:
    ea = EventAccumulator(str(path), size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    payload = {}
    for tag in tags:
        payload[tag] = [
            {'step': int(e.step), 'wall_time': float(e.wall_time), 'value': float(e.value)}
            for e in ea.Scalars(tag)
        ]
    print(json.dumps({'event_file': str(path), 'scalars': payload}, allow_nan=True, sort_keys=True))
PY

printf '%s\n' '=== fatal scan and latest progress ==='
grep -nE 'Traceback|CUDA out of memory|OutOfMemory|WorkerCrashed|RayActorError|SIGKILL|Killed process|No space left' "$RUN/driver.log" | tail -n 50 || true
grep -E 'Global Step:|Generating Rollout Epochs:|Evaluating Rollout Epochs:' "$RUN/driver.log" | tail -n 35 || true
