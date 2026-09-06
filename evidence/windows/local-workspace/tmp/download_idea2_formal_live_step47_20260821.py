from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


WORKSPACE = Path(r"C:\Users\86136\Documents\rl")
sys.path.insert(0, str(WORKSPACE))

from local_scripts.remote_exec_autodl import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_HOST_KEY_SHA256,
    DEFAULT_PORT,
    DEFAULT_USER,
    connect,
)


TARGET = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/"
    "formal_live_step47_20260821"
)
REMOTE_RUN = (
    "/root/autodl-tmp/idea2_dvac_train_runs/"
    "idea2_dvac_apply_formal_100step_2gpu16env_20260821"
)
REMOTE_RUNTIME = (
    "/root/autodl-tmp/idea2_dvac_train_runtime/"
    "idea2_dvac_apply_formal_100step_2gpu16env_20260821"
)


AUDIT_COMMAND = rf"""set -u
run={REMOTE_RUN}
runtime={REMOTE_RUNTIME}

echo '=== IDENTITY_TIME ==='
date -Is
hostname
pwd
id -u

echo '=== CONTROLLERS ==='
for name in wrapper driver observer; do
  pid_file="$runtime/$name.pid"
  if test -f "$pid_file"; then
    pid=$(cat "$pid_file")
    printf '%s_pid=%s\n' "$name" "$pid"
    ps -o pid,ppid,stat,rss,etimes,args -p "$pid" || true
  else
    printf '%s_pid_file=missing\n' "$name"
  fi
done
for file in driver.exitcode observer.exitcode launch_started_at.txt launch_finished_at.txt; do
  if test -f "$runtime/$file"; then
    printf '%s=' "$file"
    cat "$runtime/$file"
  else
    printf '%s=missing\n' "$file"
  fi
done

echo '=== LATEST_GLOBAL_STEPS ==='
grep -o 'Global Step:[[:space:]]*[0-9]\+/100' "$run/metrics.log" | tail -n 5 || true
echo '=== LATEST_METRICS_TAIL ==='
tail -n 75 "$run/metrics.log" || true

echo '=== DVAC_RANK_TAILS ==='
for file in "$run"/dvac_train/actor_rank*/runner_step_metrics.csv; do
  echo "--- $file"
  wc -l -c "$file"
  tail -n 4 "$file"
done
echo '=== DVAC_NPZ_COUNTS ==='
for dir in "$run"/dvac_train/actor_rank*; do
  printf '%s ' "$dir"
  find "$dir" -maxdepth 1 -type f -name 'rollout_step*.npz' -printf '%f\n' | sort | tail -n 3
  find "$dir" -maxdepth 1 -type f -name 'rollout_step*.npz' -printf '.' | wc -c
done

echo '=== CHECKPOINTS ==='
find "$run" -type d -name 'global_step_*' -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort || true
find "$run" -type d -name 'global_step_*' -exec du -sh {{}} \; 2>/dev/null | sort -V || true

echo '=== GPU ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,name --format=csv,noheader || true

echo '=== CGROUP_RAM ==='
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
printf 'memory.peak='; cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo unavailable
printf 'memory.high='; cat /sys/fs/cgroup/memory.high
printf 'memory.max='; cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
awk '/MemTotal|MemAvailable/ {{print}}' /proc/meminfo

echo '=== DISK_AND_RUN_SIZE ==='
df -h /root/autodl-tmp /dev/shm
du -sh "$run" "$runtime" 2>/dev/null || true

echo '=== RESOURCE_MONITOR_SUMMARY ==='
res="$runtime/resource_monitor/resources.csv"
if test -f "$res"; then
  wc -l -c "$res"
  head -n 2 "$res"
  tail -n 5 "$res"
fi
rss="$runtime/resource_monitor/process_rss.tsv"
if test -f "$rss"; then
  wc -l -c "$rss"
  tail -n 3 "$rss"
fi

echo '=== DRIVER_TAIL_AND_ERRORS ==='
tail -n 80 "$runtime/driver.log" 2>/dev/null || true
echo '--- error scan'
grep -E 'CUDA out of memory|OutOfMemory|worker died|WorkerCrashed|NCCL.*(error|Error)|oom_kill|RayTaskError|Traceback|FATAL|Fatal' "$runtime/driver.log" 2>/dev/null | tail -n 40 || true
"""


def exec_capture(client, command: str) -> tuple[int, bytes, bytes]:
    stdin, stdout, stderr = client.exec_command(command)
    stdin.channel.shutdown_write()
    out = stdout.read()
    err = stderr.read()
    return stdout.channel.recv_exit_status(), out, err


def get_file(sftp, remote: str, relative: str) -> int:
    local = TARGET / relative
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote, str(local))
    return local.stat().st_size


args = argparse.Namespace(
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    user=DEFAULT_USER,
    host_key_sha256=DEFAULT_HOST_KEY_SHA256,
    timeout=20.0,
)

client = connect(args)
try:
    rc, out, err = exec_capture(client, AUDIT_COMMAND)
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "LIVE_READONLY_AUDIT.txt").write_bytes(out)
    (TARGET / "LIVE_READONLY_AUDIT.stderr.txt").write_bytes(err)
    if rc != 0:
        raise SystemExit(f"remote audit failed with exit {rc}")

    matches = re.findall(rb"Global Step:\s*(\d+)/100", out)
    if not matches:
        raise RuntimeError("No completed global step found")
    latest_global_step = max(int(value) for value in matches)
    latest_runner_step = latest_global_step - 1

    files = [
        (f"{REMOTE_RUN}/metrics.log", "run/metrics.log"),
        (
            f"{REMOTE_RUNTIME}/resource_monitor/resources.csv",
            "runtime/resources.csv",
        ),
        (
            f"{REMOTE_RUNTIME}/resource_monitor/process_rss.tsv",
            "runtime/process_rss.tsv",
        ),
        (f"{REMOTE_RUNTIME}/driver.log", "runtime/driver.log"),
        (f"{REMOTE_RUNTIME}/launch_started_at.txt", "runtime/launch_started_at.txt"),
    ]
    for rank in (0, 1):
        remote_rank = f"{REMOTE_RUN}/dvac_train/actor_rank{rank:02d}"
        local_rank = f"run/dvac_train/actor_rank{rank:02d}"
        files.extend(
            [
                (
                    f"{remote_rank}/runner_step_metrics.csv",
                    f"{local_rank}/runner_step_metrics.csv",
                ),
                (
                    f"{remote_rank}/rolling_stats_state.json",
                    f"{local_rank}/rolling_stats_state.json",
                ),
                (
                    f"{remote_rank}/run_manifest.json",
                    f"{local_rank}/run_manifest.json",
                ),
                (
                    f"{remote_rank}/rollout_step{latest_runner_step:04d}.npz",
                    f"{local_rank}/rollout_step{latest_runner_step:04d}.npz",
                ),
            ]
        )

    count = 0
    total = len(out) + len(err)
    with client.open_sftp() as sftp:
        for remote, relative in files:
            total += get_file(sftp, remote, relative)
            count += 1
finally:
    client.close()

print(f"LATEST_GLOBAL_STEP={latest_global_step}")
print(f"LATEST_RUNNER_STEP={latest_runner_step}")
print(f"DOWNLOADED_FILES={count}")
print(f"TOTAL_LOCAL_BYTES={total}")
print(f"TARGET={TARGET}")
