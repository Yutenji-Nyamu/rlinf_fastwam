#!/usr/bin/env bash
set -uo pipefail

# One-shot, read-only formal PPO snapshot. No sleeps, loops over time, network
# probes, process signals, checkpoint reads, or server-side file writes.

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
LOG="$RUN/driver.log"
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
CKPT="$RUN/robotwin_ppo_openpi/checkpoints"
LAUNCH_WRAPPER=shenzhen_rlinf_launch_ppo_formal100_4gpu128train64eval_v1.sh
USER_UID="$(id -u)"

strip_ansi() {
  sed -E 's/\x1B\[[0-9;?]*[ -\/]*[@-~]//g'
}

printf '%s\n' '=== SNAPSHOT_IDENTITY ==='
printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
printf 'hostname=%s uid=%s run=%s\n' "$(hostname)" "$USER_UID" "$RUN"

printf '%s\n' '=== PPO_WRAPPER_DRIVER_RAY ==='
mapfile -t wrapper_rows < <(
  pgrep -u "$USER_UID" -af "$LAUNCH_WRAPPER" 2>/dev/null || true
)
printf 'wrapper_pattern=%s wrapper_match_count=%s wrapper_alive=%s\n' \
  "$LAUNCH_WRAPPER" "${#wrapper_rows[@]}" \
  "$(if test "${#wrapper_rows[@]}" -gt 0; then printf yes; else printf no; fi)"
printf '%s\n' "${wrapper_rows[@]}"

driver_pid="$(cat "$RUN/driver.pid" 2>/dev/null || true)"
driver_alive=no
driver_identity_match=no
driver_cmdline=''
if test -n "$driver_pid" && test -r "/proc/$driver_pid/cmdline"; then
  driver_alive=yes
  driver_cmdline="$(tr '\0' ' ' < "/proc/$driver_pid/cmdline")"
  if printf '%s\n' "$driver_cmdline" | grep -Fq 'train_embodied_agent.py' \
    && printf '%s\n' "$driver_cmdline" | grep -Fq "$RUN"; then
    driver_identity_match=yes
  fi
fi
printf 'driver_pid=%s driver_alive=%s driver_identity_match=%s\n' \
  "${driver_pid:-missing}" "$driver_alive" "$driver_identity_match"
if test "$driver_alive" = yes; then
  ps -p "$driver_pid" -o user=,pid=,ppid=,pgid=,sid=,stat=,etimes=,%cpu=,%mem=,rss=,vsz=,comm=,args=
fi

mapfile -t raylet_rows < <(pgrep -u "$USER_UID" -a -x raylet 2>/dev/null || true)
mapfile -t gcs_rows < <(pgrep -u "$USER_UID" -a -x gcs_server 2>/dev/null || true)
printf 'raylet_count=%s gcs_server_count=%s ray_core_alive=%s\n' \
  "${#raylet_rows[@]}" "${#gcs_rows[@]}" \
  "$(if test "${#raylet_rows[@]}" -gt 0 && test "${#gcs_rows[@]}" -gt 0; then printf yes; else printf no; fi)"
printf '%s\n' "${raylet_rows[@]}" "${gcs_rows[@]}"
ps -u "$USER_UID" -o comm= | awk '
  /^ray::/ {count[$1]++}
  END {for (name in count) printf "ray_role=%s count=%d\n", name, count[name]}
' | LC_ALL=C sort

printf '%s\n' '=== LATEST_COMPLETE_STEP_AND_PHASE ==='
if test -r "$LOG"; then
  latest_complete_line="$(
    grep -aE 'Global Step:' "$LOG" | tail -n 1 | strip_ansi | tr -d '\r' || true
  )"
  latest_complete_step="$(
    printf '%s\n' "$latest_complete_line" \
      | sed -nE 's/.*Global Step:[[:space:]]*([0-9]+)\/100.*/\1/p'
  )"
  latest_phase_line="$(
    grep -aE 'Generating Rollout Epochs:|Evaluating Rollout Epochs:' "$LOG" \
      | tail -n 1 | strip_ansi | tr -d '\r' || true
  )"
  case "$latest_phase_line" in
    *'Generating Rollout Epochs:'*) latest_phase_marker=train_rollout ;;
    *'Evaluating Rollout Epochs:'*) latest_phase_marker=eval_rollout ;;
    *) latest_phase_marker=no_rollout_marker ;;
  esac
  printf 'latest_complete_step=%s\n' "${latest_complete_step:-unknown}"
  printf 'latest_complete_step_line=%s\n' "${latest_complete_line:-none}"
  printf 'latest_phase_marker=%s\n' "$latest_phase_marker"
  printf 'latest_phase_line=%s\n' "${latest_phase_line:-none}"
  printf '%s\n' '-- recent progress markers --'
  grep -aE 'Global Step:|Generating Rollout Epochs:|Evaluating Rollout Epochs:' "$LOG" \
    | tail -n 18 | strip_ansi || true
  stat -c 'driver_log_bytes=%s driver_log_mtime=%y' "$LOG"
else
  printf 'driver_log_missing=%s\n' "$LOG"
fi

printf '%s\n' '=== LATEST_KEY_SCALARS ==='
if test -x "$VENV/bin/python" && test -d "$RUN"; then
  TF_CPP_MIN_LOG_LEVEL=3 PYTHONDONTWRITEBYTECODE=1 \
    "$VENV/bin/python" -B - "$RUN" <<'PY'
import json
import math
import pathlib
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = pathlib.Path(sys.argv[1])
wanted = (
    "env/success_once", "env/num_trajectories", "env/return", "env/reward",
    "eval/success_once", "eval/success_at_end", "eval/num_trajectories",
    "rollout/advantages_mean", "rollout/returns_mean",
    "train/advantages/mean", "train/returns/mean",
    "train/actor/approx_kl", "train/actor/clip_fraction",
    "train/actor/grad_norm", "train/actor/policy_loss",
    "train/actor/total_loss", "train/critic/value_loss",
    "train/critic/explained_variance", "train/critic/value_clip_ratio",
    "time/generate_rollouts", "time/actor_training", "time/sync_weights",
    "time/step", "time/global_step",
)
latest = {}
event_files = sorted(run.rglob("events.out.tfevents.*"))
errors = []
for event_file in event_files:
    try:
        acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        acc.Reload()
        available = set(acc.Tags().get("scalars", []))
        for tag in wanted:
            if tag not in available:
                continue
            for point in acc.Scalars(tag):
                candidate = (int(point.step), float(point.wall_time), float(point.value))
                if tag not in latest or candidate[:2] > latest[tag][:2]:
                    latest[tag] = candidate
    except Exception as exc:
        errors.append({"path": str(event_file), "error": repr(exc)})

payload = {
    tag: {
        "step": step,
        "value": value if math.isfinite(value) else repr(value),
        "wall_time": wall_time,
        "finite": math.isfinite(value),
    }
    for tag, (step, wall_time, value) in sorted(latest.items())
}
print("event_file_count=" + str(len(event_files)))
print("latest_scalars_json=" + json.dumps(payload, sort_keys=True, allow_nan=False))
missing = [tag for tag in wanted if tag not in latest]
print("missing_scalar_tags_json=" + json.dumps(missing))
if errors:
    print("event_read_errors_json=" + json.dumps(errors, sort_keys=True))
PY
else
  printf 'scalar_reader_unavailable=yes\n'
fi

printf '%s\n' '=== CGROUP_AND_HOST_MEMORY ==='
anchor_pid=''
if test "$driver_identity_match" = yes; then
  anchor_pid="$driver_pid"
elif test "${#raylet_rows[@]}" -gt 0; then
  anchor_pid="$(printf '%s\n' "${raylet_rows[0]}" | awk '{print $1}')"
fi
printf 'cgroup_anchor_pid=%s\n' "${anchor_pid:-none}"
if test -n "$anchor_pid" && test -r "/proc/$anchor_pid/cgroup"; then
  cgroup_relative="$(awk -F: '$1=="0" {print $3}' "/proc/$anchor_pid/cgroup")"
  cgroup_root="/sys/fs/cgroup${cgroup_relative}"
  printf 'cgroup_path=%s\n' "$cgroup_root"
  if test -r "$cgroup_root/memory.current"; then
    printf 'cgroup_memory_current_bytes='; cat "$cgroup_root/memory.current"
  fi
  if test -r "$cgroup_root/memory.events"; then
    awk '{printf "cgroup_memory_event_%s=%s\n", $1, $2}' "$cgroup_root/memory.events"
  fi
else
  printf 'cgroup_unavailable=yes\n'
fi
awk '
  /^MemAvailable:/ {printf "mem_available_kib=%s\n", $2}
  /^SwapTotal:/ {printf "swap_total_kib=%s\n", $2}
  /^SwapFree:/ {printf "swap_free_kib=%s\n", $2}
' /proc/meminfo

printf '%s\n' '=== FOUR_ENVWORKER_SMAPS_ROLLUP ==='
mapfile -t envworker_pids < <(
  ps -u "$USER_UID" -o pid=,comm= | awk '$2=="ray::EnvWorker" {print $1}' | sort -n
)
printf 'envworker_expected=4 envworker_observed=%s\n' "${#envworker_pids[@]}"
for pid in "${envworker_pids[@]}"; do
  if test -r "/proc/$pid/smaps_rollup"; then
    awk -v pid="$pid" '
      /^Pss:/ {pss=$2}
      /^Pss_Anon:/ {pss_anon=$2}
      /^Private_Dirty:/ {private_dirty=$2}
      END {
        printf "envworker_pid=%s pss_kib=%s pss_anon_kib=%s private_dirty_kib=%s\n",
          pid, pss+0, pss_anon+0, private_dirty+0
      }
    ' "/proc/$pid/smaps_rollup" || printf 'envworker_pid=%s smaps_read_failed=yes\n' "$pid"
  else
    printf 'envworker_pid=%s smaps_unavailable=yes\n' "$pid"
  fi
done

printf '%s\n' '=== GPU_0_7_MEMORY_AND_PROCESSES ==='
nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits
printf '%s\n' '-- compute applications --'
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits 2>&1 || true
printf '%s\n' '-- compute process owners --'
mapfile -t gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | grep -E '^[0-9]+$' | sort -nu || true
)
for pid in "${gpu_pids[@]}"; do
  if test -r "/proc/$pid/status"; then
    ps -p "$pid" -o user=,pid=,ppid=,stat=,rss=,comm=,args=
  else
    printf 'gpu_process_pid=%s proc_unavailable=yes\n' "$pid"
  fi
done

printf '%s\n' '=== CHECKPOINT_LIST ==='
if test -d "$CKPT"; then
  while IFS= read -r -d '' checkpoint; do
    bytes="$(du -sb "$checkpoint" 2>/dev/null | awk '{print $1}')"
    metadata_count="$(find "$checkpoint" -type f -name '.metadata' | wc -l)"
    full_weights_count="$(find "$checkpoint" -type f -name 'full_weights.pt' | wc -l)"
    printf 'checkpoint=%s bytes=%s metadata_count=%s full_weights_count=%s\n' \
      "$(basename "$checkpoint")" "${bytes:-unknown}" "$metadata_count" "$full_weights_count"
  done < <(
    find "$CKPT" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -print0 \
      | sort -zV
  )
else
  printf 'checkpoint_root_missing=%s\n' "$CKPT"
fi

printf '%s\n' '=== FATAL_TAIL ==='
FATAL_PATTERN='Traceback|CUDA out of memory|OutOfMemory|WorkerCrashed|RayActorError|SIGKILL|Killed process|No space left|Disk quota exceeded|Segmentation fault|illegal instruction|NCCL.*(error|fail|timeout)'
if test -r "$LOG"; then
  fatal_count="$(grep -aEic "$FATAL_PATTERN" "$LOG" || true)"
  printf 'fatal_match_count=%s\n' "${fatal_count:-0}"
  grep -anEi "$FATAL_PATTERN" "$LOG" | tail -n 25 | strip_ansi || true
else
  printf 'fatal_scan_unavailable=yes\n'
fi

printf '%s\n' '=== FILESYSTEMS_ROOT_HOME_DATA ==='
df -B1 --output=target,source,fstype,size,used,avail,pcent / /home /data
printf 'timestamp_end=%s\n' "$(date --iso-8601=seconds)"
printf '%s\n' 'FORMAL100_POST_DOWNLOAD_SINGLE_SNAPSHOT_DONE'
