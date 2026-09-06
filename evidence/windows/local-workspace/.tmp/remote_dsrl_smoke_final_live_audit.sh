set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN_ROOT="$REPO/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1"
EXP="$RUN_ROOT/robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke"
OUT="$RUN_ROOT/post_smoke_live_audit.txt"

test -s "$RUN_ROOT/base_freeze_validation.exit"
test "$(cat "$RUN_ROOT/base_freeze_validation.exit")" = 0
test -s "$RUN_ROOT/ckpt1_verify_after_resume.exit"
test "$(cat "$RUN_ROOT/ckpt1_verify_after_resume.exit")" = 0
test ! -e "$OUT"

{
  echo "POST_SMOKE_AUDIT_TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "IDENTITY=$(id -un)@$(hostname)"
  echo "BRANCH=$(git -C "$REPO" branch --show-current)"
  echo "HEAD=$(git -C "$REPO" rev-parse HEAD)"
  echo "UPSTREAM=$(git -C "$REPO" rev-parse '@{upstream}')"
  if test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"; then
    echo "WORKTREE_CLEAN=1"
  else
    echo "WORKTREE_CLEAN=0"
    git -C "$REPO" status --short
  fi

  echo "OWN_PID_STATUS"
  for name in fresh resume; do
    driver_pid=$(cat "$RUN_ROOT/$name.pid")
    monitor_pid=$(cat "$RUN_ROOT/resource_monitor/$name/monitor.pid")
    if kill -0 "$driver_pid" 2>/dev/null; then
      driver_alive=1
    else
      driver_alive=0
    fi
    if kill -0 "$monitor_pid" 2>/dev/null; then
      monitor_alive=1
    else
      monitor_alive=0
    fi
    echo "$name driver_pid=$driver_pid driver_alive=$driver_alive monitor_pid=$monitor_pid monitor_alive=$monitor_alive"
  done

  echo "TARGET_PROCESS_SCAN"
  pgrep -af 'train_embodied_agent.py|monitor_resources.py|raylet|gcs_server|EmbodiedSACFSDPPolicy|MultiStepRolloutWorker|EnvWorker' || true

  echo "GPU_STATE"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  echo "GPU_COMPUTE_PROCESSES"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true

  echo "CGROUP_CURRENT"
  printf "memory.current="
  cat /sys/fs/cgroup/memory.current
  printf "memory.max="
  cat /sys/fs/cgroup/memory.max
  cat /sys/fs/cgroup/memory.events
  grep -E '^(anon|file|shmem|file_mapped|inactive_file|active_file|kernel_stack|pagetables|slab) ' /sys/fs/cgroup/memory.stat

  echo "FRESH_MEMORY_EVENTS_EARLY"
  cat "$RUN_ROOT/resource_monitor/fresh/memory.events.early.txt"
  echo "FRESH_MEMORY_EVENTS_FINAL"
  cat "$RUN_ROOT/resource_monitor/fresh/memory.events.final.txt"
  echo "RESUME_MEMORY_EVENTS_BASELINE"
  cat "$RUN_ROOT/resource_monitor/resume/memory.events.baseline.txt"
  echo "RESUME_MEMORY_EVENTS_FINAL"
  cat "$RUN_ROOT/resource_monitor/resume/memory.events.final.txt"

  echo "FILESYSTEM_STATE"
  df -h /root/autodl-tmp /dev/shm
  du -sh "$RUN_ROOT" "$EXP/checkpoints/global_step_1" "$EXP/checkpoints/global_step_2"

  echo "POST_RUN_RELATIVE_DATA_SCAN"
  if test -d /root/data; then
    find /root/data -maxdepth 5 -type f -newermt '2026-07-28 17:07:00' -print
  else
    echo "/root/data absent"
  fi

  echo "POSTCHECK_STATUS"
  echo "base_freeze_exit=$(cat "$RUN_ROOT/base_freeze_validation.exit")"
  echo "ckpt1_verify_exit=$(cat "$RUN_ROOT/ckpt1_verify_after_resume.exit")"
  echo "ckpt1_verify_ok_files=$(grep -c ': OK$' "$RUN_ROOT/ckpt1_verify_after_resume.log")"
  echo "POST_SMOKE_LIVE_AUDIT_OK=1"
} | tee "$OUT"
