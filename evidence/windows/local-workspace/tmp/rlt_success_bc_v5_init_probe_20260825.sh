#!/usr/bin/env bash
set -u

echo "[TIME]"
date -Is

echo "[RELEVANT_PROCESSES]"
ps -eo pid,ppid,pgid,state,etime,pcpu,pmem,wchan:24,args --sort=pid | grep -E '474071|474072|RLinf|EnvWorker|Actor|Rollout|TorchInductor|ray::|gcs_server' | grep -v grep || true

echo "[CUDA_PROCESSES]"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader || true

echo "[RAY_ERRORS]"
grep -R -n -E 'Traceback|ERROR|Error|Exception|fatal|FATAL|Killed|SIG|timeout|Timeout|failed|Failed' /tmp/ray_50001/session_latest/logs 2>/dev/null | tail -n 120 || true

echo "[LATEST_WORKER_LOGS]"
find /tmp/ray_50001/session_latest/logs -maxdepth 1 -type f -name 'worker-*.out' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 12 | cut -d' ' -f2- | while IFS= read -r worker_log; do
  echo "--- ${worker_log}"
  tail -n 30 "${worker_log}" || true
done

echo "[CONTROL_TAIL]"
tail -n 100 /root/autodl-tmp/experiment_exports/rlt_single_gpu_control_success_bc_pair_smoke_20260825_v5/runtime/driver.log 2>/dev/null || true

echo "[METHOD_TAIL]"
tail -n 100 /root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_smoke_20260825_v5/runtime/driver.log 2>/dev/null || true
