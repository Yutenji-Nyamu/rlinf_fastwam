set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1
PACKET=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo

date '+TIME %Y-%m-%d %H:%M:%S %Z'

printf 'WRAPPER\n'
sed -n '1,220p' "$RUN/runtime/wrapper.sh" 2>/dev/null || true

printf 'OBSERVER\n'
sed -n '1,220p' "$RUN/runtime/observer.sh" 2>/dev/null || true

printf 'CONTRACT\n'
sed -n '1,220p' "$PACKET/contract.json" 2>/dev/null || true

printf 'SOURCE_HEAD_STATUS\n'
git -C "$WT" rev-parse HEAD
git -C "$WT" status --short --branch

printf 'ENV_OFFLOAD_REFERENCES\n'
grep -R -n --include='*.py' 'enable_offload' "$WT/rlinf/workers/env" "$WT/rlinf/runners" "$WT/rlinf/envs" 2>/dev/null | head -160 || true

printf 'ENV_WORKER_CONTEXT\n'
grep -n -B18 -A38 'enable_offload' "$WT/rlinf/workers/env/env_worker.py" 2>/dev/null | head -360 || true

printf 'ROBOTWIN_CLOSE_RESET_CONTEXT\n'
grep -n -B20 -A45 -E 'def (close|reset|_handle_auto_reset)|ThreadPoolExecutor|shutdown\(' \
  /data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support/robotwin/envs/vector_env.py \
  "$WT/rlinf/envs/robotwin/robotwin_env.py" 2>/dev/null | head -520 || true

printf 'CHECKPOINT_LOAD_SAVE_REFERENCES\n'
grep -R -n --include='*.py' -E 'resume_dir|load_checkpoint|checkpoint_format|dcp_checkpoint' \
  "$WT/rlinf/runners" "$WT/rlinf/workers" "$WT/rlinf/models" 2>/dev/null | head -300 || true

printf 'RESOURCE_LAST\n'
tail -n 40 "$RUN/runtime/resource.csv" 2>/dev/null || true

printf 'RAY_ENV_LOGS_FAILURE_FIRST_LAST\n'
SESSION=/data/chenyiteng/ray/rlt-dsrl-v3/session_latest/logs
for pid in 3590591 3590594; do
  find -L "$SESSION" -maxdepth 1 -type f \( -name "*${pid}*.err" -o -name "*${pid}*.out" \) -print 2>/dev/null | while read -r f; do
    printf 'FILE=%s\n' "$f"
    grep -anE 'OIDN Error|Fatal Python error|PyGILState_Release' "$f" | head -8 || true
    grep -anE 'OIDN Error|Fatal Python error|PyGILState_Release' "$f" | tail -8 || true
  done
done
