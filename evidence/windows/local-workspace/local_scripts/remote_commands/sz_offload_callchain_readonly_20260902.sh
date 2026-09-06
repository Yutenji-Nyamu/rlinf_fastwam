set -u

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
SUPPORT=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support

date '+TIME %Y-%m-%d %H:%M:%S %Z'

printf 'RLINF_CALL_SITES\n'
grep -RIn --include='*.py' -E '\.offload\(|enable_offload|create_env|init_env|env_group' "$WT/rlinf/workers" "$WT/rlinf/runners" "$WT/rlinf/schedulers" 2>/dev/null | head -900 || true

printf 'ENV_WORKER_FILES\n'
find "$WT/rlinf" -type f \( -name '*env*worker*.py' -o -name '*embodied*runner*.py' -o -name '*rollout*.py' \) | sort | head -200

printf 'ENV_WORKER_CONTEXT\n'
for f in $(grep -Rl --include='*.py' -E 'enable_offload.*env|env.*enable_offload|\.offload\(' "$WT/rlinf/workers" "$WT/rlinf/runners" "$WT/rlinf/schedulers" 2>/dev/null | sort -u); do
  echo "FILE $f"
  grep -n -B25 -A55 -E 'enable_offload|\.offload\(' "$f" | head -800 || true
done

printf 'SUBENV_CLOSE_RESET\n'
grep -n -B35 -A100 -E '^    def (close|reset|setup_task)' "$SUPPORT/robotwin/envs/vector_env.py" | head -900 || true

printf 'TASK_RENDERER_CALLS\n'
grep -RIn --include='*.py' -E 'renderer|setup_task|close\(|clear_cache' "$SUPPORT/robotwin/envs" | head -800 || true
