set -eu
RT=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
echo '=== locks ==='
git -C "$RT" rev-parse HEAD
git -C "$RT" status --short
git -C "$WT" rev-parse HEAD
git -C "$WT" status --short
echo '=== files ==='
for f in \
  "$WT/rlinf/workers/env/env_worker.py" \
  "$WT/rlinf/envs/robotwin/robotwin_env.py" \
  "$RT/robotwin/envs/vector_env.py" \
  "$RT/envs/_base_task.py"; do
  if test -f "$f"; then echo "FILE=$f"; fi
done
echo '=== vector symbols ==='
grep -nE 'ThreadPoolExecutor|def _init_envs|def reset|def close|clear_cache_freq|close_env|shutdown|env_thread_pool' "$RT/robotwin/envs/vector_env.py" || true
echo '=== base task candidates ==='
find "$RT" -maxdepth 4 -type f \( -name '*base_task*.py' -o -name '_base_task.py' \) -print
echo '=== base task close renderer ==='
grep -R -nE 'def close_env|SapienRenderer|set_ray_tracing_denoiser|sapien_clear_cache' "$RT/envs" 2>/dev/null | head -n 80 || true
echo '=== env worker lifecycle ==='
grep -nE 'train_enable_offload|eval_enable_offload|def _init_env|def evaluate|def rollout|offload' "$WT/rlinf/workers/env/env_worker.py" | tail -n 60
