set -u

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
SUPPORT=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support

date '+TIME %Y-%m-%d %H:%M:%S %Z'

printf 'VECTOR_ENV_SYMBOLS\n'
grep -n -E '^    def (offload|close|reset|_initialize_envs|_init|_create)|ThreadPoolExecutor|shutdown\(' "$SUPPORT/robotwin/envs/vector_env.py" || true

printf 'VECTOR_ENV_OFFLOAD\n'
line=$(grep -n '^    def offload' "$SUPPORT/robotwin/envs/vector_env.py" | head -1 | cut -d: -f1)
if test -n "$line"; then start=$((line-35)); end=$((line+110)); sed -n "${start},${end}p" "$SUPPORT/robotwin/envs/vector_env.py"; fi

printf 'ROBOTWIN_WRAPPER_OFFLOAD\n'
grep -n -B30 -A100 -E '^    def (offload|close|reset)' "$WT/rlinf/envs/robotwin/robotwin_env.py" | head -520 || true

printf 'ENV_FACTORY_OFFLOAD\n'
sed -n '1,120p' "$WT/rlinf/envs/__init__.py" 2>/dev/null || true

printf 'GLIBC_LIMITS\n'
getconf PTHREAD_KEYS_MAX 2>/dev/null || true
getconf CHILD_MAX 2>/dev/null || true
ulimit -u
cat /proc/sys/kernel/threads-max
cat /sys/fs/cgroup/pids.max 2>/dev/null || true
cat /sys/fs/cgroup/pids.current 2>/dev/null || true
