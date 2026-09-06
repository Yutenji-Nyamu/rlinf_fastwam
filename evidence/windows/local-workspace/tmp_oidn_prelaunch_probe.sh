set -euo pipefail
ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo
SLUG=fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1
RUN="$ROOT/runs/$SLUG"
PACKET="$ROOT/packets/$SLUG"
RT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin

date '+TIME %Y-%m-%d %H:%M:%S %Z'
printf 'GPU\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'GPU_PROCS\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader 2>/dev/null || true
printf 'MEM_DISK\n'
free -h | sed -n '1,2p'
df -h /data /home | tail -n +2
printf 'RAY\n'
RAY_ADDRESS=172.17.0.1:6389 "$VENV/bin/ray" status | sed -n '1,35p'
printf 'RT\n'
git -C "$RT" status --short --branch
git -C "$RT" log -1 --format='%H %s'
printf 'OLD_EXIT\n'
for f in exit_code exit_code.txt finished_at.txt; do test -f "$RUN/runtime/$f" && printf '%s=' "$f" && cat "$RUN/runtime/$f"; done
printf 'STEP10\n'
find "$RUN" -type d -name global_step_10 -print
STEP10=$(find "$RUN" -type d -name global_step_10 -print -quit)
test -n "$STEP10"
du -sh "$STEP10"
find "$STEP10" -type f \( -name '.metadata' -o -name '*.distcp' -o -name 'complete.json' -o -name 'manifest.json' \) -printf '%s %P\n' | sort
printf 'COMMAND\n'
cat "$PACKET/command.txt"
printf '\nRESUME_SCHEMA_HITS\n'
rg -n 'resume_dir|resume_from|load_checkpoint|checkpoint_path' /data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo/rlinf /data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo/examples/embodiment | head -120 || true
