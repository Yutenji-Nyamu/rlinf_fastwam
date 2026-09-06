set -euo pipefail
date --iso-8601=seconds
echo '== owned eval/reset processes =='
pgrep -af '/home/chenyiteng/venvs/lerobot-v060-sidney-py310/bin/(lerobot-eval|python)|lerobot\.scripts\.lerobot_eval|sidney.*reset' || true
echo '== gpu =='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
echo '== latest sidney runs =='
find /data/chenyiteng/results/lerobot-sidney/pi05_robotwin-e49e2ab -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -12
echo '== exit and short log tails =='
for run in $(find /data/chenyiteng/results/lerobot-sidney/pi05_robotwin-e49e2ab -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -6 | cut -d' ' -f2-); do
  echo "RUN=$run"
  test -f "$run/exit_code.txt" && { printf 'exit='; cat "$run/exit_code.txt"; } || true
  test -f "$run/eval.log" && tail -8 "$run/eval.log" || true
done
