set -euo pipefail

task=adjust_bottle
seed=1002
n_episodes=4
stamp=$(date +%Y%m%d-%H%M%S)
run_root="/data/chenyiteng/results/lerobot-sidney/pi05_robotwin-e49e2ab/${task}-${n_episodes}eps-seed${seed}-${stamp}"
mkdir -p "$run_root"

export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH="/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/third_party/RoboTwin${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data/chenyiteng/cache/huggingface-sidney
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

printf 'run_root=%s\ntask=%s\nseed=%s\nn_episodes=%s\nmodel_revision=%s\nlerobot_revision=%s\ngpu_physical=%s\n' \
  "$run_root" "$task" "$seed" "$n_episodes" \
  e49e2ab6c11f07511573b67261bd129e88d0a416 \
  30da8e687a6dfc617fcd94afc367ac7071c376ce \
  4 | tee "$run_root/run_identity.txt"

(
  printf 'timestamp,memory_used_mib,utilization_gpu_pct\n'
  while true; do
    ts=$(date --iso-8601=seconds)
    vals=$(nvidia-smi -i 4 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1)
    printf '%s,%s\n' "$ts" "$vals"
    sleep 5
  done
) > "$run_root/resource_gpu4.csv" &
sampler_pid=$!
cleanup() {
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
}
trap cleanup EXIT

set +e
cd /data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/third_party/RoboTwin
stdbuf -oL -eL /home/chenyiteng/venvs/lerobot-v060-sidney-py310/bin/lerobot-eval \
  --policy.path=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab \
  --env.type=robotwin \
  --env.task="$task" \
  --eval.batch_size=1 \
  --eval.n_episodes="$n_episodes" \
  --seed="$seed" \
  --rename_map='{"observation.images.head_camera":"observation.images.cam_high","observation.images.left_camera":"observation.images.cam_left_wrist","observation.images.right_camera":"observation.images.cam_right_wrist"}' \
  --output_dir="$run_root/eval" \
  2>&1 | tee "$run_root/eval.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$run_root/exit_code.txt"
printf 'eval_exit_code=%s\n' "$rc"
exit "$rc"
