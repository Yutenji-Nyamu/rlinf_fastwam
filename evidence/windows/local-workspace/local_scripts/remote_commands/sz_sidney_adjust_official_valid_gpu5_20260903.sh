set -euo pipefail
task=adjust_bottle
gpu=5
seeds='1004 1009'
base=/data/chenyiteng/results/lerobot-sidney/pi05_robotwin-e49e2ab
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/third_party/RoboTwin${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/data/chenyiteng/cache/huggingface-sidney
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR="/data/chenyiteng/cache/torchinductor-sidney-gpu${gpu}"
cd /data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/third_party/RoboTwin
for seed in $seeds; do
  run_root="$base/${task}-official-valid-seed${seed}-20260903"
  mkdir -p "$run_root"
  printf 'task=%s\nseed=%s\nn_episodes=1\ngpu_physical=%s\nmodel_revision=%s\nlerobot_revision=%s\n' \
    "$task" "$seed" "$gpu" \
    e49e2ab6c11f07511573b67261bd129e88d0a416 \
    30da8e687a6dfc617fcd94afc367ac7071c376ce > "$run_root/run_identity.txt"
  set +e
  stdbuf -oL -eL /home/chenyiteng/venvs/lerobot-v060-sidney-py310/bin/lerobot-eval \
    --policy.path=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab \
    --env.type=robotwin --env.task="$task" --eval.batch_size=1 --eval.n_episodes=1 --seed="$seed" \
    --rename_map='{"observation.images.head_camera":"observation.images.cam_high","observation.images.left_camera":"observation.images.cam_left_wrist","observation.images.right_camera":"observation.images.cam_right_wrist"}' \
    --output_dir="$run_root/eval" 2>&1 | tee "$run_root/eval.log"
  rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$rc" > "$run_root/exit_code.txt"
  if [ "$rc" -ne 0 ]; then
    if grep -q 'UnStableError' "$run_root/eval.log"; then
      printf 'UNSTABLE_AFTER_PROBE task=%s seed=%s\n' "$task" "$seed" | tee "$run_root/result.txt"
      continue
    fi
    printf 'NON_UNSTABLE_ERROR task=%s seed=%s rc=%s\n' "$task" "$seed" "$rc" | tee "$run_root/result.txt"
    exit "$rc"
  fi
  printf 'OFFICIAL_EVAL_DONE task=%s seed=%s rc=0\n' "$task" "$seed" | tee "$run_root/result.txt"
done
