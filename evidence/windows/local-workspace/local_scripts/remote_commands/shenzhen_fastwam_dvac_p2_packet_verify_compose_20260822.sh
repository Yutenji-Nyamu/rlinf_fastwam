#!/usr/bin/env bash
set -euo pipefail

PACKET=/data/chenyiteng/results/dvac-observation/packets/fastwam-multitask-p2-3x16-c63dc9b5-v1
LAUNCHER="$PACKET/shenzhen_fastwam_dvac_launch_p2_multitask_3x16_gpu3_20260822.sh"
EXPECTED=22faa66c72844011d811856a0ba5e67946cb97b2332edbc576de004dd27c1534
WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
CKPT=/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384.pt
STATS=/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384_dataset_stats.json

test "$(sha256sum "$LAUNCHER" | awk '{print $1}')" = "$EXPECTED"
bash -n "$LAUNCHER"
test "$(git -C "$WT" rev-parse HEAD)" = c63dc9b5384d6637a93cc862dbe2815d0332801d
test -z "$(git -C "$WT" diff --name-only)"
test -x "$ENV/bin/python"
test -s "$CKPT"
test -s "$STATS"

cd "$WT"
export PATH="$ENV/bin:$PATH"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope
export TMPDIR=/home/chenyiteng/cache/fastwam-7faa/tmp
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=

for task in move_stapler_pad turn_switch pick_diverse_bottles; do
  run_id="fastwam-${task}-p2-16ep-c63dc9b5-v1"
  "$ENV/bin/python" experiments/robotwin/eval_robotwin_single.py \
    task=robotwin_uncond_3cam_384_1e-4 \
    ckpt="$CKPT" \
    EVALUATION.dataset_stats_path="$STATS" \
    EVALUATION.task_name="$task" \
    EVALUATION.task_config=demo_clean \
    EVALUATION.eval_num_episodes=16 \
    EVALUATION.instruction_type=unseen \
    EVALUATION.action_horizon=null \
    EVALUATION.num_inference_steps=10 \
    EVALUATION.sigma_shift=5.0 \
    EVALUATION.replan_steps=24 \
    EVALUATION.rand_device=cpu \
    EVALUATION.tiled=false \
    EVALUATION.skip_get_obs_within_replan=false \
    EVALUATION.dvac_telemetry.enabled=true \
    EVALUATION.dvac_telemetry.output_dir="/data/chenyiteng/results/dvac-observation/$run_id" \
    EVALUATION.dvac_telemetry.run_id="$run_id" \
    EVALUATION.output_dir="$run_id" \
    mixed_precision=bf16 seed=42 gpu_id=3 --cfg job --resolve \
    > "$PACKET/resolved-${task}.yaml"
  sha256sum "$PACKET/resolved-${task}.yaml" > "$PACKET/resolved-${task}.yaml.sha256"
  printf 'task=%s resolved_sha256=%s\n' "$task" \
    "$(awk '{print $1}' "$PACKET/resolved-${task}.yaml.sha256")"
done

printf 'launcher_sha256=%s\n' "$EXPECTED"
printf '%s\n' 'SZ_FASTWAM_DVAC_P2_PACKET_VERIFY_COMPOSE_OK'
