set -euo pipefail

REPO=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
RELEASE=/data/chenyiteng/models/fasterwam/release-6bf9471/robotwin
WAN_BASE=/data/chenyiteng/models/fastwam/diffsynth
RUN_ID=official-move-stapler-pad-random1-20260902-v1
RUN_ROOT=/data/chenyiteng/results/fasterwam-standalone/$RUN_ID
OFFICIAL_ROOT="$REPO/evaluate_results/robotwin/step_029355/$RUN_ID"

test "$(git -C "$REPO" rev-parse HEAD)" = 83667817df0d4f823f39d90700e61ea2f432ac45
test -x "$REPO/.venvs/robotwin/bin/python"
test -s "$RELEASE/step_029355.pt"
test -s "$RELEASE/dataset_stats.json"
test -s "$WAN_BASE/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
test -s "$WAN_BASE/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
test -d "$WAN_BASE/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
test ! -e "$RUN_ROOT"
test ! -e "$OFFICIAL_ROOT"

gpu_used_mib="$(nvidia-smi -i 3 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if test "$gpu_used_mib" -ge 100; then
    printf 'GPU3 is no longer idle: %s MiB used\n' "$gpu_used_mib" >&2
    exit 23
fi

mkdir -p "$RUN_ROOT/cache/matplotlib" "$RUN_ROOT/cache/numba"
export PYTHONPATH="$REPO/src:$REPO"
export DIFFSYNTH_MODEL_BASE_PATH="$WAN_BASE"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export MPLCONFIGDIR="$RUN_ROOT/cache/matplotlib"
export NUMBA_CACHE_DIR="$RUN_ROOT/cache/numba"
export PYTHONUNBUFFERED=1

cd "$REPO"
printf 'RUN_START %s\n' "$(date --iso-8601=seconds)"
printf 'GPU_BEFORE\n'
nvidia-smi -i 3 --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

.venvs/robotwin/bin/python -u experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_fasterwam_3cam_384_1e-4 \
  ckpt="$RELEASE/step_029355.pt" \
  gpu_id=3 \
  seed=0 \
  EVALUATION.robotwin_root="$REPO/third_party/RoboTwin" \
  EVALUATION.dataset_stats_path="$RELEASE/dataset_stats.json" \
  EVALUATION.task_name=move_stapler_pad \
  EVALUATION.task_config=demo_randomized \
  EVALUATION.instruction_type=unseen \
  EVALUATION.eval_num_episodes=1 \
  EVALUATION.replan_steps=28 \
  EVALUATION.num_inference_steps=10 \
  EVALUATION.action_infer_mode=one_pass_future_cache \
  EVALUATION.skip_get_obs_within_replan=true \
  EVALUATION.timing_enabled=true \
  EVALUATION.output_dir="$RUN_ROOT" \
  2>&1 | tee "$RUN_ROOT/driver.log"

test -d "$OFFICIAL_ROOT"
ln -s "$OFFICIAL_ROOT" "$RUN_ROOT/official_output"
printf 'RUN_END %s\n' "$(date --iso-8601=seconds)"
printf 'RESULT_FILES\n'
find "$OFFICIAL_ROOT" -maxdepth 4 -type f -printf '%s %p\n' | sort
printf 'RESULT_TEXT\n'
find "$OFFICIAL_ROOT" -type f -name '_result_random.txt' -exec sed -n '1,80p' {} \;
printf 'GPU_AFTER\n'
nvidia-smi -i 3 --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
