#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
CANON=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
CKPT=/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384.pt
STATS=/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384_dataset_stats.json
RUN_ID=fastwam-adjust_bottle-p1-16ep-c63dc9b5-v1
TELEMETRY_DIR=/data/chenyiteng/results/dvac-observation/$RUN_ID
OFFICIAL_DIR="$WT/evaluate_results/robotwin/robotwin_uncond_3cam_384/$RUN_ID"
VRT="$WT/third_party/RoboTwin"

test "$(git -C "$WT" rev-parse HEAD)" = c63dc9b5384d6637a93cc862dbe2815d0332801d
test -z "$(git -C "$WT" status --porcelain)"
test -s "$CKPT"
test -s "$STATS"
test -x "$ENV/bin/python"
test ! -e "$TELEMETRY_DIR"
test ! -e "$OFFICIAL_DIR"

if nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'physical GPU 3 is not idle' >&2
  exit 1
fi

for name in assets task_config; do
  target="$CANON/third_party/RoboTwin/$name"
  link="$VRT/$name"
  test -e "$target"
  if test -L "$link"; then
    test "$(readlink -f "$link")" = "$(readlink -f "$target")"
  else
    test ! -e "$link"
    ln -s "$target" "$link"
  fi
done

cd "$WT"
export PATH="$ENV/bin:$PATH"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope
export TMPDIR=/home/chenyiteng/cache/fastwam-7faa/tmp
export PYTHONUNBUFFERED=1
unset CUDA_VISIBLE_DEVICES

mkdir -p "$TELEMETRY_DIR"
ARGS=(
  'task=robotwin_uncond_3cam_384_1e-4'
  "ckpt=$CKPT"
  "EVALUATION.dataset_stats_path=$STATS"
  'EVALUATION.task_name=adjust_bottle'
  'EVALUATION.task_config=demo_clean'
  'EVALUATION.eval_num_episodes=16'
  'EVALUATION.instruction_type=unseen'
  'EVALUATION.action_horizon=null'
  'EVALUATION.num_inference_steps=10'
  'EVALUATION.sigma_shift=5.0'
  'EVALUATION.replan_steps=24'
  'EVALUATION.rand_device=cpu'
  'EVALUATION.tiled=false'
  'EVALUATION.skip_get_obs_within_replan=false'
  'EVALUATION.dvac_telemetry.enabled=true'
  "EVALUATION.dvac_telemetry.output_dir=$TELEMETRY_DIR"
  "EVALUATION.dvac_telemetry.run_id=$RUN_ID"
  "EVALUATION.output_dir=$RUN_ID"
  'mixed_precision=bf16'
  'seed=42'
  'gpu_id=3'
)

"$ENV/bin/python" experiments/robotwin/eval_robotwin_single.py \
  "${ARGS[@]}" --cfg job --resolve > "$TELEMETRY_DIR/resolved.yaml"
sha256sum "$TELEMETRY_DIR/resolved.yaml" > "$TELEMETRY_DIR/resolved.yaml.sha256"
printf 'launch_time=%s\nsource_head=%s\nphysical_gpu=3\nepisodes=16\nmax_action_slots=6400\nmax_queries=272\n' \
  "$(date --iso-8601=seconds)" "$(git -C "$WT" rev-parse HEAD)" \
  > "$TELEMETRY_DIR/launch_manifest.txt"

nohup setsid timeout --signal=TERM --kill-after=120s 10800s \
  "$ENV/bin/python" experiments/robotwin/eval_robotwin_single.py \
  "${ARGS[@]}" > "$TELEMETRY_DIR/driver.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$TELEMETRY_DIR/driver.pid"

sleep 10
kill -0 "$pid"
printf 'telemetry=%s\nofficial_result=%s\npid=%s\nresolved_sha256=%s\n' \
  "$TELEMETRY_DIR" "$OFFICIAL_DIR" "$pid" \
  "$(awk '{print $1}' "$TELEMETRY_DIR/resolved.yaml.sha256")"
tail -n 20 "$TELEMETRY_DIR/driver.log" || true
printf '%s\n' 'SZ_FASTWAM_DVAC_P1_16EP_LAUNCHED'
