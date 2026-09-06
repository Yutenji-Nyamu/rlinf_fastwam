#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
CANON=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
CKPT=/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384.pt
STATS=/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384_dataset_stats.json
PARENT_ID=fastwam-multitask-p2-3x16-c63dc9b5-v1
PARENT_META=/data/chenyiteng/results/dvac-observation/run-metadata/$PARENT_ID
VRT="$WT/third_party/RoboTwin"
TASKS=(move_stapler_pad turn_switch pick_diverse_bottles)

preflight() {
  test "$(git -C "$WT" rev-parse HEAD)" = c63dc9b5384d6637a93cc862dbe2815d0332801d
  test -z "$(git -C "$WT" diff --name-only)"
  test -s "$CKPT"
  test -s "$STATS"
  test -x "$ENV/bin/python"
  test ! -e "$PARENT_META"

  if nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -Eq '^[[:space:]]*[0-9]+'; then
    printf '%s\n' 'physical GPU 3 is not idle' >&2
    exit 1
  fi

  for name in assets task_config; do
    target="$CANON/third_party/RoboTwin/$name"
    link="$VRT/$name"
    test -L "$link"
    test "$(readlink -f "$link")" = "$(readlink -f "$target")"
  done
  test -L "$VRT/policy/fastwam_policy"
  test "$(readlink -f "$VRT/policy/fastwam_policy")" = \
    "$(readlink -f "$WT/experiments/robotwin/fastwam_policy")"

  for task in "${TASKS[@]}"; do
    test -f "$VRT/envs/$task.py"
    run_id="fastwam-${task}-p2-16ep-c63dc9b5-v1"
    test ! -e "/data/chenyiteng/results/dvac-observation/$run_id"
    test ! -e "/data/chenyiteng/results/dvac-observation/run-metadata/$run_id"
    test ! -e "$WT/evaluate_results/robotwin/robotwin_uncond_3cam_384/$run_id"
  done
}

run_task() {
  local task="$1"
  local run_id="fastwam-${task}-p2-16ep-c63dc9b5-v1"
  local meta="/data/chenyiteng/results/dvac-observation/run-metadata/$run_id"
  local payload="/data/chenyiteng/results/dvac-observation/$run_id"

  mkdir -p "$meta"
  local args=(
    'task=robotwin_uncond_3cam_384_1e-4'
    "ckpt=$CKPT"
    "EVALUATION.dataset_stats_path=$STATS"
    "EVALUATION.task_name=$task"
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
    "EVALUATION.dvac_telemetry.output_dir=$payload"
    "EVALUATION.dvac_telemetry.run_id=$run_id"
    "EVALUATION.output_dir=$run_id"
    'mixed_precision=bf16'
    'seed=42'
    'gpu_id=3'
  )

  "$ENV/bin/python" experiments/robotwin/eval_robotwin_single.py \
    "${args[@]}" --cfg job --resolve > "$meta/resolved.yaml"
  sha256sum "$meta/resolved.yaml" > "$meta/resolved.yaml.sha256"
  printf 'launch_time=%s\nsource_head=%s\nphysical_gpu=3\ntask=%s\nepisodes=16\nmax_action_slots=6400\nmax_queries=272\n' \
    "$(date --iso-8601=seconds)" "$(git -C "$WT" rev-parse HEAD)" "$task" \
    > "$meta/launch_manifest.txt"

  printf 'task_start=%s task=%s run_id=%s\n' "$(date --iso-8601=seconds)" "$task" "$run_id"
  "$ENV/bin/python" experiments/robotwin/eval_robotwin_single.py \
    "${args[@]}" > "$meta/driver.log" 2>&1
  printf 'task_complete=%s task=%s run_id=%s\n' "$(date --iso-8601=seconds)" "$task" "$run_id"
}

run_all() {
  cd "$WT"
  export PATH="$ENV/bin:$PATH"
  export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
  export DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
  export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope
  export TMPDIR=/home/chenyiteng/cache/fastwam-7faa/tmp
  export PYTHONUNBUFFERED=1
  unset CUDA_VISIBLE_DEVICES

  printf 'wrapper_start=%s\n' "$(date --iso-8601=seconds)"
  for task in "${TASKS[@]}"; do
    run_task "$task"
  done
  printf 'wrapper_complete=%s\n' "$(date --iso-8601=seconds)"
}

if test "${1:-}" = --run; then
  run_all
  exit 0
fi

preflight
mkdir -p "$PARENT_META"
printf 'launch_time=%s\nsource_head=%s\nphysical_gpu=3\ntasks=%s\nepisodes_per_task=16\ntotal_new_episodes=48\n' \
  "$(date --iso-8601=seconds)" "$(git -C "$WT" rev-parse HEAD)" "${TASKS[*]}" \
  > "$PARENT_META/launch_manifest.txt"

nohup setsid timeout --signal=TERM --kill-after=120s 43200s \
  bash "$0" --run > "$PARENT_META/driver.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$PARENT_META/driver.pid"
sleep 10
kill -0 "$pid"

printf 'parent_metadata=%s\npid=%s\ntasks=%s\n' \
  "$PARENT_META" "$pid" "${TASKS[*]}"
tail -n 20 "$PARENT_META/driver.log" || true
printf '%s\n' 'SZ_FASTWAM_DVAC_P2_MULTITASK_3X16_LAUNCHED'
