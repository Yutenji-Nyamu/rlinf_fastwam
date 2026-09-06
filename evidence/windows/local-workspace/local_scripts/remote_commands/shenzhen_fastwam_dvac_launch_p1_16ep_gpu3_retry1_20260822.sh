#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
CANON=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
CKPT=/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384.pt
STATS=/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384_dataset_stats.json
RUN_ID=fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2
META=/data/chenyiteng/results/dvac-observation/run-metadata/$RUN_ID
PAYLOAD=/data/chenyiteng/results/dvac-observation/$RUN_ID
OFFICIAL_DIR="$WT/evaluate_results/robotwin/robotwin_uncond_3cam_384/$RUN_ID"
VRT="$WT/third_party/RoboTwin"

test "$(git -C "$WT" rev-parse HEAD)" = c63dc9b5384d6637a93cc862dbe2815d0332801d
test -z "$(git -C "$WT" diff --name-only)"
mapfile -t untracked < <(git -C "$WT" ls-files --others --exclude-standard | LC_ALL=C sort)
expected_untracked=(
  third_party/RoboTwin/assets
  third_party/RoboTwin/policy/fastwam_policy
  third_party/RoboTwin/task_config
)
test "${#untracked[@]}" = "${#expected_untracked[@]}"
for i in "${!expected_untracked[@]}"; do
  test "${untracked[$i]}" = "${expected_untracked[$i]}"
done
test -s "$CKPT"
test -s "$STATS"
test -x "$ENV/bin/python"
test ! -e "$META"
test ! -e "$PAYLOAD"
test ! -e "$OFFICIAL_DIR"

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

cd "$WT"
export PATH="$ENV/bin:$PATH"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope
export TMPDIR=/home/chenyiteng/cache/fastwam-7faa/tmp
export PYTHONUNBUFFERED=1
unset CUDA_VISIBLE_DEVICES

mkdir -p "$META"
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
  "EVALUATION.dvac_telemetry.output_dir=$PAYLOAD"
  "EVALUATION.dvac_telemetry.run_id=$RUN_ID"
  "EVALUATION.output_dir=$RUN_ID"
  'mixed_precision=bf16'
  'seed=42'
  'gpu_id=3'
)

"$ENV/bin/python" experiments/robotwin/eval_robotwin_single.py \
  "${ARGS[@]}" --cfg job --resolve > "$META/resolved.yaml"
sha256sum "$META/resolved.yaml" > "$META/resolved.yaml.sha256"
printf 'launch_time=%s\nsource_head=%s\nphysical_gpu=3\nepisodes=16\nmax_action_slots=6400\nmax_queries=272\n' \
  "$(date --iso-8601=seconds)" "$(git -C "$WT" rev-parse HEAD)" \
  > "$META/launch_manifest.txt"

nohup setsid timeout --signal=TERM --kill-after=120s 10800s \
  "$ENV/bin/python" experiments/robotwin/eval_robotwin_single.py \
  "${ARGS[@]}" > "$META/driver.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$META/driver.pid"

sleep 10
kill -0 "$pid"
printf 'metadata=%s\ntelemetry=%s\nofficial_result=%s\npid=%s\nresolved_sha256=%s\n' \
  "$META" "$PAYLOAD" "$OFFICIAL_DIR" "$pid" \
  "$(awk '{print $1}' "$META/resolved.yaml.sha256")"
tail -n 20 "$META/driver.log" || true
printf '%s\n' 'SZ_FASTWAM_DVAC_P1_16EP_RETRY1_LAUNCHED'
