#!/usr/bin/env bash
set -euo pipefail

# FW-SZ-500: one official Fast-WAM 7faa711 RoboTwin release episode.
# Preconditions: FW-SZ-300/310 inputs and FW-SZ-400 already passed.

FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
FW_PIN=7faa71108368fbb3b6885649f112af607427a2d4
VRT="$FW/third_party/RoboTwin"
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
PYTHON="$ENV/bin/python"
CACHE_ROOT=/home/chenyiteng/cache/fastwam-7faa
MODEL_BASE=/data/chenyiteng/models/fastwam/diffsynth
RELEASE_ROOT=/data/chenyiteng/models/fastwam/release-8eaceeb
CKPT="$RELEASE_ROOT/robotwin_uncond_3cam_384.pt"
STATS="$RELEASE_ROOT/robotwin_uncond_3cam_384_dataset_stats.json"
RESULT_ROOT=/data/chenyiteng/results/fastwam-standalone
RUN_LEAF="${FASTWAM_S5_RUN_ID:-fw-sz-500-$(date +%Y%m%d_%H%M%S)}"
EVAL_TIMEOUT_SECONDS=7200
SEED=42
ENV_START_SEED=4300000

GPU="${FASTWAM_PHYSICAL_GPU:-3}"
[[ "$GPU" =~ ^[0-9]+$ ]]
[[ "$RUN_LEAF" =~ ^[A-Za-z0-9._-]+$ ]]

RUN_DIR="$FW/evaluate_results/robotwin/robotwin_uncond_3cam_384/$RUN_LEAF"
TASK_DIR="$RUN_DIR/adjust_bottle"
EVIDENCE_PREFIX="$RESULT_ROOT/$RUN_LEAF"
CONSOLE_LOG="$EVIDENCE_PREFIX.console.log"
RESOURCE_CSV="$EVIDENCE_PREFIX.resource_2s.csv"
RESOURCE_PEAK="$EVIDENCE_PREFIX.resource_peak.txt"
COMMAND_TXT="$EVIDENCE_PREFIX.command.txt"
RESOLVED_CFG="$EVIDENCE_PREFIX.resolved_config.yaml"
VIDEO_PROBE="$EVIDENCE_PREFIX.video_probe.txt"

stop_monitor() {
  if [[ -n "${MONITOR_PID:-}" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  MONITOR_PID=
}
trap stop_monitor EXIT

monitor_resources() {
  printf 'epoch,mem_total_kib,mem_available_kib,gpu_memory_used_mib,gpu_utilization_pct\n' > "$RESOURCE_CSV"
  while :; do
    local total available gpu_values gpu_mem gpu_util
    total=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
    available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    gpu_values=$(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits | tr -d ' ')
    IFS=, read -r gpu_mem gpu_util <<< "$gpu_values"
    printf '%s,%s,%s,%s,%s\n' "$(date +%s)" "$total" "$available" "$gpu_mem" "$gpu_util" >> "$RESOURCE_CSV"
    sleep 2
  done
}

printf 'fw_stage=FW-SZ-500\ntimestamp_start=%s\nrun_leaf=%s\nphysical_gpu=%s\n' \
  "$(date --iso-8601=seconds)" "$RUN_LEAF" "$GPU"
test -x "$PYTHON"
test "$(git -C "$FW" rev-parse HEAD)" = "$FW_PIN"
git -C "$FW" diff HEAD --quiet -- .
test "$(git hash-object "$VRT/task_config/demo_clean.yml")" = 7e0de0aec1ad1b151570486120c7d25050be01c4

POLICY_LINK="$VRT/policy/fastwam_policy"
POLICY_SOURCE="$FW/experiments/robotwin/fastwam_policy"
[[ -L "$POLICY_LINK" ]]
[[ "$(readlink -f "$POLICY_LINK")" == "$(readlink -f "$POLICY_SOURCE")" ]]

gpu_apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits 2>/dev/null || true)
[[ -z "${gpu_apps//[[:space:]]/}" ]] || {
  printf 'STOP_GPU_BUSY physical_gpu=%s\n%s\n' "$GPU" "$gpu_apps" >&2
  exit 30
}
printf '%s\n' '=== before: live GPU/RAM/disk and output absence ==='
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
free -h
df -h /data /home
pgrep -af '([t]rain_embodied_agent\.py|[r]aylet|[e]val_robotwin_single\.py|[c]ollect_data\.py)' || true
[[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]]
for path in "$CONSOLE_LOG" "$RESOURCE_CSV" "$RESOURCE_PEAK" "$COMMAND_TXT" \
  "$RESOLVED_CFG" "$VIDEO_PROBE"; do
  [[ ! -e "$path" && ! -L "$path" ]]
done
mkdir -p "$RESULT_ROOT"
printf 'output_absent_before=1 path=%s\n' "$RUN_DIR"

[[ -s "$CKPT" && -s "$STATS" ]]

export PATH="$ENV/bin:$PATH"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_BASE"
export MODELSCOPE_CACHE="$CACHE_ROOT/modelscope"
export TMPDIR="$CACHE_ROOT/tmp"
export PYTHONUNBUFFERED=1
mkdir -p "$MODELSCOPE_CACHE" "$TMPDIR"

# The single entry overwrites its child's CUDA_VISIBLE_DEVICES from gpu_id, so
# gpu_id is the selected physical ID. Do not outer-remap and then pass zero.
unset CUDA_VISIBLE_DEVICES
OVERRIDES=(
  task=robotwin_uncond_3cam_384_1e-4
  "ckpt=$CKPT"
  "EVALUATION.dataset_stats_path=$STATS"
  EVALUATION.task_name=adjust_bottle
  EVALUATION.task_config=demo_clean
  EVALUATION.eval_num_episodes=1
  EVALUATION.instruction_type=unseen
  EVALUATION.action_horizon=null
  EVALUATION.num_inference_steps=10
  EVALUATION.sigma_shift=5.0
  EVALUATION.replan_steps=24
  EVALUATION.skip_get_obs_within_replan=true
  "EVALUATION.output_dir=$RUN_LEAF"
  "seed=$SEED"
  "gpu_id=$GPU"
)
CMD=("$PYTHON" experiments/robotwin/eval_robotwin_single.py "${OVERRIDES[@]}")

{
  printf 'cwd=%q\n' "$FW"
  printf 'actual_output_dir=%q\n' "$RUN_DIR"
  printf 'physical_gpu=%q\n' "$GPU"
  printf 'source_seed=%q\nsource_start_seed=%q\n' "$SEED" "$ENV_START_SEED"
  printf 'command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "$COMMAND_TXT"

cd "$FW"
"$PYTHON" experiments/robotwin/eval_robotwin_single.py --cfg job --resolve \
  "${OVERRIDES[@]}" > "$RESOLVED_CFG"
grep -Fq 'sigma_shift: 5.0' "$RESOLVED_CFG"
grep -Fq 'replan_steps: 24' "$RESOLVED_CFG"
grep -Fq "gpu_id: $GPU" "$RESOLVED_CFG"
grep -Fq "seed: $SEED" "$RESOLVED_CFG"
cat "$COMMAND_TXT"

MONITOR_PID=
monitor_resources &
MONITOR_PID=$!
set +e
timeout --signal=TERM --kill-after=60s "$EVAL_TIMEOUT_SECONDS" \
  "${CMD[@]}" 2>&1 | tee "$CONSOLE_LOG"
eval_rc=${PIPESTATUS[0]}
set -e
[[ "$eval_rc" == 0 ]]
stop_monitor
(( $(wc -l < "$RESOURCE_CSV") > 1 ))

grep -Fq 'Evaluation finished successfully. Log saved to:' "$CONSOLE_LOG"
CONFIG_FILE="$RUN_DIR/eval_config_adjust_bottle.yaml"
RESULT_FILE="$TASK_DIR/_result_clean.txt"
[[ -s "$CONFIG_FILE" && -s "$RESULT_FILE" ]]
mapfile -t EVAL_LOGS < <(find "$RUN_DIR" -maxdepth 1 -type f -name 'eval_adjust_bottle_*.log' -print)
mapfile -t VIDEOS < <(find "$TASK_DIR" -maxdepth 1 -type f \
  -name 'episode0_randomized-false_success-*.mp4' -print)
[[ "${#EVAL_LOGS[@]}" == 1 && "${#VIDEOS[@]}" == 1 ]]
EVAL_LOG="${EVAL_LOGS[0]}"
VIDEO_FILE="${VIDEOS[0]}"
[[ -s "$EVAL_LOG" && -s "$VIDEO_FILE" ]]
grep -Fq 'Render Well' "$EVAL_LOG"
! grep -Fq 'Render Error' "$EVAL_LOG"

rate=$(awk 'NF {last=$0} END {print last}' "$RESULT_FILE")
case "$rate" in
  1.0) [[ "$(basename "$VIDEO_FILE")" == *'_success-true.mp4' ]] ;;
  0.0) [[ "$(basename "$VIDEO_FILE")" == *'_success-false.mp4' ]] ;;
  *) printf 'STOP_INVALID_SINGLE_EPISODE_RATE=%q\n' "$rate" >&2; exit 32 ;;
esac
accepted_seed=$(sed -r 's/\x1B\[[0-9;]*[[:alpha:]]//g' "$EVAL_LOG" \
  | sed -n -E 's/.*current seed:[[:space:]]*([0-9]+).*/\1/p' | tail -n 1)
[[ "$accepted_seed" =~ ^[0-9]+$ ]]

ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=codec_name,width,height,nb_read_frames:format=duration \
  -of default=noprint_wrappers=1 "$VIDEO_FILE" | tee "$VIDEO_PROBE"
grep -Fq 'codec_name=h264' "$VIDEO_PROBE"
grep -Fq 'width=640' "$VIDEO_PROBE"
grep -Fq 'height=480' "$VIDEO_PROBE"

awk -F, 'NR > 1 {
  used=$2-$3; if (used>host) host=used; if ($4>gpu) gpu=$4
} END {printf "peak_host_used_kib=%s\npeak_gpu_memory_used_mib=%s\n", host, gpu}' \
  "$RESOURCE_CSV" | tee "$RESOURCE_PEAK"

printf 'single_episode_rate=%s\naccepted_expert_seed=%s\n' "$rate" "$accepted_seed"
printf '%s\n' '=== S5 outputs ==='
stat -c 'bytes=%s path=%n' -- "$COMMAND_TXT" "$RESOLVED_CFG" \
  "$CONSOLE_LOG" "$RESOURCE_CSV" "$RESOURCE_PEAK" "$CONFIG_FILE" "$EVAL_LOG" \
  "$RESULT_FILE" "$VIDEO_FILE" "$VIDEO_PROBE"
printf '%s\n' '=== after: selected GPU/RAM ==='
nvidia-smi -i "$GPU" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
free -h
printf 'timestamp_end=%s\nFASTWAM_FW_SZ_500_SINGLE_EVAL_OK\n' "$(date --iso-8601=seconds)"
