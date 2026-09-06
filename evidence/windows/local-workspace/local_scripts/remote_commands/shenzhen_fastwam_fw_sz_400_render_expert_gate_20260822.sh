#!/usr/bin/env bash
set -euo pipefail

# FW-SZ-400: the two remaining S4 behaviors after FW-SZ-110 eager imports:
# official vendor render and one complete adjust_bottle expert episode.

FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
FW_PIN=7faa71108368fbb3b6885649f112af607427a2d4
VRT="$FW/third_party/RoboTwin"
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
PYTHON="$ENV/bin/python"
RESULT_ROOT=/data/chenyiteng/results/fastwam-standalone
RUN_ID="${FASTWAM_S4_RUN_ID:-fw-sz-400-$(date +%Y%m%d_%H%M%S)}"
RENDER_TIMEOUT_SECONDS=300
EXPERT_TIMEOUT_SECONDS=3600

GPU="${FASTWAM_PHYSICAL_GPU:-3}"
[[ "$GPU" =~ ^[0-9]+$ ]]
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]

RUN_ROOT="$RESULT_ROOT/$RUN_ID"
CFG_NAME="${RUN_ID}_demo_clean_1ep"
DERIVED_CFG="$VRT/task_config/$CFG_NAME.yml"
EXPERT_ROOT="$VRT/data/adjust_bottle/$CFG_NAME"
RESOURCE_CSV="$RUN_ROOT/resource_2s.csv"

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

printf 'fw_stage=FW-SZ-400\ntimestamp_start=%s\nrun_id=%s\nphysical_gpu=%s\n' \
  "$(date --iso-8601=seconds)" "$RUN_ID" "$GPU"
test -x "$PYTHON"
test "$(git -C "$FW" rev-parse HEAD)" = "$FW_PIN"
git -C "$FW" diff HEAD --quiet -- .
test "$(git hash-object "$VRT/task_config/demo_clean.yml")" = 7e0de0aec1ad1b151570486120c7d25050be01c4

gpu_apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits 2>/dev/null || true)
[[ -z "${gpu_apps//[[:space:]]/}" ]] || {
  printf 'STOP_GPU_BUSY physical_gpu=%s\n%s\n' "$GPU" "$gpu_apps" >&2
  exit 20
}
printf '%s\n' '=== before: live GPU/RAM/disk ==='
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
free -h
df -h /data /home
pgrep -af '([t]rain_embodied_agent\.py|[r]aylet|[e]val_robotwin_single\.py|[c]ollect_data\.py)' || true

[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]]
[[ ! -e "$DERIVED_CFG" && ! -L "$DERIVED_CFG" ]]
[[ ! -e "$EXPERT_ROOT" && ! -L "$EXPERT_ROOT" ]]
mkdir -p "$RESULT_ROOT"
mkdir "$RUN_ROOT"

"$PYTHON" - "$VRT/task_config/demo_clean.yml" "$DERIVED_CFG" <<'PY'
from pathlib import Path
import sys
import yaml

source, target = map(Path, sys.argv[1:])
cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
assert cfg["use_seed"] is False
assert cfg["save_path"] == "./data"
assert cfg["collect_data"] is True
cfg["episode_num"] = 1
with target.open("x", encoding="utf-8", newline="\n") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print(f"derived_task_config={target}")
PY

export PATH="$ENV/bin:$PATH"
MONITOR_PID=
monitor_resources &
MONITOR_PID=$!

cd "$VRT"
printf 'render_command=CUDA_VISIBLE_DEVICES=%s %q script/test_render.py\n' "$GPU" "$PYTHON"
set +e
CUDA_VISIBLE_DEVICES="$GPU" timeout --signal=TERM --kill-after=30s "$RENDER_TIMEOUT_SECONDS" \
  "$PYTHON" script/test_render.py 2>&1 | tee "$RUN_ROOT/test_render.log"
render_rc=${PIPESTATUS[0]}
set -e
[[ "$render_rc" == 0 ]]
grep -Fq 'Render Well' "$RUN_ROOT/test_render.log"
! grep -Fq 'Render Error' "$RUN_ROOT/test_render.log"

printf 'expert_command=CUDA_VISIBLE_DEVICES=%s %q script/collect_data.py adjust_bottle %q\n' \
  "$GPU" "$PYTHON" "$CFG_NAME"
set +e
CUDA_VISIBLE_DEVICES="$GPU" timeout --signal=TERM --kill-after=60s "$EXPERT_TIMEOUT_SECONDS" \
  "$PYTHON" script/collect_data.py adjust_bottle "$CFG_NAME" \
  2>&1 | tee "$RUN_ROOT/collect_data.log"
expert_rc=${PIPESTATUS[0]}
set -e
[[ "$expert_rc" == 0 ]]
grep -Fq 'simulate data episode 0 success!' "$RUN_ROOT/collect_data.log"

stop_monitor
(( $(wc -l < "$RESOURCE_CSV") > 1 ))

SEED_FILE="$EXPERT_ROOT/seed.txt"
HDF5_FILE="$EXPERT_ROOT/data/episode0.hdf5"
VIDEO_FILE="$EXPERT_ROOT/video/episode0.mp4"
SCENE_FILE="$EXPERT_ROOT/scene_info.json"
INSTRUCTION_FILE="$EXPERT_ROOT/instructions/episode0.json"
for artifact in "$SEED_FILE" "$HDF5_FILE" "$VIDEO_FILE" "$SCENE_FILE" "$INSTRUCTION_FILE"; do
  [[ -f "$artifact" && -s "$artifact" ]]
done
seed_count=$(wc -w < "$SEED_FILE")
[[ "$seed_count" == 1 ]]
accepted_expert_seed=$(xargs < "$SEED_FILE")
printf 'accepted_expert_seed=%s\n' "$accepted_expert_seed"

ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,nb_frames:format=duration \
  -of default=noprint_wrappers=1 "$VIDEO_FILE" | tee "$RUN_ROOT/expert_video_probe.txt"
grep -Fq 'codec_name=h264' "$RUN_ROOT/expert_video_probe.txt"

awk -F, 'NR > 1 {
  used=$2-$3; if (used>host) host=used; if ($4>gpu) gpu=$4
} END {printf "peak_host_used_kib=%s\npeak_gpu_memory_used_mib=%s\n", host, gpu}' \
  "$RESOURCE_CSV" | tee "$RUN_ROOT/resource_peak.txt"

printf '%s\n' '=== S4 outputs ==='
stat -c 'bytes=%s path=%n' -- "$DERIVED_CFG" "$SEED_FILE" "$HDF5_FILE" \
  "$VIDEO_FILE" "$SCENE_FILE" "$INSTRUCTION_FILE" "$RUN_ROOT"/*.log \
  "$RESOURCE_CSV" "$RUN_ROOT/resource_peak.txt" "$RUN_ROOT/expert_video_probe.txt"
printf '%s\n' '=== after: selected GPU/RAM ==='
nvidia-smi -i "$GPU" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
free -h
printf 'timestamp_end=%s\nFASTWAM_FW_SZ_400_S4_OK\n' "$(date --iso-8601=seconds)"
