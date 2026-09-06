set -euo pipefail

RAY_ADDRESS=172.17.0.1:6389
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
FASTWAM_SRC=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/src
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
OLD_SLUG=fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v1
NEW_SLUG=fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2
OLD_EXPERIMENT=fastwam_grpo_control_formal100_2gpu32x4_g8_b1024_u2_m10_fixed32_eval5_phys67_dcp_v1
NEW_EXPERIMENT=fastwam_grpo_control_formal100_2gpu32x8_g8_b2048_u2_m10_fixed32_eval5_phys67_dcp_v2
PACKET_ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets
RUN_ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs
OLD_PACKET="$PACKET_ROOT/$OLD_SLUG"
OLD_RUN="$RUN_ROOT/$OLD_SLUG"
NEW_PACKET="$PACKET_ROOT/$NEW_SLUG"
NEW_RUN="$RUN_ROOT/$NEW_SLUG"

test "$(git -C "$WT" rev-parse HEAD)" = 7b2331c55d14397cfb4cb16181470ddc8afae44a
test -z "$(git -C "$WT" status --porcelain)"
test -d "$FASTWAM_SRC/fastwam"
test -f "$OLD_PACKET/command.txt"
test -f "$OLD_PACKET/resolved.yaml"
test ! -e "$NEW_PACKET"
test ! -e "$NEW_RUN"
test "$(cat "$OLD_RUN/runtime/exit_code.txt")" != 0 || true
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null

for gpu_index in 6 7; do
  gpu_used=$(nvidia-smi -i "$gpu_index" --query-gpu=memory.used --format=csv,noheader,nounits)
  test "$gpu_used" -lt 256
done

mkdir -p "$NEW_PACKET" "$NEW_RUN/runtime"
cp -a "$OLD_PACKET/." "$NEW_PACKET/"
cp "$OLD_RUN/runtime/wrapper.sh" "$NEW_RUN/runtime/wrapper.sh"
cp "$OLD_RUN/runtime/observer.sh" "$NEW_RUN/runtime/observer.sh"

sed -i "s/$OLD_SLUG/$NEW_SLUG/g" "$NEW_PACKET/command.txt" "$NEW_PACKET/resolved.yaml"
sed -i "s/$OLD_EXPERIMENT/$NEW_EXPERIMENT/g" "$NEW_PACKET/command.txt" "$NEW_PACKET/resolved.yaml"
test "$(grep -o "$NEW_EXPERIMENT" "$NEW_PACKET/command.txt" | wc -l)" -eq 1
if grep -qE "$OLD_SLUG|$OLD_EXPERIMENT" "$NEW_PACKET/command.txt" "$NEW_PACKET/resolved.yaml"; then
  exit 1
fi

sha256sum "$NEW_PACKET/command.txt" > "$NEW_PACKET/command.sha256"
sha256sum "$NEW_PACKET/resolved.yaml" > "$NEW_PACKET/resolved.sha256"
cp "$NEW_PACKET/command.txt" "$NEW_RUN/runtime/command.txt"
cp "$NEW_PACKET/contract.json" "$NEW_RUN/runtime/contract.json"
cp "$NEW_PACKET/resolved.yaml" "$NEW_RUN/runtime/resolved.yaml"
cp "$NEW_PACKET/source_head.txt" "$NEW_RUN/runtime/source_head.txt"
cp "$NEW_PACKET/packet_complete.txt" "$NEW_RUN/runtime/packet_complete.txt"
cp "$OLD_RUN/runtime/launch_manifest.txt" "$NEW_RUN/runtime/launch_manifest.txt"
launch_time=$(date --iso-8601=seconds)
sed -i -e "s/^started_at=.*/started_at=$launch_time/" "$NEW_RUN/runtime/launch_manifest.txt"
printf '%s\n' 'runtime_fix=restored source-locked official Fast-WAM PYTHONPATH and cache variables; scientific config unchanged' >> "$NEW_RUN/runtime/launch_manifest.txt"

printf 'COMMAND_DIFF_FROM_FAILED_STARTUP\n'
diff -u "$OLD_PACKET/command.txt" "$NEW_PACKET/command.txt" || true
printf 'IMPORT_PROBE\n'
PYTHONPATH="$WT:$FASTWAM_SRC:$ROBOTWIN" "$VENV/bin/python" -c 'import fastwam, rlinf; print(fastwam.__file__); print(rlinf.__file__)'

export RAY_ADDRESS ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$FASTWAM_SRC:$ROBOTWIN"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope
export TMPDIR=/home/chenyiteng/cache/fastwam-7faa/tmp
export HF_HOME=/data/chenyiteng/cache/huggingface
export XDG_CACHE_HOME=/data/chenyiteng/cache
cd "$WT"

nohup setsid bash "$NEW_RUN/runtime/wrapper.sh" "$NEW_RUN/runtime" \
  > "$NEW_RUN/runtime/wrapper.log" 2>&1 < /dev/null &
wrapper_pid=$!
printf '%s\n' "$wrapper_pid" > "$NEW_RUN/runtime/wrapper.pid"
sleep 2
test -d "/proc/$wrapper_pid"
wrapper_pgid=$(ps -o pgid= -p "$wrapper_pid" | tr -d ' ')
printf '%s\n' "$wrapper_pgid" > "$NEW_RUN/runtime/owned.pgid"

nohup setsid bash "$NEW_RUN/runtime/observer.sh" "$wrapper_pid" "$NEW_RUN/runtime/resource.csv" \
  > "$NEW_RUN/runtime/observer.log" 2>&1 < /dev/null &
observer_pid=$!
printf '%s\n' "$observer_pid" > "$NEW_RUN/runtime/observer.pid"

printf 'LAUNCHED_V2 wrapper_pid=%s wrapper_pgid=%s observer_pid=%s\n' "$wrapper_pid" "$wrapper_pgid" "$observer_pid"
sleep 8
ps -o user,pid,ppid,pgid,etimes,rss,stat,args -p "$wrapper_pid" "$observer_pid"
tail -n 60 "$NEW_RUN/runtime/driver.log" 2>/dev/null || true
