set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney
HEAD=639444db8ad7bc9c934e056ad8511139ed94eba9
OLD_P_RUN="$ROOT/smokes/parity-core224-m10-phys4-v7"
OLD_A="$ROOT/packets/b1-adjust-badseed-retry-m10-phys4-v10"
OLD_B="$ROOT/packets/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v10"
P_NAME=parity-core224-m10-compare-existing-v8
A_NAME=b1-adjust-badseed-retry-m10-phys4-v11
B_NAME=move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v11
P_PACKET="$ROOT/packets/$P_NAME"
A_PACKET="$ROOT/packets/$A_NAME"
B_PACKET="$ROOT/packets/$B_NAME"
P_RUN="$ROOT/smokes/$P_NAME"
A_RUN="$ROOT/smokes/$A_NAME"
B_RUN="$ROOT/smokes/$B_NAME"

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -s "$OLD_P_RUN/input.npz"
test -s "$OLD_P_RUN/native.pt"
test -s "$OLD_P_RUN/rlinf.pt"
test -s "$OLD_A/command.txt"
test -s "$OLD_B/command.txt"
for path in "$P_PACKET" "$A_PACKET" "$B_PACKET" "$P_RUN" "$A_RUN" "$B_RUN"; do
  test ! -e "$path"
done

RAY_ADDRESS=172.17.0.1:6389 "$VENV/bin/ray" status > /tmp/sidney_ray_status_compare_v8_20260903.txt
mkdir -p "$P_PACKET" "$A_PACKET" "$B_PACKET" "$P_RUN/runtime"
for packet in "$P_PACKET" "$A_PACKET" "$B_PACKET"; do
  cp /tmp/sidney_ray_status_compare_v8_20260903.txt "$packet/ray-status-before.txt"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits > "$packet/gpu-before.csv"
  awk '/MemTotal:|MemAvailable:/ {print}' /proc/meminfo > "$packet/memory-before.txt"
  df -h / /home /data > "$packet/disk-before.txt"
  printf '%s\n' "$HEAD" > "$packet/source-head.txt"
  printf '%s\n' prepared > "$packet/packet-complete.txt"
done

cat > "$P_PACKET/command.txt" <<EOF
$VENV/bin/python $WT/toolkits/lerobot/sidney_pi05_parity.py compare --native $OLD_P_RUN/native.pt --rlinf $OLD_P_RUN/rlinf.pt --rtol 1e-2 --atol 5e-3 > $P_RUN/report.json
EOF
sha256sum "$OLD_P_RUN/input.npz" "$OLD_P_RUN/native.pt" "$OLD_P_RUN/rlinf.pt" > "$P_PACKET/input-artifact-sha256.txt"
cat > "$P_PACKET/contract.json" <<'EOF'
{
  "kind": "CPU-only recompare of existing 224x224 core parity artifacts",
  "source_artifacts": "parity-core224-m10-phys4-v7",
  "processor_tolerances": {"images_atol": 1e-6, "state_atol": 1e-5, "tokens": "exact"},
  "action_tolerances": {"rtol": 0.01, "atol": 0.005},
  "production_code_changed": false
}
EOF

cp "$OLD_A/eval-seeds-adjust-bad1001.json" "$A_PACKET/eval-seeds-adjust-bad1001.json"
sed -e 's/b1-adjust-badseed-retry-m10-phys4-v10/b1-adjust-badseed-retry-m10-phys4-v11/g' \
    -e 's/pi05_sidney_b1_adjust_badseed_retry_m10_phys4_v10/pi05_sidney_b1_adjust_badseed_retry_m10_phys4_v11/g' \
    "$OLD_A/command.txt" > "$A_PACKET/command.txt"
sed -e 's/b1-adjust-badseed-retry-m10-phys4-v10/b1-adjust-badseed-retry-m10-phys4-v11/g' \
    -e 's/pi05_sidney_b1_adjust_badseed_retry_m10_phys4_v10/pi05_sidney_b1_adjust_badseed_retry_m10_phys4_v11/g' \
    "$OLD_A/resolved.yaml" > "$A_PACKET/resolved.yaml"
cp "$OLD_A/contract.json" "$A_PACKET/contract.json"
cp "$OLD_A/model-sha256.txt" "$A_PACKET/model-sha256.txt"

sed -e 's/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v10/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v11/g' \
    -e 's/pi05_sidney_move_grpo_smoke1_2gpu64x1_g8_b512_u2_m10_fixed8_phys45_localshard_v10/pi05_sidney_move_grpo_smoke1_2gpu64x1_g8_b512_u2_m10_fixed8_phys45_localshard_v11/g' \
    "$OLD_B/command.txt" > "$B_PACKET/command.txt"
sed -e 's/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v10/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v11/g' \
    -e 's/pi05_sidney_move_grpo_smoke1_2gpu64x1_g8_b512_u2_m10_fixed8_phys45_localshard_v10/pi05_sidney_move_grpo_smoke1_2gpu64x1_g8_b512_u2_m10_fixed8_phys45_localshard_v11/g' \
    "$OLD_B/resolved.yaml" > "$B_PACKET/resolved.yaml"
cp "$OLD_B/contract.json" "$B_PACKET/contract.json"
cp "$OLD_B/model-sha256.txt" "$B_PACKET/model-sha256.txt"

cat > "$P_PACKET/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PACKET=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/parity-core224-m10-compare-existing-v8
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-core224-m10-compare-existing-v8
test "$(git -C "$WT" rev-parse HEAD)" = 639444db8ad7bc9c934e056ad8511139ed94eba9
test -z "$(git -C "$WT" status --porcelain)"
set +e
bash -e "$PACKET/command.txt" > "$RUN/runtime/wrapper.log" 2>&1
RC=$?
set -e
printf '%s\n' "$RC" > "$RUN/runtime/exit_code.txt"
exit "$RC"
EOF

cat > "$A_PACKET/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
PACKET=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/b1-adjust-badseed-retry-m10-phys4-v11
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/b1-adjust-badseed-retry-m10-phys4-v11
test "$(git -C "$WT" rev-parse HEAD)" = 639444db8ad7bc9c934e056ad8511139ed94eba9
test -z "$(git -C "$WT" status --porcelain)"
if nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'physical GPU4 is occupied; refusing Sidney B=1 smoke' >&2
  exit 20
fi
mkdir -p "$RUN/runtime"
cd "$WT"
source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS=172.17.0.1:6389
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
monitor() { while true; do date --iso-8601=seconds; nvidia-smi -i 4 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; awk '/MemAvailable:/ {print}' /proc/meminfo; sleep 30; done; }
monitor > "$RUN/runtime/resources.log" 2>&1 &
MONITOR_PID=$!
cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT
set +e
bash -e "$PACKET/command.txt" > "$RUN/runtime/driver.log" 2>&1
RC=$?
set -e
printf '%s\n' "$RC" > "$RUN/runtime/exit_code.txt"
exit "$RC"
EOF

cat > "$B_PACKET/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
PACKET=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v11
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v11
test "$(git -C "$WT" rev-parse HEAD)" = 639444db8ad7bc9c934e056ad8511139ed94eba9
test -z "$(git -C "$WT" status --porcelain)"
if nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'physical GPU4/5 is occupied; refusing Sidney GRPO smoke' >&2
  exit 20
fi
mkdir -p "$RUN/runtime"
cd "$WT"
source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS=172.17.0.1:6389
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
monitor() { while true; do date --iso-8601=seconds; nvidia-smi -i 4,5 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; awk '/MemAvailable:/ {print}' /proc/meminfo; sleep 30; done; }
monitor > "$RUN/runtime/resources.log" 2>&1 &
MONITOR_PID=$!
cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT
set +e
bash -e "$PACKET/command.txt" > "$RUN/runtime/driver.log" 2>&1
RC=$?
set -e
printf '%s\n' "$RC" > "$RUN/runtime/exit_code.txt"
exit "$RC"
EOF

chmod 0755 "$P_PACKET/run.sh" "$A_PACKET/run.sh" "$B_PACKET/run.sh"
sha256sum "$P_PACKET/run.sh" > "$P_PACKET/run-sha256.txt"
sha256sum "$A_PACKET/run.sh" > "$A_PACKET/run-sha256.txt"
sha256sum "$B_PACKET/run.sh" > "$B_PACKET/run-sha256.txt"
cat "$P_PACKET/run-sha256.txt" "$A_PACKET/run-sha256.txt" "$B_PACKET/run-sha256.txt"
echo SIDNEY_COMPARE_V8_AB_V11_PACKET_PREPARED
