set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney
P_PACKET="$ROOT/packets/parity-m10-phys4-v6-chw"
A_PACKET="$ROOT/packets/b1-adjust-badseed-retry-m10-phys4-v9"
B_PACKET="$ROOT/packets/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v9"

for packet in "$P_PACKET" "$A_PACKET" "$B_PACKET"; do
  test "$(cat "$packet/source-head.txt")" = 92fea8f72271b1ff37e58f3433a6fb64abe316cc
  test "$(cat "$packet/packet-complete.txt")" = prepared
  test ! -e "$packet/run.sh"
done

cat > "$P_PACKET/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PACKET=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/parity-m10-phys4-v6-chw
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-m10-phys4-v6-chw
test "$(git -C "$WT" rev-parse HEAD)" = 92fea8f72271b1ff37e58f3433a6fb64abe316cc
test -z "$(git -C "$WT" status --porcelain)"
if nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'physical GPU4 is occupied; refusing Sidney parity' >&2
  exit 20
fi
mkdir -p "$RUN/runtime"
monitor() {
  while true; do
    date --iso-8601=seconds
    nvidia-smi -i 4 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    awk '/MemAvailable:/ {print}' /proc/meminfo
    sleep 30
  done
}
monitor > "$RUN/runtime/resources.log" 2>&1 &
MONITOR_PID=$!
cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT
set +e
bash -e "$PACKET/commands.txt" > "$RUN/runtime/wrapper.log" 2>&1
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
PACKET=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/b1-adjust-badseed-retry-m10-phys4-v9
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/b1-adjust-badseed-retry-m10-phys4-v9
test "$(git -C "$WT" rev-parse HEAD)" = 92fea8f72271b1ff37e58f3433a6fb64abe316cc
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
monitor() {
  while true; do
    date --iso-8601=seconds
    nvidia-smi -i 4 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    awk '/MemAvailable:/ {print}' /proc/meminfo
    sleep 30
  done
}
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
PACKET=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v9
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v9
test "$(git -C "$WT" rev-parse HEAD)" = 92fea8f72271b1ff37e58f3433a6fb64abe316cc
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
monitor() {
  while true; do
    date --iso-8601=seconds
    nvidia-smi -i 4,5 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    awk '/MemAvailable:/ {print}' /proc/meminfo
    sleep 30
  done
}
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
