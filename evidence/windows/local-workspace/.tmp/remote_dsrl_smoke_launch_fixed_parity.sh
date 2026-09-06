set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
PY=/root/autodl-tmp/RLinf/.venv/bin/python
ROBOTWIN=/root/autodl-tmp/RoboTwin_RLinf
RUN_ROOT="$REPO/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1"
SCRIPT="$RUN_ROOT/fixed_latent_parity_probe_v2.py"
OUT="$RUN_ROOT/fixed_latent_parity_v2"

test "$(git -C "$REPO" rev-parse HEAD)" = 3c7d35d716cfd6964b15249e548a0aab99d4cb27
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -x "$PY"
test -s "$RUN_ROOT/FRESH_SMOKE_VALIDATED_RESOLVED_20260728.yaml"
test ! -e "$OUT"
echo "b12b4c78592bd5a913c76de5b10c25d710e4ee14e04f4f5ac7e9f706655e539d  $SCRIPT" | sha256sum -c -
if pgrep -x raylet || pgrep -x gcs_server; then
  echo "TARGET_PROCESSES_PRESENT=1"
  exit 72
fi
if pgrep -af '^/root/autodl-tmp/RLinf/.venv/bin/python .*train_embodied_agent.py|^/root/autodl-tmp/RLinf/.venv/bin/python .*monitor_resources.py|^ray::EmbodiedSACFSDPPolicy|^ray::MultiStepRolloutWorker|^ray::EnvWorker'; then
  echo "TARGET_PROCESSES_PRESENT=1"
  exit 72
fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"

mkdir -p "$OUT/resource_monitor"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:$ROBOTWIN"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

CMD=("$PY" -B "$SCRIPT")
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
nohup "${CMD[@]}" > "$OUT/driver.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$OUT/driver.pid"

nohup "$PY" -B "$REPO/examples/embodiment/monitor_resources.py" \
  --pid "$pid" \
  --out-dir "$OUT/resource_monitor" \
  --interval 2 \
  > "$OUT/resource_monitor/monitor.log" 2>&1 < /dev/null &
monitor_pid=$!
printf '%s\n' "$monitor_pid" > "$OUT/resource_monitor/monitor.pid"
echo "PARITY_PID=$pid"
echo "PARITY_MONITOR_PID=$monitor_pid"
