#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
PYTHON=/root/autodl-tmp/RLinf/.venv/bin/python
DATASET=/root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-v1
MANIFEST=/root/autodl-tmp/datasets/robotwin2/manifests/pi0-aloha-clean50-v1.json
MODEL=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
STATS=${MODEL}/physical-intelligence/robotwin/norm_stats.json
RUN_ROOT=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1
EXPORT_ROOT=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1
RUNTIME_ROOT=${EXPORT_ROOT}/runtime
NAME=robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1
MONITOR=/root/autodl-tmp/tmp/rlt_stage1_resource_monitor_20260729.sh
EXPECTED_HEAD=4ac48d54c63b3a83d99f551fb54f738297525acf
EXPECTED_CONFIG_SHA=c293bc476ec7458c6bfc5c5c59393e48b286f3e12007f3039ccc282e30645a4c
EXPECTED_RESOLVED_SHA=5aa824fc9ac5cc361dace2b1162b2ef1bdf52adab3c775b8cd2e1ae468dfd67e
EXPECTED_MANIFEST_SHA=12ce2ed68632e2b18cf96f52b717edec00bcebb6cc0a446f83da1670d81ef86c
EXPECTED_STATS_SHA=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

cd "$RLT_ROOT"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git status --porcelain)"
test "$(git rev-list --left-right --count HEAD...@{upstream})" = $'0\t0'
test "$(sha256sum examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml | awk '{print $1}')" = "$EXPECTED_CONFIG_SHA"
test "$(sha256sum "$EXPORT_ROOT/formal_resolved.yaml" | awk '{print $1}')" = "$EXPECTED_RESOLVED_SHA"
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$EXPECTED_MANIFEST_SHA"
test "$(sha256sum "$STATS" | awk '{print $1}')" = "$EXPECTED_STATS_SHA"
test -d "$DATASET"
test -f "$MONITOR"
test ! -e "$RUN_ROOT"
test ! -L "$RUN_ROOT"
test ! -e "$RUNTIME_ROOT"
test ! -L "$RUNTIME_ROOT"

if pgrep -af 'train_vla_sft|ray::|raylet|gcs_server' | grep -v -E 'pgrep -af|start_stage1_formal' >/dev/null; then
  printf 'REFUSING_EXISTING_PROCESS\n'
  pgrep -af 'train_vla_sft|ray::|raylet|gcs_server' || true
  exit 40
fi
while IFS=, read -r index used util; do
  used=${used// /}
  util=${util// /}
  test "$used" -eq 0
  test "$util" -eq 0
done < <(
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
)
available_bytes=$(df -B1 --output=avail /root/autodl-tmp | tail -n 1)
test "$available_bytes" -ge 107374182400

mkdir -p "$RUNTIME_ROOT"
cat > "$RUNTIME_ROOT/run_foreground.sh" <<EOF
#!/usr/bin/env bash
set +e
cd "$RLT_ROOT"
export PYTHONPATH="$RLT_ROOT:/root/autodl-tmp/RoboTwin_RLinf"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0,1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export ROBOTWIN_RLT_CLEAN50_PATH="$DATASET"
export ROBOTWIN_PI0_BASE_PATH="$MODEL"
export ROBOTWIN_PI0_NORM_STATS_PATH="$STATS"
export RLT_LOG_ROOT="$RUN_ROOT"
date --iso-8601=seconds > "$RUNTIME_ROOT/started_at.txt"
timeout --signal=TERM --kill-after=120s 64800s \
  "$PYTHON" -B examples/sft/train_vla_sft.py \
    --config-path "$RLT_ROOT/examples/sft/config" \
    --config-name robotwin_rlt_stage1_sft_openpi \
    runner.logger.experiment_name="$NAME"
rc=\$?
printf '%s\n' "\$rc" > "$RUNTIME_ROOT/exit_code.txt"
date --iso-8601=seconds > "$RUNTIME_ROOT/finished_at.txt"
exit "\$rc"
EOF
chmod 700 "$RUNTIME_ROOT/run_foreground.sh"

{
  printf 'launched_at\t%s\n' "$(date --iso-8601=seconds)"
  printf 'branch\t%s\n' "$(git branch --show-current)"
  printf 'head\t%s\n' "$(git rev-parse HEAD)"
  printf 'dataset\t%s\n' "$DATASET"
  printf 'dataset_manifest_sha256\t%s\n' "$EXPECTED_MANIFEST_SHA"
  printf 'source_config_sha256\t%s\n' "$EXPECTED_CONFIG_SHA"
  printf 'resolved_config_sha256\t%s\n' "$EXPECTED_RESOLVED_SHA"
  printf 'norm_stats_sha256\t%s\n' "$EXPECTED_STATS_SHA"
  printf 'run_root\t%s\n' "$RUN_ROOT"
  printf 'experiment_name\t%s\n' "$NAME"
  printf 'checkpoint\t%s/%s/checkpoints/global_step_2000\n' "$RUN_ROOT" "$NAME"
  printf 'timeout_seconds\t64800\n'
} > "$EXPORT_ROOT/run_provenance.tsv"

nohup "$RUNTIME_ROOT/run_foreground.sh" \
  > "$RUNTIME_ROOT/driver.log" \
  2>&1 \
  < /dev/null &
driver_pid=$!
printf '%s\n' "$driver_pid" > "$RUNTIME_ROOT/driver_pid.txt"

nohup bash "$MONITOR" \
  "$driver_pid" \
  "$RUNTIME_ROOT/resources.csv" \
  > "$RUNTIME_ROOT/monitor.log" \
  2>&1 \
  < /dev/null &
monitor_pid=$!
printf '%s\n' "$monitor_pid" > "$RUNTIME_ROOT/monitor_pid.txt"

sleep 2
kill -0 "$driver_pid"
printf 'DRIVER_PID\t%s\n' "$driver_pid"
printf 'MONITOR_PID\t%s\n' "$monitor_pid"
printf 'RUNTIME_ROOT\t%s\n' "$RUNTIME_ROOT"
printf 'RUN_ROOT\t%s\n' "$RUN_ROOT"
printf 'CHECKPOINT\t%s/%s/checkpoints/global_step_2000\n' "$RUN_ROOT" "$NAME"
