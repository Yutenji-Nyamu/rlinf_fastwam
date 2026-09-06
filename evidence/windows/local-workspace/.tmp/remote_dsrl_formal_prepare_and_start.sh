set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
ROBOTWIN=/root/autodl-tmp/RoboTwin_RLinf
PY=/root/autodl-tmp/RLinf/.venv/bin/python
EMBODIED_PATH="$REPO/examples/embodiment"
CFG=robotwin_adjust_bottle_dsrl_openpi
EXPERIMENT=robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_formal_v1
RUN_ROOT="$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1"
STAGE="$REPO/logs/.20260728_dsrl_pi0_robotwin_n20_formal_v1.prepare.$$"
SUPPORT=/root/autodl-tmp/dsrl_formal_support_20260728
EXPECTED_HEAD=d664bf349b63b75f41d51c8295cb0a330780d783

test -x "$PY"
test -d "$ROBOTWIN"
test -s "$SUPPORT/formal_cgroup_detail_monitor.py"
test ! -e "$RUN_ROOT"
test ! -e "$STAGE"
test "$(git -C "$REPO" branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$REPO" rev-parse '@{upstream}')" = "$EXPECTED_HEAD"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
if pgrep -x raylet >/dev/null || pgrep -x gcs_server >/dev/null; then
  echo "Refusing launch: Ray process already exists" >&2
  exit 11
fi
if pgrep -af 'train_embodied_agent.py' >/dev/null; then
  echo "Refusing launch: training driver already exists" >&2
  exit 12
fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
  awk '$1 > 512 { bad=1 } END { exit bad ? 1 : 0 }'

mkdir -p "$STAGE/resource_monitor"

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export EMBODIED_PATH
export REPO_PATH="$REPO"
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$REPO:$ROBOTWIN"
export CUDA_VISIBLE_DEVICES=0,1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

CFG_ARGS=(
  --config-path "$EMBODIED_PATH/config"
  --config-name "$CFG"
  "runner.logger.log_path=$RUN_ROOT"
  "runner.logger.experiment_name=$EXPERIMENT"
  "runner.max_steps=650"
  "runner.val_check_interval=13"
  "runner.save_interval=65"
  "runner.resume_dir=null"
  "runner.ckpt_path=null"
  "env.train.video_cfg.save_video=false"
  "env.train.video_cfg.video_base_dir=$RUN_ROOT/video/train"
  "env.train.task_config.save_path=$RUN_ROOT/robotwin_data/train"
  "env.eval.video_cfg.save_video=false"
  "env.eval.video_cfg.video_base_dir=$RUN_ROOT/video/eval"
  "env.eval.task_config.save_path=$RUN_ROOT/robotwin_data/eval"
)
FORMAL_CMD=(
  "$PY" -B "$EMBODIED_PATH/train_embodied_agent.py"
  "${CFG_ARGS[@]}"
)
RESOLVED="$STAGE/FORMAL_RUN_VALIDATED_RESOLVED_20260728.yaml"

"${FORMAL_CMD[@]}" --cfg job --resolve > "$RESOLVED" \
  2> "$STAGE/formal_resolve_stderr.log"
"$PY" -B -c '
import sys
from omegaconf import OmegaConf
c = OmegaConf.load(sys.argv[1])
r = sys.argv[2]
assert c.runner.max_steps == 650
assert c.runner.val_check_interval == 13
assert c.runner.save_interval == 65
assert c.runner.resume_dir is None and c.runner.ckpt_path is None
assert c.runner.logger.log_path == r
assert c.runner.logger.experiment_name == "robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_formal_v1"
assert c.env.train.total_num_envs == 4 and c.env.eval.total_num_envs == 4
assert c.env.train.rollout_epoch == 1 and c.env.eval.rollout_epoch == 3
assert c.env.train.use_fixed_reset_state_ids is False
assert c.env.eval.use_fixed_reset_state_ids is False
assert c.actor.model.num_action_chunks == 20
assert c.actor.model.openpi.action_horizon == 50
assert c.actor.model.openpi.dsrl_state_dim == 14
assert c.actor.model.openpi.dsrl_action_noise_dim == 32
assert c.algorithm.utd_ratio == 20
assert c.algorithm.replay_buffer.capacity == 25000
assert c.algorithm.replay_buffer.warmup_size == 500
assert c.env.train.task_config.save_path == f"{r}/robotwin_data/train"
assert c.env.eval.task_config.save_path == f"{r}/robotwin_data/eval"
' "$RESOLVED" "$RUN_ROOT"

printf '%q ' "${FORMAL_CMD[@]}" > "$STAGE/formal_command.txt"
printf '\n' >> "$STAGE/formal_command.txt"
printf '%s\n' "${CFG_ARGS[@]}" > "$STAGE/formal_overrides.txt"
{
  echo "prepared_at=$(date --iso-8601=seconds)"
  echo "identity=$(id -un)@$(hostname)"
  echo "repo=$REPO"
  echo "branch=$(git -C "$REPO" branch --show-current)"
  echo "head=$(git -C "$REPO" rev-parse HEAD)"
  echo "upstream=$(git -C "$REPO" rev-parse '@{upstream}')"
  echo "source_config=$EMBODIED_PATH/config/$CFG.yaml"
  echo "source_config_sha256=$(sha256sum "$EMBODIED_PATH/config/$CFG.yaml" | awk '{print $1}')"
  echo "resolved_config_sha256=$(sha256sum "$RESOLVED" | awk '{print $1}')"
  echo "python=$PY"
  echo "python_version=$("$PY" --version 2>&1)"
  echo "run_root=$RUN_ROOT"
  echo "budget_max_steps=650"
  echo "eval_interval_cycles=13"
  echo "save_interval_cycles=65"
  echo "train_envs=4"
  echo "eval_episodes_per_check=12"
  echo "requested_action_chunks=20"
} > "$STAGE/run_provenance.txt"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
  printf 'memory.current='
  cat /sys/fs/cgroup/memory.current
  printf 'memory.max='
  cat /sys/fs/cgroup/memory.max
  cat /sys/fs/cgroup/memory.events
  grep -E '^(anon|file|inactive_file|active_file|shmem|slab|kernel_stack|pagetables) ' \
    /sys/fs/cgroup/memory.stat
  cat /sys/fs/cgroup/memory.pressure
  df -h /root/autodl-tmp /dev/shm
} > "$STAGE/prelaunch_resource_snapshot.txt"

mv "$STAGE" "$RUN_ROOT"
cd "$RUN_ROOT"

nohup "${FORMAL_CMD[@]}" > "$RUN_ROOT/formal_driver.log" 2>&1 < /dev/null &
FORMAL_PID=$!
printf '%s\n' "$FORMAL_PID" > "$RUN_ROOT/formal.pid"
printf '%s\n' "$(date --iso-8601=seconds)" > "$RUN_ROOT/formal_started_at.txt"

nohup "$PY" -B "$REPO/examples/embodiment/monitor_resources.py" \
  --pid "$FORMAL_PID" \
  --out-dir "$RUN_ROOT/resource_monitor" \
  --interval 2 \
  > "$RUN_ROOT/resource_monitor/monitor.log" 2>&1 < /dev/null &
MONITOR_PID=$!
printf '%s\n' "$MONITOR_PID" > "$RUN_ROOT/resource_monitor/monitor.pid"

nohup "$PY" -B "$SUPPORT/formal_cgroup_detail_monitor.py" \
  --pid "$FORMAL_PID" \
  --out "$RUN_ROOT/resource_monitor/cgroup_detail.csv" \
  --interval 2 \
  > "$RUN_ROOT/resource_monitor/cgroup_detail_monitor.log" 2>&1 < /dev/null &
DETAIL_MONITOR_PID=$!
printf '%s\n' "$DETAIL_MONITOR_PID" \
  > "$RUN_ROOT/resource_monitor/cgroup_detail_monitor.pid"

echo "FORMAL_STARTED=1"
echo "FORMAL_PID=$FORMAL_PID"
echo "MONITOR_PID=$MONITOR_PID"
echo "DETAIL_MONITOR_PID=$DETAIL_MONITOR_PID"
echo "RUN_ROOT=$RUN_ROOT"
echo "RESOLVED_SHA256=$(sha256sum "$RUN_ROOT/FORMAL_RUN_VALIDATED_RESOLVED_20260728.yaml" | awk '{print $1}')"
echo "STARTED_AT=$(cat "$RUN_ROOT/formal_started_at.txt")"
