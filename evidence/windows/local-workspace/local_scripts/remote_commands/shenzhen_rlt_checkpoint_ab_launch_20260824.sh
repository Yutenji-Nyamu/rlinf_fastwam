#!/usr/bin/env bash
set -euo pipefail

: "${CASE_NAME:?set CASE_NAME}"
: "${RUN_ROOT:?set RUN_ROOT}"
: "${WARMUP_MIN_SIZE:?set WARMUP_MIN_SIZE}"
: "${WARMUP_POST_COLLECT_UPDATES:?set WARMUP_POST_COLLECT_UPDATES}"

worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-checkpoint-diagnosis-7d07a421
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
model=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
norm_stats=$model/physical-intelligence/robotwin/norm_stats.json
stage1_root=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2/stage1
stage1_checkpoint=$stage1_root/robotwin_adjust_bottle_rlt_stage1_current_ar_clean50_2k_v1/checkpoints/global_step_2000
stage1_manifest=$stage1_root/artifacts/stage1_artifact_manifest.json
stage1_manifest_id=sz-rlt-stage1-current-ar-clean50-2k-v1
runtime=$RUN_ROOT/runtime
experiment=robotwin_adjust_bottle_rlt_ckpt_diag_$CASE_NAME
observer=/data/chenyiteng/results/rlinf-rlt/launchers/shenzhen_rlt_current_smoke_observer_gpu4_5_20260823.sh
ray_address=172.17.0.1:6389
expected_head=f3ea5f691b99fe39e024e5571c0e6ee3d83c51b4
expected_branch=codex/sz-rlt-checkpoint-diagnosis

test "$(git -C "$worktree" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$worktree" branch --show-current)" = "$expected_branch"
git -C "$worktree" diff --check
test -z "$(git -C "$worktree" ls-files --others --exclude-standard)"
test -s "$stage1_checkpoint/actor/model_state_dict/full_weights.pt"
test -s "$stage1_checkpoint/actor/dcp_checkpoint/.metadata"
test -s "$stage1_manifest"
test -s "$norm_stats"
test -x "$venv/bin/python"
test -x "$venv/bin/ray"
test -d "$robotwin"
test -x "$observer"
test ! -e "$RUN_ROOT"
RAY_ADDRESS="$ray_address" "$venv/bin/ray" status >/dev/null
if nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'physical GPU 4-5 are not idle' >&2
  exit 1
fi
RAY_ADDRESS="$ray_address" "$venv/bin/python" - <<'PY'
import os
import ray

ray.init(
    address=os.environ["RAY_ADDRESS"],
    namespace="codex_rlt_ckpt_ab_preflight",
    logging_level="ERROR",
)
names = sorted(
    row["name"]
    for row in ray.util.list_named_actors(all_namespaces=True)
    if row.get("namespace") == "RLinf"
)
ray.shutdown()
if names:
    raise SystemExit(f"RLinf namespace is not empty: {names}")
PY

source "$venv/bin/activate"
unset CUDA_VISIBLE_DEVICES
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS="$ray_address"
export ROBOTWIN_PI0_BASE_PATH="$model"
export ROBOTWIN_PATH="$robotwin"
export ROBOTWIN_ASSETS_PATH="$robotwin"
export ROBOT_PLATFORM=ALOHA
export REPO_PATH="$worktree"
export EMBODIED_PATH="$worktree/examples/embodiment"
export PYTHONPATH="$worktree:$robotwin${PYTHONPATH:+:$PYTHONPATH}"
export RLINF_CODE_WORKING_DIR="$worktree"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export RLT_LOG_ROOT="$RUN_ROOT"
export RLT_STAGE1_MODEL_PATH="$stage1_checkpoint"
export RLT_STAGE1_MANIFEST_PATH="$stage1_manifest"
export RLT_STAGE1_MANIFEST_ID="$stage1_manifest_id"
export RLT_STAGE1_MANIFEST_SHA256
RLT_STAGE1_MANIFEST_SHA256=$(sha256sum "$stage1_manifest" | awk '{print $1}')
export RLT_NORM_STATS_SHA256
RLT_NORM_STATS_SHA256=$(sha256sum "$norm_stats" | awk '{print $1}')
export ROBOTWIN_PI0_NORM_STATS_PATH="$norm_stats"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export RLINF_CKPT_DIAG=1

mkdir -p "$runtime"

args=(
  --config-path "$worktree/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env250
  'cluster.component_placement={actor\,env\,rollout:4-5}'
  "runner.logger.log_path=$RUN_ROOT"
  "runner.logger.experiment_name=$experiment"
  'runner.max_steps=1'
  'runner.val_check_interval=-1'
  'runner.save_interval=1'
  'runner.resume_dir=null'
  'runner.ckpt_path=null'
  'algorithm.rlt_schedule.max_updates_per_train_step=1'
  "algorithm.rlt_schedule.warmup_min_size=$WARMUP_MIN_SIZE"
  "algorithm.rlt_schedule.warmup_post_collect_updates=$WARMUP_POST_COLLECT_UPDATES"
  'env.train.total_num_envs=4'
  'env.train.rollout_epoch=1'
  'env.train.max_episode_steps=20'
  'env.train.max_steps_per_rollout_epoch=20'
  'env.train.video_cfg.save_video=false'
  "env.train.video_cfg.video_base_dir=$RUN_ROOT/video/train"
  "env.train.task_config.save_path=$RUN_ROOT/robotwin_data/train"
  'env.eval.total_num_envs=4'
  'env.eval.rollout_epoch=1'
  'env.eval.max_episode_steps=20'
  'env.eval.max_steps_per_rollout_epoch=20'
  'env.eval.video_cfg.save_video=false'
  "env.eval.video_cfg.video_base_dir=$RUN_ROOT/video/eval"
  "env.eval.task_config.save_path=$RUN_ROOT/robotwin_data/eval"
  'actor.micro_batch_size=128'
  'actor.global_batch_size=512'
)

"$venv/bin/python" "$worktree/examples/embodiment/train_embodied_agent.py" \
  "${args[@]}" --cfg job --resolve > "$runtime/resolved.yaml"
sha256sum "$runtime/resolved.yaml" > "$runtime/resolved.yaml.sha256"
printf '%q ' "$venv/bin/python" "$worktree/examples/embodiment/train_embodied_agent.py" \
  "${args[@]}" > "$runtime/command.txt"
printf '\n' >> "$runtime/command.txt"
printf '%s\n' \
  "prepared_at=$(date --iso-8601=seconds)" \
  "case=$CASE_NAME" \
  "source_head=$expected_head" \
  "source_diff=$(git -C "$worktree" diff --stat | tr '\n' ';')" \
  "shared_ray_address=$ray_address" \
  'physical_gpus=4,5; actor_world_size=2' \
  'train=4 env x 1 cycle; max 8 transitions; H50/C10/D14' \
  "warmup_min_size=$WARMUP_MIN_SIZE" \
  "warmup_post_collect_updates=$WARMUP_POST_COLLECT_UPDATES" \
  'max_updates_per_train_step=1' \
  'eval=disabled; save=cycle1' \
  'normal_stop=checkpoint global_step_1 and exit0' \
  'failure_stop=nonzero exit or 1800s hard timeout' \
  > "$runtime/launch_manifest.txt"

nohup setsid bash -c '
  runtime=$1
  shift
  date --iso-8601=seconds > "$runtime/started_at.txt"
  timeout --signal=TERM --kill-after=120s 1800s "$@" > "$runtime/driver.log" 2>&1
  rc=$?
  printf "%s\n" "$rc" > "$runtime/exit_code.txt"
  date --iso-8601=seconds > "$runtime/finished_at.txt"
  exit "$rc"
' _ \
  "$runtime" \
  "$venv/bin/python" \
  "$worktree/examples/embodiment/train_embodied_agent.py" \
  "${args[@]}" \
  > "$runtime/wrapper.log" 2>&1 < /dev/null &
wrapper_pid=$!
printf '%s\n' "$wrapper_pid" > "$runtime/wrapper.pid"
printf '%s\n' "$wrapper_pid" > "$runtime/owned.pgid"

nohup setsid bash "$observer" "$wrapper_pid" "$runtime/resource.csv" \
  > "$runtime/resource_observer.log" 2>&1 < /dev/null &
observer_pid=$!
printf '%s\n' "$observer_pid" > "$runtime/resource_observer.pid"

sleep 12
if ! kill -0 "$wrapper_pid" 2>/dev/null; then
  tail -n 100 "$runtime/driver.log" >&2 || true
  exit 1
fi
kill -0 "$observer_pid"

printf 'case=%s\nrun_root=%s\nwrapper_pid=%s\nobserver_pid=%s\nresolved_sha256=%s\n' \
  "$CASE_NAME" "$RUN_ROOT" "$wrapper_pid" "$observer_pid" \
  "$(awk '{print $1}' "$runtime/resolved.yaml.sha256")"
tail -n 40 "$runtime/driver.log" || true
printf '%s\n' 'SZ_RLT_CHECKPOINT_AB_LAUNCHED'
