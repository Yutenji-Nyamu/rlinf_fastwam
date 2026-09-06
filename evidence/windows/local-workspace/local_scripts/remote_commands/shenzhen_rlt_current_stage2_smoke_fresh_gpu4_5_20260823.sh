#!/usr/bin/env bash
set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
model=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
norm_stats=$model/physical-intelligence/robotwin/norm_stats.json
stage1_root=/data/chenyiteng/results/rlinf-rlt/smoke-stage1-current-ar-2step-20260823
stage1_runtime=$stage1_root/runtime
stage1_experiment=robotwin_adjust_bottle_rlt_stage1_current_ar_smoke2_v1
stage1_checkpoint=$stage1_root/$stage1_experiment/checkpoints/global_step_2
stage1_manifest=$stage1_root/artifacts/stage1_artifact_manifest.json
stage1_manifest_id=sz-rlt-stage1-current-ar-smoke2-v1
run_root=/data/chenyiteng/results/rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823
fresh_log=$run_root/fresh
experiment=robotwin_adjust_bottle_rlt_stage2_current_ar_smoke_fresh1_v1
observer=/data/chenyiteng/results/rlinf-rlt/launchers/shenzhen_rlt_current_smoke_observer_gpu4_5_20260823.sh
expected_head=bdd875283b3f3516c439e5c79c902cf5c2da58b6
expected_branch=codex/sz-rlt-pi0-robotwin-ar
expected_config_sha=c6a6499fedad81f054e5aac06a6745dbddf4e75e361a613f058a43a8156b255a
expected_norm_sha=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a
config=$worktree/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current.yaml

test "$(git -C "$worktree" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$worktree" branch --show-current)" = "$expected_branch"
test "$(git -C "$worktree" rev-parse "personal/$expected_branch")" = "$expected_head"
test -z "$(git -C "$worktree" status --porcelain)"
test "$(sha256sum "$config" | awk '{print $1}')" = "$expected_config_sha"
test "$(sha256sum "$norm_stats" | awk '{print $1}')" = "$expected_norm_sha"
test "$(cat "$stage1_runtime/exit_code.txt")" = 0
test -s "$stage1_checkpoint/actor/model_state_dict/full_weights.pt"
test -s "$stage1_checkpoint/actor/dcp_checkpoint/.metadata"
test -s "$stage1_manifest"
test -x "$venv/bin/python"
test -d "$robotwin"
test -r "$observer"
test ! -e "$run_root"

# RLT Stage2 and DSRL are both Ray jobs. Never overlap their control planes.
if pgrep -u "$(id -u)" -x raylet >/dev/null || \
  pgrep -u "$(id -u)" -x gcs_server >/dev/null; then
  printf '%s\n' 'chenyiteng still has a live Ray cluster; wait for DSRL to exit' >&2
  exit 1
fi
if nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'physical GPU 4-5 are not idle' >&2
  exit 1
fi

source "$venv/bin/activate"
unset CUDA_VISIBLE_DEVICES
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export ROBOTWIN_PATH="$robotwin"
export ROBOTWIN_ASSETS_PATH="$robotwin"
export ROBOT_PLATFORM=ALOHA
export REPO_PATH="$worktree"
export EMBODIED_PATH="$worktree/examples/embodiment"
export PYTHONPATH="$worktree:$robotwin${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export RLT_LOG_ROOT="$fresh_log"
export RLT_STAGE1_MODEL_PATH="$stage1_checkpoint"
export RLT_STAGE1_MANIFEST_PATH="$stage1_manifest"
export RLT_STAGE1_MANIFEST_ID="$stage1_manifest_id"
export RLT_STAGE1_MANIFEST_SHA256
RLT_STAGE1_MANIFEST_SHA256=$(sha256sum "$stage1_manifest" | awk '{print $1}')
export RLT_NORM_STATS_SHA256="$expected_norm_sha"
export ROBOTWIN_PI0_NORM_STATS_PATH="$norm_stats"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$fresh_log"

args=(
  --config-path "$worktree/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_current
  'cluster.component_placement={actor\,env\,rollout:4-5}'
  "runner.logger.log_path=$fresh_log"
  "runner.logger.experiment_name=$experiment"
  'runner.max_steps=1'
  'runner.val_check_interval=1'
  'runner.save_interval=1'
  'runner.resume_dir=null'
  'runner.ckpt_path=null'
  'algorithm.rlt_schedule.max_updates_per_train_step=20'
  'algorithm.rlt_schedule.warmup_min_size=2'
  'algorithm.rlt_schedule.warmup_post_collect_updates=8'
  'algorithm.actor_weight_schedule.warmup_updates=4'
  'algorithm.actor_weight_schedule.ramp_updates=8'
  'env.train.total_num_envs=4'
  'env.train.rollout_epoch=1'
  'env.train.max_episode_steps=20'
  'env.train.max_steps_per_rollout_epoch=20'
  'env.eval.total_num_envs=4'
  'env.eval.rollout_epoch=1'
  'env.eval.max_episode_steps=20'
  'env.eval.max_steps_per_rollout_epoch=20'
  'actor.micro_batch_size=128'
  'actor.global_batch_size=512'
)

"$venv/bin/python" "$worktree/examples/embodiment/train_embodied_agent.py" \
  "${args[@]}" --cfg job --resolve > "$fresh_log/resolved.yaml"
sha256sum "$fresh_log/resolved.yaml" > "$fresh_log/resolved.yaml.sha256"
printf '%q ' \
  "$venv/bin/python" \
  "$worktree/examples/embodiment/train_embodied_agent.py" \
  "${args[@]}" \
  > "$fresh_log/command.txt"
printf '\n' >> "$fresh_log/command.txt"
printf '%s\n' \
  "prepared_at=$(date --iso-8601=seconds)" \
  "source_head=$expected_head" \
  "stage1_checkpoint=$stage1_checkpoint" \
  "stage1_manifest=$stage1_manifest" \
  "stage1_manifest_id=$stage1_manifest_id" \
  "stage1_manifest_sha256=$RLT_STAGE1_MANIFEST_SHA256" \
  'physical_gpus=4,5' \
  'actor_world_size=2' \
  'train_envs=4' \
  'eval_envs=4' \
  'primitive_steps_per_episode_max=20' \
  'executed_chunk=10' \
  'train_macro_transitions_max=8' \
  'fresh_expected_critic_updates=8' \
  'fresh_expected_actor_updates=4' \
  'fresh_expected_update_step=8' \
  'global_batch=512' \
  'micro_batch=128' \
  'max_steps=1' \
  'save_interval=1' \
  'eval_interval=1' \
  > "$fresh_log/launch_manifest.txt"

nohup setsid bash -c '
  log=$1
  shift
  date --iso-8601=seconds > "$log/started_at.txt"
  "$@" > "$log/driver.log" 2>&1
  rc=$?
  printf "%s\n" "$rc" > "$log/exit_code.txt"
  date --iso-8601=seconds > "$log/finished_at.txt"
  exit "$rc"
' _ \
  "$fresh_log" \
  "$venv/bin/python" \
  "$worktree/examples/embodiment/train_embodied_agent.py" \
  "${args[@]}" \
  > "$fresh_log/wrapper.log" 2>&1 < /dev/null &
wrapper_pid=$!
printf '%s\n' "$wrapper_pid" > "$fresh_log/wrapper.pid"
printf '%s\n' "$wrapper_pid" > "$fresh_log/owned.pgid"

nohup setsid bash "$observer" "$wrapper_pid" "$fresh_log/resource.csv" \
  > "$fresh_log/resource_observer.log" 2>&1 < /dev/null &
observer_pid=$!
printf '%s\n' "$observer_pid" > "$fresh_log/resource_observer.pid"

sleep 10
if ! kill -0 "$wrapper_pid" 2>/dev/null; then
  tail -n 100 "$fresh_log/driver.log" >&2 || true
  exit 1
fi
kill -0 "$observer_pid"

printf 'run_root=%s\nfresh_log=%s\nwrapper_pid=%s\nobserver_pid=%s\nresolved_sha256=%s\ncheckpoint=%s\n' \
  "$run_root" \
  "$fresh_log" \
  "$wrapper_pid" \
  "$observer_pid" \
  "$(awk '{print $1}' "$fresh_log/resolved.yaml.sha256")" \
  "$fresh_log/$experiment/checkpoints/global_step_1"
tail -n 30 "$fresh_log/driver.log" || true
printf '%s\n' 'SZ_RLT_CURRENT_STAGE2_FRESH1_LAUNCHED'
