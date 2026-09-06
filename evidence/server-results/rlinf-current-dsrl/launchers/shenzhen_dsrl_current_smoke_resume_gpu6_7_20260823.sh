#!/usr/bin/env bash
set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
model=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
run_root=/data/chenyiteng/results/rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823
fresh_log=$run_root/fresh
fresh_experiment=dsrl-fresh-step1
checkpoint1=$fresh_log/$fresh_experiment/checkpoints/global_step_1
resume_log=$run_root/resume
experiment=dsrl-resume-step2
observer=/data/chenyiteng/results/rlinf-current-dsrl/launchers/shenzhen_dsrl_current_smoke_observer_gpu6_7_20260823.sh
expected_head=4b609178d10d2534f3f972435ad972e4e015c392
expected_branch=codex/sz-current-dsrl-pi0-robotwin

test "$(git -C "$worktree" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$worktree" branch --show-current)" = "$expected_branch"
test "$(git -C "$worktree" rev-parse "personal/$expected_branch")" = "$expected_head"
test -z "$(git -C "$worktree" status --porcelain)"
test -x "$venv/bin/python"
test -s "$model/model-00001-of-00002.safetensors"
test -s "$model/model-00002-of-00002.safetensors"
test -d "$robotwin"
test -r "$observer"
test "$(cat "$fresh_log/exit_code.txt")" = 0
test ! -e "$resume_log"
test -d "$checkpoint1/actor"
test -d "$checkpoint1/actor/sac_components/alpha"
for rank in 0 1; do
  test -f "$checkpoint1/actor/sac_components/dsrl_trainer_state_rank_${rank}.pt"
  test -f "$checkpoint1/actor/sac_components/target_model/checkpoint_rank_${rank}.pt"
  test -f "$checkpoint1/actor/sac_components/replay_buffer/rank_${rank}/dsrl_transition_replay.pt"
done

fresh_pid=$(cat "$fresh_log/wrapper.pid")
if kill -0 "$fresh_pid" 2>/dev/null; then
  printf 'fresh wrapper is still alive: %s\n' "$fresh_pid" >&2
  exit 1
fi
if pgrep -u "$(id -u)" -x raylet >/dev/null || \
  pgrep -u "$(id -u)" -x gcs_server >/dev/null; then
  printf '%s\n' 'chenyiteng still has a live Ray cluster' >&2
  exit 1
fi
if nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'physical GPU 6-7 are not idle' >&2
  exit 1
fi

source "$venv/bin/activate"
unset CUDA_VISIBLE_DEVICES
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export ROBOTWIN_PATH="$robotwin"
export ROBOT_PLATFORM=ALOHA
export REPO_PATH="$worktree"
export EMBODIED_PATH="$worktree/examples/embodiment"
export PYTHONPATH="$worktree:$robotwin${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$resume_log"

args=(
  --config-path "$worktree/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_dsrl_openpi
  'cluster.component_placement={actor\,\ env\,\ rollout:6-7}'
  "runner.logger.log_path=$resume_log"
  "runner.logger.experiment_name=$experiment"
  'runner.max_epochs=1000'
  'runner.max_steps=2'
  'runner.val_check_interval=-1'
  'runner.save_interval=1'
  "runner.resume_dir=$checkpoint1"
  'runner.ckpt_path=null'
  'algorithm.utd_ratio=20'
  'algorithm.critic_actor_ratio=1'
  'algorithm.replay_buffer.capacity=25000'
  'algorithm.replay_buffer.warmup_size=4'
  'env.train.total_num_envs=4'
  'env.train.rollout_epoch=1'
  'env.train.max_episode_steps=200'
  'env.train.max_steps_per_rollout_epoch=200'
  "env.train.assets_path=$robotwin"
  "env.eval.assets_path=$robotwin"
  'actor.micro_batch_size=64'
  'actor.global_batch_size=256'
  "actor.model.model_path=$model"
  'actor.model.num_action_chunks=20'
  'actor.model.openpi.action_horizon=50'
  'actor.model.openpi.rtc_enabled=false'
  'actor.model.openpi.use_dsrl=true'
  'actor.model.openpi.dsrl_gaussian_warmup=true'
  'actor.model.openpi.dsrl_eval_deterministic=false'
  'actor.fsdp_config.save_full_model_weights=false'
)

"$venv/bin/python" "$worktree/examples/embodiment/train_embodied_agent.py" \
  "${args[@]}" --cfg job --resolve > "$resume_log/resolved.yaml"
sha256sum "$resume_log/resolved.yaml" > "$resume_log/resolved.yaml.sha256"
printf '%q ' \
  "$venv/bin/python" \
  "$worktree/examples/embodiment/train_embodied_agent.py" \
  "${args[@]}" \
  > "$resume_log/command.txt"
printf '\n' >> "$resume_log/command.txt"
printf '%s\n' \
  "prepared_at=$(date --iso-8601=seconds)" \
  "source_head=$expected_head" \
  "resume_from=$checkpoint1" \
  'physical_gpus=6,7' \
  'actor_world_size=2' \
  'train_envs=4' \
  'action_horizon=50' \
  'executed_chunk=20' \
  'global_batch=256' \
  'micro_batch=64' \
  'replay_capacity=25000' \
  'warmup_size=4' \
  'utd_ratio=20' \
  'max_steps=2' \
  'save_interval=1' \
  'eval_enabled=false' \
  > "$resume_log/launch_manifest.txt"

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
  "$resume_log" \
  "$venv/bin/python" \
  "$worktree/examples/embodiment/train_embodied_agent.py" \
  "${args[@]}" \
  > "$resume_log/wrapper.log" 2>&1 < /dev/null &
wrapper_pid=$!
printf '%s\n' "$wrapper_pid" > "$resume_log/wrapper.pid"
printf '%s\n' "$wrapper_pid" > "$resume_log/owned.pgid"

nohup setsid bash "$observer" "$wrapper_pid" "$resume_log/resource.csv" \
  > "$resume_log/resource_observer.log" 2>&1 < /dev/null &
observer_pid=$!
printf '%s\n' "$observer_pid" > "$resume_log/resource_observer.pid"

sleep 10
if ! kill -0 "$wrapper_pid" 2>/dev/null; then
  tail -n 80 "$resume_log/driver.log" >&2 || true
  exit 1
fi
kill -0 "$observer_pid"

printf 'run_root=%s\nresume_log=%s\nwrapper_pid=%s\nowned_pgid=%s\nobserver_pid=%s\nresolved_sha256=%s\n' \
  "$run_root" \
  "$resume_log" \
  "$wrapper_pid" \
  "$wrapper_pid" \
  "$observer_pid" \
  "$(awk '{print $1}' "$resume_log/resolved.yaml.sha256")"
tail -n 30 "$resume_log/driver.log" || true
printf '%s\n' 'SZ_DSRL_CURRENT_RESUME_STEP2_LAUNCHED'
