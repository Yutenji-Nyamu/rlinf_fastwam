#!/usr/bin/env bash
set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
model=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
norm_stats=$model/physical-intelligence/robotwin/norm_stats.json
stage1_root=/data/chenyiteng/results/rlinf-rlt/smoke-stage1-current-ar-2step-20260823
stage1_experiment=robotwin_adjust_bottle_rlt_stage1_current_ar_smoke2_v1
stage1_checkpoint=$stage1_root/$stage1_experiment/checkpoints/global_step_2
stage1_manifest=$stage1_root/artifacts/stage1_artifact_manifest.json
stage1_manifest_id=sz-rlt-stage1-current-ar-smoke2-v1
run_root=/data/chenyiteng/results/rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823
fresh_log=$run_root/fresh
fresh_experiment=robotwin_adjust_bottle_rlt_stage2_current_ar_smoke_fresh1_v1
checkpoint1=$fresh_log/$fresh_experiment/checkpoints/global_step_1
resume_log=$run_root/resume
experiment=robotwin_adjust_bottle_rlt_stage2_current_ar_smoke_resume1_v1
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
test "$(cat "$fresh_log/exit_code.txt")" = 0
test -s "$stage1_checkpoint/actor/model_state_dict/full_weights.pt"
test -s "$stage1_manifest"
test -s "$checkpoint1/actor/dcp_checkpoint/.metadata"
test -s "$checkpoint1/actor/sac_components/rlt_trainer_state/complete.json"
for rank in 0 1; do
  test -s "$checkpoint1/actor/sac_components/rlt_trainer_state/checkpoint_rank_${rank}.pt"
  test -d "$checkpoint1/actor/sac_components/replay_buffer/rank_${rank}"
done
test -x "$venv/bin/python"
test -d "$robotwin"
test -r "$observer"
test ! -e "$resume_log"

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
if nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'physical GPU 4-5 are not idle' >&2
  exit 1
fi

"$venv/bin/python" -B -c '
import json, sys, torch
from pathlib import Path
checkpoint = Path(sys.argv[1])
complete = json.loads((checkpoint / "actor/sac_components/rlt_trainer_state/complete.json").read_text())
assert complete["complete"] is True
assert complete["actor_world_size"] == 2
assert complete["saved_runner_step"] == 1
assert complete["update_step"] == 8
assert complete["rank_files"] == ["checkpoint_rank_0.pt", "checkpoint_rank_1.pt"]
for rank in range(2):
    state = torch.load(
        checkpoint / f"actor/sac_components/rlt_trainer_state/checkpoint_rank_{rank}.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert state["rank"] == rank
    assert state["actor_world_size"] == 2
    assert state["saved_runner_step"] == 1
    assert state["update_step"] == 8
    assert state["local_total_transitions_added"] == 4
    assert state["global_warmup_ready_total_transitions"] == 8
print("FRESH_RLT_SIDECAR_OK update_step=8 local_transitions=4/rank")
' "$checkpoint1" > "$resume_log.preflight.txt"

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
export RLT_LOG_ROOT="$resume_log"
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

mkdir -p "$resume_log"
mv "$resume_log.preflight.txt" "$resume_log/preflight.txt"

args=(
  --config-path "$worktree/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_current
  'cluster.component_placement={actor\,env\,rollout:4-5}'
  "runner.logger.log_path=$resume_log"
  "runner.logger.experiment_name=$experiment"
  'runner.max_steps=2'
  'runner.val_check_interval=1'
  'runner.save_interval=1'
  "runner.resume_dir=$checkpoint1"
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
  "stage1_manifest_sha256=$RLT_STAGE1_MANIFEST_SHA256" \
  'physical_gpus=4,5' \
  'actor_world_size=2' \
  'train_envs=4' \
  'eval_envs=4' \
  'new_train_macro_transitions_max=8' \
  'resume_expected_new_critic_updates=20' \
  'resume_expected_new_actor_updates=10' \
  'resume_expected_update_step=28' \
  'global_batch=512' \
  'micro_batch=128' \
  'max_steps=2' \
  'save_interval=1' \
  'eval_interval=1' \
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
  tail -n 100 "$resume_log/driver.log" >&2 || true
  exit 1
fi
kill -0 "$observer_pid"

printf 'run_root=%s\nresume_log=%s\nwrapper_pid=%s\nobserver_pid=%s\nresolved_sha256=%s\ncheckpoint=%s\n' \
  "$run_root" \
  "$resume_log" \
  "$wrapper_pid" \
  "$observer_pid" \
  "$(awk '{print $1}' "$resume_log/resolved.yaml.sha256")" \
  "$resume_log/$experiment/checkpoints/global_step_2"
tail -n 30 "$resume_log/driver.log" || true
printf '%s\n' 'SZ_RLT_CURRENT_STAGE2_RESUME1_LAUNCHED'
