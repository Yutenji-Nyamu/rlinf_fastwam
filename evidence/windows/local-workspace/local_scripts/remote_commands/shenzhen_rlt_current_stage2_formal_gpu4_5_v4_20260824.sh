#!/usr/bin/env bash
set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
model=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
norm_stats=$model/physical-intelligence/robotwin/norm_stats.json
chain_root=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2
stage1=$chain_root/stage1
stage1_runtime=$stage1/runtime
stage1_experiment=robotwin_adjust_bottle_rlt_stage1_current_ar_clean50_2k_v1
stage1_checkpoint=$stage1/$stage1_experiment/checkpoints/global_step_2000
stage1_manifest=$stage1/artifacts/stage1_artifact_manifest.json
stage1_manifest_id=sz-rlt-stage1-current-ar-clean50-2k-v1
run_root=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix
runtime=$run_root/runtime
experiment=robotwin_adjust_bottle_rlt_stage2_current_ar_8env250_optimizer_warmup_v3
observer=/data/chenyiteng/results/rlinf-rlt/launchers/shenzhen_rlt_current_smoke_observer_gpu4_5_20260823.sh
ray_address=172.17.0.1:6389
expected_head=8bbd0216113ec6eca8bbc67e79d9d642d48b8ed1
expected_branch=codex/sz-rlt-pi0-robotwin-ar
config=$worktree/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env250.yaml
eval_seeds=$worktree/rlinf/envs/robotwin/seeds/eval_seeds_adjust_bottle_rlt_periodic20_v1.json
expected_config_sha=92e4a212b78148fd5af65fdb82fc49359b9c5321c3531e0d667bde4bed613bf6
expected_seeds_sha=fb9c3353e27b83aad6fe7ff778437d960b084de9d981c2af68615d52769952a7
expected_norm_sha=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

test "$(git -C "$worktree" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$worktree" branch --show-current)" = "$expected_branch"
test "$(git -C "$worktree" rev-parse "personal/$expected_branch")" = "$expected_head"
test -z "$(git -C "$worktree" status --porcelain)"
test "$(sha256sum "$config" | awk '{print $1}')" = "$expected_config_sha"
test "$(sha256sum "$eval_seeds" | awk '{print $1}')" = "$expected_seeds_sha"
test "$(sha256sum "$norm_stats" | awk '{print $1}')" = "$expected_norm_sha"
test "$(cat "$stage1_runtime/exit_code.txt")" = 0
test -s "$stage1_checkpoint/actor/model_state_dict/full_weights.pt"
test -s "$stage1_checkpoint/actor/dcp_checkpoint/.metadata"
test -s "$stage1_manifest"
test -x "$venv/bin/python"
test -x "$venv/bin/ray"
test -d "$robotwin"
test -r "$observer"
test ! -e "$run_root"
RAY_ADDRESS="$ray_address" "$venv/bin/ray" status >/dev/null
if nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'physical GPU 4-5 are not idle' >&2
  exit 1
fi

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
export RLT_LOG_ROOT="$run_root"
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

mkdir -p "$runtime"

args=(
  --config-path "$worktree/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env250
  'cluster.component_placement={actor\,env\,rollout:4-5}'
  "runner.logger.log_path=$run_root"
  "runner.logger.experiment_name=$experiment"
  'runner.resume_dir=null'
  'runner.ckpt_path=null'
  'env.train.video_cfg.save_video=false'
  "env.train.video_cfg.video_base_dir=$run_root/video/train"
  "env.train.task_config.save_path=$run_root/robotwin_data/train"
  'env.eval.video_cfg.save_video=false'
  "env.eval.video_cfg.video_base_dir=$run_root/video/eval"
  "env.eval.task_config.save_path=$run_root/robotwin_data/eval"
)

"$venv/bin/python" "$worktree/examples/embodiment/train_embodied_agent.py" \
  "${args[@]}" --cfg job --resolve > "$runtime/resolved.yaml"
sha256sum "$runtime/resolved.yaml" > "$runtime/resolved.yaml.sha256"
printf '%q ' \
  "$venv/bin/python" \
  "$worktree/examples/embodiment/train_embodied_agent.py" \
  "${args[@]}" \
  > "$runtime/command.txt"
printf '\n' >> "$runtime/command.txt"
printf '%s\n' \
  "prepared_at=$(date --iso-8601=seconds)" \
  "source_head=$expected_head" \
  "shared_ray_address=$ray_address" \
  "ray_job_code_sync=$worktree/rlinf" \
  "stage1_checkpoint=$stage1_checkpoint" \
  "stage1_manifest=$stage1_manifest" \
  "stage1_manifest_sha256=$RLT_STAGE1_MANIFEST_SHA256" \
  'physical_gpus=4,5; actor_world_size=2' \
  'train=8 env x 250 cycles; 2000 episodes; H50/C10/D14' \
  'update=UTD5; critic:actor=2; GB512/MB128' \
  'eval=every25 cycles; 4 env x 5 waves = fixed20' \
  'save=every25 cycles; expected 10 checkpoints; multi-optimizer state eager-initialized at step0' \
  "train_output=$run_root/robotwin_data/train" \
  "eval_output=$run_root/robotwin_data/eval" \
  'normal_stop=cycle250' \
  'failure_stop=nonzero driver exit or hard timeout 72000s' \
  > "$runtime/launch_manifest.txt"

nohup setsid bash -c '
  runtime=$1
  shift
  date --iso-8601=seconds > "$runtime/started_at.txt"
  timeout --signal=TERM --kill-after=180s 72000s "$@" > "$runtime/driver.log" 2>&1
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

printf 'run_root=%s\nruntime=%s\nwrapper_pid=%s\nobserver_pid=%s\nresolved_sha256=%s\n' \
  "$run_root" "$runtime" "$wrapper_pid" "$observer_pid" \
  "$(awk '{print $1}' "$runtime/resolved.yaml.sha256")"
tail -n 40 "$runtime/driver.log" || true
printf '%s\n' 'SZ_RLT_CURRENT_STAGE2_FORMAL_V4_LAUNCHED'


