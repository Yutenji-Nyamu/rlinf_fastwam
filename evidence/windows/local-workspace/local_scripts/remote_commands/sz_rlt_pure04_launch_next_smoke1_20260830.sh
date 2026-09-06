#!/usr/bin/env bash
set -euo pipefail

worktree=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-dvac-pure-single-gpu-7d07a421
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
robotwin=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
model=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
norm_stats=$model/physical-intelligence/robotwin/norm_stats.json
stage1_root=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2/stage1
stage1_experiment=robotwin_adjust_bottle_rlt_stage1_current_ar_clean50_2k_v1
stage1_checkpoint=$stage1_root/$stage1_experiment/checkpoints/global_step_2000
stage1_manifest=$stage1_root/artifacts/stage1_artifact_manifest.json
ray_address=172.17.0.1:6389
expected_head=b1e01364b01a9f6d6072e2645cd7ba3bdf0df8fd
control_root=/data/chenyiteng/results/rlinf-rlt/smokes/current-single-gpu-control-phys2-1cycle-20260830-v2
pure_root=/data/chenyiteng/results/rlinf-rlt/smokes/current-single-gpu-pure04-phys3-1cycle-20260830-v2
action_run=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1
st_run=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1

if [[ ! -e "$control_root" ]]; then
  case_name=control
  gpu=2
  config=robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env_single_gpu_control
  run_root=$control_root
  experiment=robotwin_rlt_current_single_gpu_control_smoke1_phys2_v2
elif [[ ! -e "$pure_root" ]]; then
  case_name=pure04
  gpu=3
  config=robotwin_adjust_bottle_rlt_stage2_ac_mlp_current_8env_single_gpu_dvac_pure04
  run_root=$pure_root
  experiment=robotwin_rlt_current_single_gpu_pure04_smoke1_phys3_v2
else
  printf 'both_smoke_roots_already_exist\ncontrol=%s\npure=%s\n' "$control_root" "$pure_root"
  exit 2
fi
runtime=$run_root/runtime

test "$(git -C "$worktree" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$worktree" rev-parse personal/codex/sz-rlt-dvac-pure-single-gpu)" = "$expected_head"
test -z "$(git -C "$worktree" status --porcelain)"
test -s "$stage1_checkpoint/actor/model_state_dict/full_weights.pt"
test -s "$stage1_manifest"
test -s "$norm_stats"
test ! -e "$run_root"
RAY_ADDRESS="$ray_address" "$venv/bin/ray" status >/dev/null
if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf 'physical GPU %s is not idle\n' "$gpu" >&2
  exit 1
fi

for grpo_run in "$action_run" "$st_run"; do
  wrapper=$(<"$grpo_run/runtime/wrapper.pid")
  kill -0 "$wrapper"
done

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
export RLT_STAGE1_MANIFEST_ID=sz-rlt-stage1-current-ar-clean50-2k-v1
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
export OMP_NUM_THREADS=1

mkdir -p "$runtime/tmp"
export TMPDIR="$runtime/tmp"

args=(
  --config-path "$worktree/examples/embodiment/config"
  --config-name "$config"
  "cluster.component_placement={actor\\,env\\,rollout:$gpu}"
  "runner.logger.log_path=$run_root"
  "runner.logger.experiment_name=$experiment"
  'runner.max_steps=1'
  'runner.val_check_interval=-1'
  'runner.save_interval=1'
  'runner.resume_dir=null'
  'runner.ckpt_path=null'
  'algorithm.rlt_schedule.max_updates_per_train_step=20'
  'algorithm.rlt_schedule.warmup_min_size=2'
  'algorithm.rlt_schedule.warmup_post_collect_updates=8'
  'algorithm.actor_weight_schedule.warmup_updates=4'
  'algorithm.actor_weight_schedule.ramp_updates=8'
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
printf '%q ' "$venv/bin/python" "$worktree/examples/embodiment/train_embodied_agent.py" \
  "${args[@]}" > "$runtime/command.txt"
printf '\n' >> "$runtime/command.txt"

for grpo_run in "$action_run" "$st_run"; do
  wrapper=$(<"$grpo_run/runtime/wrapper.pid")
  printf '%s\t%s\t%s\t%s\n' "$grpo_run" "$wrapper" \
    "$(stat -c %s "$grpo_run/runtime/driver.log")" \
    "$(stat -c %Y "$grpo_run/runtime/driver.log")"
done > "$runtime/grpo_before.tsv"

printf '%s\n' \
  "prepared_at=$(date --iso-8601=seconds)" \
  "case=$case_name" \
  "source_head=$expected_head" \
  "shared_ray=$ray_address" \
  "physical_gpu=$gpu; actor_world_size=1; CUDA_VISIBLE_DEVICES=unset" \
  'train=8 env x 1 cycle; H50/C10/D14; GB512/MB256' \
  'updates=8 critic / 4 actor expected; max20' \
  'eval=disabled; save=global_step_1' \
  'normal_stop=exit0 + complete global_step_1' \
  'failure_stop=nonzero exit or 7200s timeout' \
  > "$runtime/launch_manifest.txt"

nohup setsid bash -c '
  runtime=$1
  shift
  date --iso-8601=seconds > "$runtime/started_at.txt"
  timeout --signal=TERM --kill-after=180s 7200s "$@" > "$runtime/driver.log" 2>&1
  rc=$?
  printf "%s\n" "$rc" > "$runtime/exit_code.txt"
  date --iso-8601=seconds > "$runtime/finished_at.txt"
  exit "$rc"
' _ "$runtime" "$venv/bin/python" \
  "$worktree/examples/embodiment/train_embodied_agent.py" "${args[@]}" \
  > "$runtime/wrapper.log" 2>&1 < /dev/null &
wrapper_pid=$!
printf '%s\n' "$wrapper_pid" > "$runtime/wrapper.pid"
printf '%s\n' "$wrapper_pid" > "$runtime/owned.pgid"

nohup setsid bash -c '
  pid=$1; csv=$2
  printf "%s\n" "timestamp,alive,mem_available_kib,gpu2_mib,gpu2_util,gpu3_mib,gpu3_util,gpu4_mib,gpu4_util,gpu5_mib,gpu5_util,gpu6_mib,gpu6_util,gpu7_mib,gpu7_util" > "$csv"
  while kill -0 "$pid" 2>/dev/null; do
    printf "%s,1,%s" "$(date --iso-8601=seconds)" "$(grep "^MemAvailable:" /proc/meminfo | tr -s " " | cut -d " " -f2)" >> "$csv"
    for card in 2 3 4 5 6 7; do
      printf ",%s" "$(nvidia-smi -i "$card" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d " ")" >> "$csv"
    done
    printf "\n" >> "$csv"
    sleep 10
  done
' _ "$wrapper_pid" "$runtime/resources.csv" \
  > "$runtime/resource_observer.log" 2>&1 < /dev/null &
observer_pid=$!
printf '%s\n' "$observer_pid" > "$runtime/resource_observer.pid"

sleep 12
kill -0 "$wrapper_pid"
kill -0 "$observer_pid"
printf 'case=%s\nrun_root=%s\nwrapper_pid=%s\nobserver_pid=%s\nLAUNCHED\n' \
  "$case_name" "$run_root" "$wrapper_pid" "$observer_pid"
tail -n 30 "$runtime/driver.log" || true
