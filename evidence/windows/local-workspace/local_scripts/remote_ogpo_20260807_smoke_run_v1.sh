#!/usr/bin/env bash
set +e
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_smoke_20260807_v1
experiment_name=robotwin_adjust_bottle_ogpo_ca_smoke_8env_1update_v1
runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_smoke_20260807_v1/runtime

cd "$repo" || exit 91
export PYTHONPATH="${repo}:${assets}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export EMBODIED_PATH="${repo}/examples/embodiment"
export REPO_PATH="$repo"
export ROBOTWIN_PATH="$assets"
export ROBOTWIN_ASSETS_PATH="$assets"
export ROBOT_PLATFORM=ALOHA
export CUDA_VISIBLE_DEVICES=0,1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY

date --iso-8601=seconds >"$runtime_root/started_at.txt"
timeout --signal=TERM --kill-after=120s 1800s \
  "${venv}/bin/python" -B \
  examples/embodiment/train_embodied_agent.py \
  --config-path "${repo}/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_ogpo_openpi \
  "runner.logger.log_path=${run_root}" \
  "runner.logger.experiment_name=${experiment_name}" \
  "runner.max_epochs=2" \
  "runner.max_steps=2" \
  "runner.resume_dir=null" \
  "runner.ckpt_path=null" \
  "algorithm.ogpo.start_training_rows=79" \
  "algorithm.ogpo.total_online_rows=80" \
  "algorithm.ogpo.replay_capacity=80" \
  "algorithm.ogpo.utd_q=1.0" \
  "algorithm.ogpo.utd_pi=1.0" \
  "algorithm.ogpo.baseline_eval=false" \
  "algorithm.ogpo.final_eval=true" \
  "algorithm.ogpo.eval_interval_rows=80" \
  "algorithm.ogpo.checkpoint_interval_rows=80" \
  "env.train.rollout_epoch=1" \
  "env.train.total_num_envs=8" \
  "env.train.max_steps_per_rollout_epoch=10" \
  "env.eval.rollout_epoch=1" \
  "env.eval.total_num_envs=4" \
  "env.eval.max_steps_per_rollout_epoch=10" \
  "actor.optim.total_training_steps=1"
rc=$?
printf '%s\n' "$rc" >"$runtime_root/exit_code.txt"
date --iso-8601=seconds >"$runtime_root/finished_at.txt"
{
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  free -b
  cat /sys/fs/cgroup/memory.current
  cat /sys/fs/cgroup/memory.events
  df -B1 /root/autodl-tmp
} >"$runtime_root/resources_after.txt"
exit "$rc"
