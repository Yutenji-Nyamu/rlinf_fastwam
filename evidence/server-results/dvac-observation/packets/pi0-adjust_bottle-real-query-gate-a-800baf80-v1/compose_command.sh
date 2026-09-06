#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export NVIDIA_VISIBLE_DEVICES=none
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export ROBOTWIN_PATH=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
export ROBOT_PLATFORM=ALOHA
export REPO_PATH=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
export EMBODIED_PATH=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421/examples/embodiment
export PYTHONPATH=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HYDRA_FULL_ERROR=1
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421/evaluations/eval_embodied_agent.py --config-path /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421/evaluations/robotwin --config-name robotwin_adjust_bottle_openpi_dvac_eval 'cluster.component_placement={env\,\ rollout:2}' runner.logger.log_path=/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-real-query-gate-a-800baf80-v1 rollout.model.model_path=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50 env.eval.assets_path=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support env.eval.total_num_envs=1 env.eval.rollout_epoch=1 env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200 env.eval.use_fixed_reset_state_ids=true rollout.dvac_telemetry.enabled=false --cfg job --resolve
