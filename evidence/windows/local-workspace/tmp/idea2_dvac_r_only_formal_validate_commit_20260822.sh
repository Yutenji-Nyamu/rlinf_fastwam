#!/usr/bin/env bash
set -euo pipefail

source_root=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
config_name=robotwin_adjust_bottle_grpo_openpi_dvac_r_only_downweight_100step_formal
config_file="$source_root/examples/embodiment/config/${config_name}.yaml"
launcher="$runtime_dir/launch_formal.sh"
observer="$runtime_dir/observe_resources.sh"
python_bin=/root/autodl-tmp/RLinf/.venv/bin/python
robotwin_root=/root/autodl-tmp/idea2_dvac_train_wamppo/RoboTwin_RLinf

chmod 755 "$launcher" "$observer"
bash -n "$launcher"
bash -n "$observer"
sha256sum "$config_file" "$launcher" "$observer"
git -C "$source_root" status --short --branch

export CUDA_VISIBLE_DEVICES=0,1
export EMBODIED_PATH="$source_root/examples/embodiment"
export REPO_PATH="$source_root"
export ROBOTWIN_PATH="$robotwin_root"
export ROBOT_PLATFORM=ALOHA
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$source_root:$robotwin_root${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" "$source_root/examples/embodiment/train_embodied_agent.py" \
  --config-path "$source_root/examples/embodiment/config/" \
  --config-name "$config_name" \
  --cfg job --resolve > "$runtime_dir/resolved_config.yaml"

grep -n -E 'max_steps: 100|rollout_epoch: 16|total_num_envs: 16|group_size: 8|global_batch_size: 512|micro_batch_size: 32|update_epoch: 2|signal_mode: per_h_robust_residual|weight_min: 0.5|weight_max: 1.2' "$runtime_dir/resolved_config.yaml"

git -C "$source_root" add "examples/embodiment/config/${config_name}.yaml"
git -C "$source_root" commit -m 'config: add R-only DVAC 100-step formal run'
git -C "$source_root" push personal codex/idea2-dvac-residual-downweight
git -C "$source_root" rev-parse HEAD
git -C "$source_root" status --short --branch
