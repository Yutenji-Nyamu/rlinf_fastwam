set -euo pipefail
source_root=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
robotwin_root=/root/autodl-tmp/idea2_dvac_train_wamppo/RoboTwin_RLinf
runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823
run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823
python_bin=/root/autodl-tmp/RLinf/.venv/bin/python
base_cfg="$source_root/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_train_100step_formal.yaml"
new_cfg="$source_root/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_global_z_w0to2_100step_formal.yaml"

echo '=== identity_source ==='
hostname
id -u
date '+%Y-%m-%d %H:%M:%S %Z'
git -C "$source_root" rev-parse HEAD
git -C "$source_root" status --short
sha256sum "$base_cfg" "$new_cfg"

echo '=== exact_source_diff ==='
diff -u "$base_cfg" "$new_cfg" || true
test "$(diff -U0 "$base_cfg" "$new_cfg" | grep -Ec '^[+-]    (log_path|experiment_name|strength):')" -eq 6
test "$(diff -U0 "$base_cfg" "$new_cfg" | grep -E '^[+-]' | grep -Ev '^---|^\+\+\+' | wc -l)" -eq 6
grep -q 'strength: 0.5' "$new_cfg"
grep -q 'stats_scope: global_all_queries_all_h' "$new_cfg"
grep -q 'z_clip: 2.0' "$new_cfg"

echo '=== mapping_and_backward ==='
"$python_bin" -c 'import torch; z=torch.tensor([-2.,-1.,0.,1.,2.]); w=1.+0.5*torch.clamp(z,-2,2); assert torch.equal(w,torch.tensor([0.,.5,1.,1.5,2.])); x=torch.arange(5.,requires_grad=True); y=x.detach()+w.detach()*(x-x.detach()); assert torch.equal(y,x); y.sum().backward(); assert torch.equal(x.grad,w); print("WEIGHTS",w.tolist(),"FORWARD_EQUAL",bool(torch.equal(y,x)),"GRAD",x.grad.tolist())'

echo '=== compose_resolve ==='
test ! -e "$run_dir"
export CUDA_VISIBLE_DEVICES=0,1
export EMBODIED_PATH="$source_root/examples/embodiment"
export REPO_PATH="$source_root"
export ROBOTWIN_PATH="$robotwin_root"
export ROBOT_PLATFORM=ALOHA
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$source_root:$robotwin_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$source_root"
"$python_bin" "$source_root/examples/embodiment/train_embodied_agent.py" \
  --config-path "$source_root/examples/embodiment/config/" \
  --config-name robotwin_adjust_bottle_grpo_openpi_dvac_global_z_w0to2_100step_formal \
  --cfg job --resolve > "$runtime_dir/resolved_config.yaml"
sha256sum "$runtime_dir/resolved_config.yaml"
grep -nE 'log_path:|experiment_name:|max_steps: 100|stats_scope: global_all_queries_all_h|z_clip: 2.0|strength: 0.5|total_num_envs: 16|rollout_epoch: 16|group_size: 8|global_batch_size: 512|mini_batch_size: 32|update_epoch: 2|clip_grad: 1.0|lr: 5.6e-06' "$runtime_dir/resolved_config.yaml"

echo '=== launch_and_resources ==='
bash -n "$runtime_dir/launch_formal.sh"
bash -n "$runtime_dir/observe_resources.sh"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
df -h /root/autodl-tmp | tail -n 1
echo GLOBAL_Z_W0TO2_PRETEST_OK
