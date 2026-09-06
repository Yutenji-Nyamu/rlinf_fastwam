#!/usr/bin/env bash
set -euo pipefail

source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin
cd /data/chenyiteng/projects/robotwin-native/RoboTwin

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 TORCH_DISABLE_ADDR2LINE=1 \
timeout --signal=TERM --kill-after=30s 600s /usr/bin/time -f 'elapsed=%e maxrss_kb=%M' python3 - <<'PY'
import os
import torch
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

yml_path = os.path.abspath('assets/embodiments/aloha-agilex/curobo_left.yml')
world_config = {'cuboid': {'table': {'dims': [0.7, 2, 0.04], 'pose': [-0.65, 0.0, 0.74, 1, 0, 0, 0.0]}}}
cfg = MotionGenConfig.load_from_robot_config(
    yml_path,
    world_config,
    interpolation_dt=1 / 250,
    num_trajopt_seeds=1,
    use_cuda_graph=False,
)

changed = []
for field in ('ik_solver', 'trajopt_solver', 'js_trajopt_solver', 'finetune_js_trajopt_solver', 'finetune_trajopt_solver'):
    solver = getattr(cfg, field, None)
    if solver is None:
        continue
    inner = getattr(solver, 'solver', None)
    for index, optimizer in enumerate(getattr(inner, 'optimizers', ())):
        if hasattr(optimizer, 'use_cuda_kernel'):
            optimizer.use_cuda_kernel = False
            changed.append(f'{field}.solver.optimizers[{index}]')
print('disabled_fused_lbfgs', changed, flush=True)
mg = MotionGen(cfg)
print('MotionGen constructed', flush=True)
mg.warmup()
torch.cuda.synchronize()
print('UNFUSED_LBFGS_WARMUP_OK', flush=True)
PY
