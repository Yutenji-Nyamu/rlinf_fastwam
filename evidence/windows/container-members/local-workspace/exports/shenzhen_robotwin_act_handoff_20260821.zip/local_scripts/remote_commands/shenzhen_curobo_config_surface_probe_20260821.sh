#!/usr/bin/env bash
set -euo pipefail

source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin
cd /data/chenyiteng/projects/robotwin-native/RoboTwin

CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 180s python3 - <<'PY'
import os
from curobo.wrap.reacher.motion_gen import MotionGenConfig

yml_path = os.path.abspath('assets/embodiments/aloha-agilex/curobo_left.yml')
world_config = {'cuboid': {'table': {'dims': [0.7, 2, 0.04], 'pose': [-0.65, 0.0, 0.74, 1, 0, 0, 0.0]}}}
cfg = MotionGenConfig.load_from_robot_config(
    yml_path,
    world_config,
    interpolation_dt=1 / 250,
    num_trajopt_seeds=1,
    use_cuda_graph=False,
)

seen = set()
def walk(obj, path, depth):
    if depth > 10 or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, 'use_cuda_kernel'):
        print('FOUND', path, type(obj).__module__ + '.' + type(obj).__name__, 'use_cuda_kernel=', getattr(obj, 'use_cuda_kernel'))
    if isinstance(obj, dict):
        for key, value in obj.items():
            walk(value, f'{path}[{key!r}]', depth + 1)
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            walk(value, f'{path}[{index}]', depth + 1)
    elif hasattr(obj, '__dict__') and type(obj).__module__.startswith('curobo'):
        for key, value in vars(obj).items():
            walk(value, f'{path}.{key}', depth + 1)

walk(cfg, 'cfg', 0)
print('TOP_FIELDS', sorted(vars(cfg)))
PY
