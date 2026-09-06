#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/robotwin-native/RoboTwin
CUROBO="$ROOT/envs/curobo"

echo '=== toolkits ==='
command -v nvcc || true
nvcc --version || true
find /usr/local -maxdepth 1 -type d -name 'cuda*' -printf '%f\n' 2>/dev/null | sort || true

echo '=== lbfgs config ==='
sed -n '80,185p' "$CUROBO/src/curobo/opt/newton/lbfgs.py"

echo '=== config construction references ==='
rg -n 'LBFGSOptConfig|use_cuda_kernel|lbfgs' \
  "$CUROBO/src/curobo/wrap/reacher" \
  "$CUROBO/src/curobo/opt" \
  "$ROOT/envs/robot/planner.py" | head -n 240 || true

echo '=== planner construction ==='
sed -n '1,260p' "$ROOT/envs/robot/planner.py"

echo '=== curobo build products ==='
find "$CUROBO" -maxdepth 4 -type f \( -name '*.so' -o -name '*lbfgs*' \) -printf '%p %s\n' | sort

echo '=== environment packages ==='
source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin
python - <<'PY'
import importlib.metadata as m
for p in ('torch','torchvision','nvidia-cuda-runtime-cu12','nvidia-cuda-nvcc-cu12','warp-lang'):
    try:
        print(p, m.version(p))
    except m.PackageNotFoundError:
        print(p, 'NOT_INSTALLED')
PY
