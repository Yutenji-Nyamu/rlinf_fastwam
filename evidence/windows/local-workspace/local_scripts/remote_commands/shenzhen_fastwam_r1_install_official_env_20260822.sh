#!/usr/bin/env bash
set -euo pipefail

FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
PIN=7faa71108368fbb3b6885649f112af607427a2d4
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
CACHE_ROOT=/home/chenyiteng/cache/fastwam-7faa

test "$(git -C "$FW" rev-parse HEAD)" = "$PIN"
test -z "$(git -C "$FW" status --porcelain)"
test ! -e "$ENV"

source /etc/profile.d/mihomo-proxy.sh
export PATH=/home/chenyiteng/miniforge3/bin:$PATH
source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh

mkdir -p \
  "$CACHE_ROOT/pip" \
  "$CACHE_ROOT/huggingface" \
  "$CACHE_ROOT/torch_extensions" \
  "$CACHE_ROOT/tmp"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export TMPDIR="$CACHE_ROOT/tmp"
unset PYTHONPATH

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
printf 'env=%s\n' "$ENV"
printf 'cache=%s\n' "$CACHE_ROOT"

conda create -p "$ENV" python=3.10 pip -y
conda activate "$ENV"

python -m pip install --index-url https://pypi.org/simple \
  --upgrade pip wheel ninja setuptools_scm 'setuptools==80.9.0'

python -m pip install --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128'

cd "$FW"
python -m pip install --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e .

printf '%s\n' '=== official environment verification ==='
python - <<'PY'
import json
import sys
import torch
import torchvision
import fastwam

assert sys.version_info[:2] == (3, 10)
assert torch.__version__ == '2.7.1+cu128'
assert torchvision.__version__ == '0.22.1+cu128'
assert torch.cuda.is_available()
x = torch.ones(8, device='cuda:0')
assert float((x * x).sum()) == 8.0
print(json.dumps({
    'python': sys.version,
    'torch': torch.__version__,
    'torch_cuda': torch.version.cuda,
    'torchvision': torchvision.__version__,
    'gpu0': torch.cuda.get_device_name(0),
    'capability0': torch.cuda.get_device_capability(0),
    'fastwam_file': fastwam.__file__,
}, indent=2))
PY
python -m pip check
python -m pip freeze | sort > "$CACHE_ROOT/pip-freeze-after-fastwam.txt"
sha256sum "$CACHE_ROOT/pip-freeze-after-fastwam.txt"
du -sh "$ENV" "$CACHE_ROOT"
git -C "$FW" status --short --ignored | sed -n '1,120p'
printf '%s\n' FASTWAM_R1_OFFICIAL_ENV_OK
