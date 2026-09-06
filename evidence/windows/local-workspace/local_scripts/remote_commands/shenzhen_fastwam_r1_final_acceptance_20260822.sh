#!/usr/bin/env bash
set -euo pipefail

FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
PIN=7faa71108368fbb3b6885649f112af607427a2d4
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
CACHE_ROOT=/home/chenyiteng/cache/fastwam-7faa
FREEZE="$CACHE_ROOT/pip-freeze-after-fastwam.txt"

test "$(git -C "$FW" rev-parse HEAD)" = "$PIN"
test -z "$(git -C "$FW" status --porcelain)"
test -x "$ENV/bin/python"

export PATH="$ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=3
unset PYTHONPATH

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
python -m pip check
python - <<'PY'
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path

import fastwam
import numpy
import torch
import torchvision

root = Path('/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711').resolve()
direct = json.loads((Path(md.distribution('fastwam')._path) / 'direct_url.json').read_text())
editable_root = Path(direct['url'].removeprefix('file://')).resolve()
installed = {dist.metadata['Name'].lower().replace('_', '-') for dist in md.distributions()}
assert 'setuptools-scm' not in installed
assert 'vcs-versioning' not in installed
assert sys.version_info[:2] == (3, 10)
assert torch.__version__ == '2.7.1+cu128'
assert torch.version.cuda == '12.8'
assert torchvision.__version__ == '0.22.1+cu128'
assert numpy.__version__ == '2.2.6'
assert md.version('packaging') == '25.0'
assert editable_root == root
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
x = torch.arange(8, device='cuda:0', dtype=torch.float32)
assert float(x.sum()) == 28.0
print(json.dumps({
    'python': sys.version,
    'torch': torch.__version__,
    'torch_cuda': torch.version.cuda,
    'torchvision': torchvision.__version__,
    'numpy': numpy.__version__,
    'packaging': md.version('packaging'),
    'fastwam_version': md.version('fastwam'),
    'fastwam_file': fastwam.__file__,
    'fastwam_editable_root': str(editable_root),
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'visible_gpu0': torch.cuda.get_device_name(0),
    'visible_capability0': torch.cuda.get_device_capability(0),
}, indent=2))
PY
test -s "$FREEZE"
sha256sum "$FREEZE"
du -sh "$ENV" "$CACHE_ROOT"
df -h /home /data
free -h
git -C "$FW" status --short --branch
nvidia-smi -i 3,4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf '%s\n' FASTWAM_R1_OFFICIAL_ENV_OK
