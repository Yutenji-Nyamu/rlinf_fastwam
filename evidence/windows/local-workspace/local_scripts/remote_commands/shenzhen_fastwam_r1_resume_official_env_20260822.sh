#!/usr/bin/env bash
set -euo pipefail

FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
PIN=7faa71108368fbb3b6885649f112af607427a2d4
OFFICIAL_REMOTE=https://github.com/yuantianyuan01/FastWAM.git
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
CACHE_ROOT=/home/chenyiteng/cache/fastwam-7faa
FREEZE="$CACHE_ROOT/pip-freeze-after-fastwam.txt"

printf 'timestamp_start=%s\n' "$(date --iso-8601=seconds)"
printf 'source=%s\nenv=%s\ncache=%s\n' "$FW" "$ENV" "$CACHE_ROOT"

# Resume the exact interrupted prefix; never recreate or replace it.
test -x "$ENV/bin/python"
test "$(git -C "$FW" rev-parse HEAD)" = "$PIN"
test "$(git -C "$FW" remote get-url origin)" = "$OFFICIAL_REMOTE"
test -z "$(git -C "$FW" status --porcelain)"
"$ENV/bin/python" - <<'PY'
import sys
assert sys.version_info[:2] == (3, 10), sys.version
print(f"python_pre={sys.version}")
PY

printf '%s\n' '=== preflight resources and package state ==='
df -h /home /data
free -h
nvidia-smi -i 3 --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
gpu3_apps="$(nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
test -z "$gpu3_apps"
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
"$ENV/bin/python" -m pip list --format=freeze \
  | grep -Ei '^(torch|torchvision|fastwam|numpy|packaging|pip|setuptools|wheel|ninja|setuptools-scm)=' \
  | sort || true

# Proxy and every cache override live only in this command process.
source /etc/profile.d/mihomo-proxy.sh
mkdir -p \
  "$CACHE_ROOT/pip" \
  "$CACHE_ROOT/huggingface/hub" \
  "$CACHE_ROOT/modelscope" \
  "$CACHE_ROOT/torch" \
  "$CACHE_ROOT/torch_extensions" \
  "$CACHE_ROOT/triton" \
  "$CACHE_ROOT/cuda" \
  "$CACHE_ROOT/xdg" \
  "$CACHE_ROOT/tmp"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export MODELSCOPE_CACHE="$CACHE_ROOT/modelscope"
export TORCH_HOME="$CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TMPDIR="$CACHE_ROOT/tmp"
export PIP_DEFAULT_TIMEOUT=120
export CUDA_VISIBLE_DEVICES=3
export PATH="$ENV/bin:$PATH"
unset PYTHONPATH
unset PIP_TRUSTED_HOST

printf '%s\n' '=== process-only route ==='
printf 'PIP_CACHE_DIR=%s\nHF_HOME=%s\nMODELSCOPE_CACHE=%s\nTORCH_EXTENSIONS_DIR=%s\nTMPDIR=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$PIP_CACHE_DIR" "$HF_HOME" "$MODELSCOPE_CACHE" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$CUDA_VISIBLE_DEVICES"
env | grep -iE '^(http|https|all|no)_proxy=' \
  | sed -E 's#(https?://)[^/@]+@#\1REDACTED@#I' \
  | sort
curl -sSIL --max-time 20 --connect-timeout 8 -o /dev/null \
  -w 'pypi http=%{http_code} remote=%{remote_ip} total=%{time_total}\n' \
  https://pypi.org/simple/
curl -sSIL --max-time 20 --connect-timeout 8 -o /dev/null \
  -w 'pytorch_cu128 http=%{http_code} remote=%{remote_ip} total=%{time_total}\n' \
  https://download.pytorch.org/whl/cu128/

# Official README environment semantics, resumed inside the existing prefix.
python -m pip install --index-url https://pypi.org/simple \
  --upgrade pip wheel ninja setuptools_scm 'setuptools==80.9.0'

python -m pip install --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128'

cd "$FW"
python -m pip install --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e .

printf '%s\n' '=== official base-environment acceptance ==='
FW="$FW" python - <<'PY'
import importlib.metadata as metadata
import json
import os
import pathlib
import sys
import urllib.parse

import fastwam
import torch
import torchvision

fw = pathlib.Path(os.environ["FW"]).resolve()
fastwam_file = pathlib.Path(fastwam.__file__).resolve()
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__ == "2.7.1+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torchvision.__version__ == "0.22.1+cu128", torchvision.__version__
assert metadata.version("fastwam") == "0.1.0"
assert metadata.version("numpy") == "2.2.6"
assert fw in fastwam_file.parents, (fw, fastwam_file)
direct_url = json.loads(metadata.distribution("fastwam").read_text("direct_url.json"))
editable_path = pathlib.Path(urllib.parse.unquote(urllib.parse.urlparse(direct_url["url"]).path)).resolve()
assert direct_url.get("dir_info", {}).get("editable") is True, direct_url
assert editable_path == fw, (editable_path, fw)
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "3"
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
x = torch.ones(8, device="cuda:0")
assert float((x * x).sum()) == 8.0
torch.cuda.synchronize()
print(json.dumps({
    "python": sys.version,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torchvision": torchvision.__version__,
    "numpy": metadata.version("numpy"),
    "fastwam_version": metadata.version("fastwam"),
    "fastwam_file": str(fastwam_file),
    "fastwam_editable_root": str(editable_path),
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "visible_gpu0": torch.cuda.get_device_name(0),
    "visible_capability0": torch.cuda.get_device_capability(0),
}, indent=2))
PY

python -m pip check
test ! -e "$FREEZE"
python -m pip freeze | LC_ALL=C sort > "$FREEZE"
sha256sum "$FREEZE"
du -sh "$ENV" "$CACHE_ROOT"
git -C "$FW" diff --quiet
git -C "$FW" diff --cached --quiet
git -C "$FW" status --short --branch

printf '%s\n' '=== postflight GPU boundary ==='
nvidia-smi -i 3 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'timestamp_end=%s\n' "$(date --iso-8601=seconds)"
printf '%s\n' FASTWAM_R1_RESUME_OFFICIAL_ENV_OK
