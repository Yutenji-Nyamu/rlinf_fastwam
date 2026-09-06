#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
RUN=/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821

test "$(git -C "$ROOT" rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839
test "$(git -C "$ROBOTWIN" rev-parse HEAD)" = 0008ae6800df9f75fc8de7098bacb01735fd8fd2
test -s "$MODEL/model-00001-of-00002.safetensors"
test -x "$VENV/bin/python"

source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export REPO_PATH="$ROOT"
export EMBODIED_PATH="$ROOT/examples/embodiment"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1

exec > >(tee "$RUN/runtime_compose_model_load_retry2.log") 2>&1

printf '%s\n' '=== UV/PACKAGE CHECK ==='
set +e
uv pip check --python "$VENV/bin/python"
pip_check_rc=$?
set -e
printf 'uv_pip_check_exit=%s\n' "$pip_check_rc"

timeout --signal=INT --kill-after=60s 600s \
  "$VENV/bin/python" - "$RUN/runtime_versions.json" <<'PY'
from __future__ import annotations

import importlib.metadata as md
import json
from pathlib import Path
import sys

import torch

assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0).startswith("NVIDIA H100")
assert torch.cuda.get_device_capability(0)[0] == 9
x = torch.ones(8, device="cuda")
assert float((x * x).sum()) == 8.0

import rlinf
import ray
import openpi
import jax
import orbax.checkpoint
import sapien
import mplib
import warp as wp
import curobo
from pytorch3d import _C
from curobo.types.base import TensorDeviceType
from robotwin.envs import vector_env

wp.init()
TensorDeviceType()

names = [
    "rlinf-openpi", "rlinf-transformer-openpi", "transformers", "tokenizers",
    "jax", "jaxlib", "jax-cuda12-plugin", "orbax-checkpoint", "ray", "sapien",
    "mplib", "pytorch3d", "warp-lang", "nvidia-curobo", "flash-attn",
    "huggingface-hub", "torch", "torchvision", "torchaudio",
]
versions = {}
for name in names:
    try:
        versions[name] = md.version(name)
    except md.PackageNotFoundError:
        versions[name] = None

manifest = {
    "versions": versions,
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "rlinf_file": rlinf.__file__,
    "robotwin_vector_env_file": vector_env.__file__,
    "jax_version": jax.__version__,
}
Path(sys.argv[1]).write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

printf '%s\n' '=== OFFICIAL EVAL CONFIG COMPOSE ==='
RESOLVED="$RUN/robotwin_adjust_bottle_openpi_eval.resolved.yaml"
timeout --signal=INT --kill-after=30s 300s \
  "$VENV/bin/python" "$ROOT/evaluations/eval_embodied_agent.py" \
    --config-path "$ROOT/evaluations/robotwin" \
    --config-name robotwin_adjust_bottle_openpi_eval \
    "runner.logger.log_path=$RUN/compose-only" \
    "rollout.model.model_path=$MODEL" \
    "env.eval.assets_path=$ROBOTWIN" \
    --cfg job --resolve > "$RESOLVED"
test -s "$RESOLVED"
grep -Fq "$MODEL" "$RESOLVED"
grep -Fq "$ROBOTWIN" "$RESOLVED"
grep -E 'model_config|image_keys|action_dim|action_chunks|total_num_envs|assets_path|model_path' "$RESOLVED" | head -n 80

printf '%s\n' '=== OFFICIAL OPENPI CHECKPOINT LOAD-ONLY ==='
timeout --signal=INT --kill-after=120s 1800s \
  "$VENV/bin/python" - "$RESOLVED" "$RUN/model_load.json" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
from rlinf.models.embodiment.openpi import get_model

cfg = OmegaConf.load(sys.argv[1])
model = get_model(cfg.rollout.model)
parameters = list(model.parameters())
assert parameters
assert not any(parameter.is_meta for parameter in parameters)
assert hasattr(model, "_input_transform")
assert hasattr(model, "_output_transform")
result = {
    "class": type(model).__name__,
    "parameters": sum(parameter.numel() for parameter in parameters),
    "parameter_device": str(parameters[0].device),
    "config_name": model.config.config_name,
    "action_chunk": model.config.action_chunk,
    "action_env_dim": model.config.action_env_dim,
}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

printf '%s\n' '=== REPRODUCIBILITY MANIFEST ==='
uv pip freeze --python "$VENV/bin/python" > "$RUN/uv-pip-freeze.txt"
find "$VENV/lib/python3.11/site-packages" -path '*/direct_url.json' -type f -print -exec cat {} \; \
  > "$RUN/direct-url-manifest.txt"
if test -s "$ROOT/uv.lock"; then
  sha256sum "$ROOT/uv.lock" | tee "$RUN/uv-lock.sha256"
fi
git -C "$ROOT" status --short --branch
git -C "$ROBOTWIN" status --short --branch
du -sh "$VENV" /home/chenyiteng/.cache/uv "$MODEL" "$ROBOTWIN/assets"
df -h / /home /data
printf '%s\n' 'R1_RUNTIME_COMPOSE_MODEL_LOAD_OK'
