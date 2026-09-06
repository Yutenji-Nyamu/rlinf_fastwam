set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-action-dvac-adv
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/src
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
HEAD=a6ad77ea9ee9bf0b355251324c4cf88b6e9a47e7

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
if nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'physical GPU 2 is not idle' >&2
  exit 20
fi

export CUDA_VISIBLE_DEVICES=2
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
export RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$FW:$ROBOTWIN"
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope
export TMPDIR=/home/chenyiteng/cache/fastwam-7faa/tmp

"$VENV/bin/python" - <<'PY'
import json
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.dvac_train_weighting import compute_endpoint_variance
from rlinf.models.embodiment.fastwam.builder import build_fastwam_policy

cfg = OmegaConf.load(
    "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-action-dvac-adv/"
    "examples/embodiment/config/model/fastwam_robotwin.yaml"
)
policy = build_fastwam_policy(cfg, torch.bfloat16)
policy.eval()
generator = torch.Generator(device="cpu").manual_seed(20260831)
obs = {
    "main_images": torch.randint(0, 256, (1, 480, 640, 3), dtype=torch.uint8, generator=generator),
    "wrist_images": torch.randint(0, 256, (1, 2, 480, 640, 3), dtype=torch.uint8, generator=generator),
    "states": torch.zeros((1, 14), dtype=torch.float32),
    "task_descriptions": ["move the stapler onto the pad"],
}

torch.manual_seed(314159)
actions_off, result_off = policy.predict_action_batch(
    obs, mode="train", return_dvac_telemetry=False
)
torch.manual_seed(314159)
actions_on, result_on = policy.predict_action_batch(
    obs, mode="train", return_dvac_telemetry=True
)

action_max_abs = (actions_on - actions_off).abs().max().item()
logprob_max_abs = (
    result_on["prev_logprobs"] - result_off["prev_logprobs"]
).abs().max().item()
if action_max_abs != 0.0 or logprob_max_abs != 0.0:
    raise RuntimeError(
        f"endpoint opt-in changed rollout: action={action_max_abs}, logprob={logprob_max_abs}"
    )
if "dvac_telemetry" in result_off:
    raise RuntimeError("endpoint telemetry appeared while disabled")
trace = result_on["dvac_telemetry"]["z_endpoint"]
if tuple(trace.shape) != (1, 10, 24, 14):
    raise RuntimeError(f"unexpected endpoint shape: {tuple(trace.shape)}")
variance = compute_endpoint_variance(trace, 5)
if tuple(variance.shape) != (1, 24):
    raise RuntimeError(f"unexpected variance shape: {tuple(variance.shape)}")
if not torch.isfinite(trace).all() or not torch.isfinite(variance).all():
    raise RuntimeError("endpoint trace or L5 variance is non-finite")
if not bool((variance > 0).any()):
    raise RuntimeError("L5 endpoint variance is identically zero")

print(json.dumps({
    "status": "FASTWAM_ACTION_DVAC_ENDPOINT_OK",
    "endpoint_shape": list(trace.shape),
    "variance_shape": list(variance.shape),
    "action_max_abs_off_vs_on": action_max_abs,
    "old_logprob_max_abs_off_vs_on": logprob_max_abs,
    "variance_min": float(variance.min()),
    "variance_max": float(variance.max()),
    "max_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    "max_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
}, sort_keys=True))
PY
