#!/usr/bin/env bash
set -u

base=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run/dsrl-current-formal-200c-v2/checkpoints
py=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python

"$py" - "$base" <<'PY'
import json
import sys
from pathlib import Path
import torch

base = Path(sys.argv[1])

def summarize(value, depth=0, key=None):
    if key == "target_shadow_f32" and isinstance(value, dict):
        return {
            "parameter_count": len(value),
            "tensor_numel": sum(v.numel() for v in value.values() if isinstance(v, torch.Tensor)),
            "all_fp32": all((not isinstance(v, torch.Tensor)) or v.dtype == torch.float32 for v in value.values()),
        }
    if depth > 3:
        return type(value).__name__
    if isinstance(value, dict):
        return {str(k): summarize(v, depth + 1, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) <= 16:
            return [summarize(v, depth + 1) for v in value]
        return {"type": type(value).__name__, "len": len(value)}
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return {"type": "Tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__

out = {}
for ckpt in sorted(base.glob("global_step_*")):
    state_path = ckpt / "actor/sac_components/dsrl_trainer_state_rank_0.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    out[ckpt.name] = summarize(state)
print(json.dumps(out, indent=2, sort_keys=True))
PY
