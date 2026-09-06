set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
PYTHON_BIN=/root/autodl-tmp/RLinf/.venv/bin/python
DATA_ROOT=/root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-contract-ep0-v1
MODEL_ROOT=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
STATS_PATH="$MODEL_ROOT/physical-intelligence/robotwin/norm_stats.json"
CONFIG=examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml
EVIDENCE_ROOT=/root/autodl-tmp/experiment_exports/rlt_stage1_smoke_20260729_v1
RESOLVED_TMP=/root/autodl-tmp/tmp/rlt_stage1_scheduler_fixed_resolved_20260729.yaml
OUTPUT="$EVIDENCE_ROOT/lr_scheduler_contract.json"

test "$(git -C "$RLT_ROOT" branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git -C "$RLT_ROOT" rev-parse HEAD)" = e4127fd49e38362161eac08c551a7a98c11e9802
test "$(sha256sum "$RLT_ROOT/$CONFIG" | awk '{print $1}')" = \
  8340ef4e953877de510da18548d0a69802104b7b2f8218698cd0fb586b49a8f2
test ! -e "$RESOLVED_TMP"
test ! -e "$OUTPUT"
trap 'rm -f -- "$RESOLVED_TMP"' EXIT

cd "$RLT_ROOT"
export PYTHONPATH="$RLT_ROOT:/root/autodl-tmp/RoboTwin_RLinf"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORMS=cpu
export ROBOTWIN_RLT_CLEAN50_PATH="$DATA_ROOT"
export ROBOTWIN_PI0_BASE_PATH="$MODEL_ROOT"
export ROBOTWIN_PI0_NORM_STATS_PATH="$STATS_PATH"

"$PYTHON_BIN" -B examples/sft/train_vla_sft.py \
  --config-path "$RLT_ROOT/examples/sft/config" \
  --config-name robotwin_rlt_stage1_sft_openpi \
  --cfg job \
  --resolve \
  > "$RESOLVED_TMP"

"$PYTHON_BIN" -B - "$RESOLVED_TMP" "$OUTPUT" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler

resolved_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
cfg = OmegaConf.load(resolved_path)
optim = cfg.actor.optim

assert float(optim.lr) == 2.5e-5
assert int(optim.total_training_steps) == 2000
assert int(optim.lr_warmup_steps) == 100
assert str(optim.lr_scheduler) == "cosine"
assert "min_lr" not in optim
assert float(optim.min_lr_rate) == 0.1


def trace(*, warmup: int, total: int, min_lr=None, min_lr_rate=None):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], "lr": 2.5e-5}],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )
    defaults_lr = float(optimizer.defaults["lr"])
    group_lr_before_scheduler = float(optimizer.param_groups[0]["lr"])
    scheduler = get_lr_scheduler(
        "cosine",
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=total,
        num_cycles=0.5,
        min_lr=min_lr if min_lr is not None else 0.0,
        min_lr_rate=min_lr_rate,
    )
    selected = {0: float(scheduler.get_last_lr()[0])}
    wanted = {1, warmup, (warmup + total) // 2, total}
    for step in range(1, total + 1):
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
        if step in wanted:
            selected[step] = float(scheduler.get_last_lr()[0])
    return {
        "optimizer_defaults_lr": defaults_lr,
        "param_group_lr_before_scheduler": group_lr_before_scheduler,
        "scheduler_base_lrs": [float(value) for value in scheduler.base_lrs],
        "selected_lr_after_scheduler_step": {
            str(step): value for step, value in sorted(selected.items())
        },
    }


legacy_smoke = trace(warmup=1, total=2, min_lr=2.5e-6)
fixed_smoke = trace(warmup=1, total=2, min_lr_rate=0.1)
fixed_formal = trace(warmup=100, total=2000, min_lr_rate=0.1)

legacy_expected = {
    "0": 0.0,
    "1": 2.5e-5,
    "2": 6.25e-8,
}
fixed_smoke_expected = {
    "0": 0.0,
    "1": 2.5e-5,
    "2": 2.5e-6,
}
for key, expected in legacy_expected.items():
    actual = legacy_smoke["selected_lr_after_scheduler_step"][key]
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), (
        "legacy",
        key,
        actual,
        expected,
    )
for key, expected in fixed_smoke_expected.items():
    actual = fixed_smoke["selected_lr_after_scheduler_step"][key]
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), (
        "fixed_smoke",
        key,
        actual,
        expected,
    )
formal_expected = {
    "0": 0.0,
    "1": 2.5e-7,
    "100": 2.5e-5,
    "1050": 1.375e-5,
    "2000": 2.5e-6,
}
for key, expected in formal_expected.items():
    actual = fixed_formal["selected_lr_after_scheduler_step"][key]
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), (
        key,
        actual,
        expected,
    )
assert fixed_formal["optimizer_defaults_lr"] == 1e-3
assert fixed_formal["scheduler_base_lrs"] == [2.5e-5]

result = {
    "contract": "robotwin_rlt_stage1_lr_scheduler_v1",
    "resolved_optim": {
        "lr": float(optim.lr),
        "total_training_steps": int(optim.total_training_steps),
        "lr_warmup_steps": int(optim.lr_warmup_steps),
        "lr_scheduler": str(optim.lr_scheduler),
        "num_cycles": float(optim.num_cycles),
        "min_lr_rate": float(optim.min_lr_rate),
        "min_lr_present": "min_lr" in optim,
    },
    "legacy_smoke_negative_control": legacy_smoke,
    "fixed_smoke": fixed_smoke,
    "fixed_formal_2k": fixed_formal,
    "assertions": {
        "legacy_floor_reproduced": True,
        "fixed_smoke_floor": True,
        "fixed_formal_key_steps": True,
        "gpu_or_model_loaded": False,
    },
}
output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY

sha256sum "$RLT_ROOT/$CONFIG" "$OUTPUT"
printf '%s\n' LR_SCHEDULER_CONTRACT_OK
