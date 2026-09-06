from __future__ import annotations

import json
from pathlib import Path

import torch


root = Path(
    "/root/autodl-tmp/experiments/"
    "rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v2/"
    "robotwin_adjust_bottle_rlt_teacher_dvac_w0to2_smoke_8env1c_v2/"
    "checkpoints/global_step_1/actor/sac_components/rlt_trainer_state"
)
for rank in (0, 1):
    state = torch.load(root / f"checkpoint_rank_{rank}.pt", map_location="cpu")
    print(f"rank={rank}")
    print(f"keys={sorted(state)}")
    for key in (
        "update_step",
        "critic_updates_run",
        "actor_updates_run",
        "rlt_dvac_baseline_mean",
        "rlt_dvac_baseline_std",
        "rlt_dvac_baseline_count",
        "rlt_dvac_baseline_frozen",
    ):
        print(f"{key}={state.get(key)!r}")
    print(f"rlt_dvac_baseline={state.get('rlt_dvac_baseline')!r}")
print("complete_json=" + json.dumps(json.loads((root / "rlt_trainer_state_complete.json").read_text()), sort_keys=True))
