"""Render and ZIP final high-information evidence for the stopped 2026-08-31 pair."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(r"C:\Users\86136\Documents\rl")
BASE = ROOT / "local_scripts" / "package_sz_action_adv_prism_stopped_20260828.py"


def load_base():
    spec = importlib.util.spec_from_file_location("package_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()
base.DEFAULT_SNAPSHOT = (
    ROOT / "docs" / "rlinf-shenzhen-grpo-dvac-action-adv" / "evidence"
    / "action-mid-st-narrow-stopped-20260831"
)
base.RUNS = {
    "action_mid": {
        "title": "GRPO-DVAC Action-Adv Fix [0.5,1.5]",
        "name": "shenzhen_grpo_dvac_action_adv_fix_w0p5to1p5_step62_light_evidence_20260831",
        "remote_run": "/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1",
        "gpus": (4, 5),
        "termination": "user-authorized exact stop after complete Step 62 to replace it with PPO Control",
        "caveat": "Corrected H-sum Action-Adv implementation; checkpoints, videos, Ray logs and per-action tensors are excluded.",
    },
    "st_narrow": {
        "title": "GRPO-DVAC ST [0.8,1.2]",
        "name": "shenzhen_grpo_dvac_st_w0p8to1p2_step46_light_evidence_20260831",
        "remote_run": "/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v2",
        "gpus": (6, 7),
        "termination": "user-authorized exact stop after complete Step 46 to replace it with PPO-DVAC [0.5,1.5]",
        "caveat": "Local-shard run; checkpoints, videos, Ray logs and per-action tensors are excluded.",
    },
}


if __name__ == "__main__":
    base.main()
