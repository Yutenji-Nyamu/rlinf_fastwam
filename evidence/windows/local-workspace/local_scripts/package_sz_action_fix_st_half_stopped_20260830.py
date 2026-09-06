"""Package high-information closeout bundles for the two replaced 2-GPU runs."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(r"C:\Users\86136\Documents\rl")
BASE = ROOT / "local_scripts" / "package_sz_action_adv_prism_stopped_20260828.py"


def load_base():
    spec = importlib.util.spec_from_file_location("closeout_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()
base.DEFAULT_SNAPSHOT = (
    ROOT
    / "docs"
    / "rlinf-shenzhen-grpo-dvac-action-adv"
    / "evidence"
    / "action-fix-st-half-stopped-20260830"
)
base.RUNS = {
    "action_fix": {
        "title": "GRPO-DVAC Action-Adv Fix [0,2]",
        "name": "shenzhen_grpo_dvac_action_adv_fix_w0to2_stopped_light_evidence_20260830",
        "remote_run": (
            "/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/"
            "dvac-action-adv-fix-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1"
        ),
        "gpus": (4, 5),
        "termination": (
            "user-authorized exact stop to replace this run with Action-Adv Fix [0.5,1.5]"
        ),
        "caveat": (
            "This is the corrected H-sum Action-Adv implementation with weights [0,2]. "
            "Checkpoints, videos, Ray session logs and per-action tensors are intentionally excluded."
        ),
    },
    "st_half": {
        "title": "GRPO-DVAC ST [0.5,1.5]",
        "name": "shenzhen_grpo_dvac_st_w0p5to1p5_ray_memory_exit_step53_light_evidence_20260830",
        "remote_run": (
            "/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/"
            "dvac-st-global-z-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1"
        ),
        "gpus": (6, 7),
        "termination": (
            "Ray userspace memory monitor killed workers at 95.5891%; exit 255 after complete Step 53"
        ),
        "caveat": (
            "The run ended through Ray's host-memory monitor, not a numerical failure. "
            "Checkpoints, videos, Ray session logs and per-action tensors are intentionally excluded."
        ),
    },
}


if __name__ == "__main__":
    base.main()
