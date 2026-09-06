"""Render the live pi0.5 GRPO Control/DVAC pair."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(r"C:\Users\86136\Documents\rl")
OUT = ROOT / "docs/rlinf-shenzhen-pi05-robotwin/evidence/pi05-grpo-pair-live-20260901-1804"
SOURCE = ROOT / "local_scripts/render_sz_six_2gpu_grpo_comparison_20260829.py"

spec = importlib.util.spec_from_file_location("grpo_renderer", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("renderer unavailable")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)

renderer.OUT = OUT
renderer.TITLE = "SZ-H100 pi0.5 two-GPU GRPO — live through Step 11"
renderer.SUBTITLE = "Matched 64 env x 4 rollout epochs, G8, B512, update5 and M5; fixed-32 every 5 steps."
renderer.RUNS = {
    "control": {
        "name": "pi0.5 GRPO Control",
        "color": "#111827",
        "pattern": None,
        "marker": "circle",
        "path": OUT / "raw/control/runtime/driver.log",
    },
    "dvac": {
        "name": "pi0.5 DVAC Action-Adv [0.5,1.5]",
        "color": "#E4572E",
        "pattern": (14, 7),
        "marker": "diamond",
        "path": OUT / "raw/dvac/runtime/driver.log",
    },
}
renderer.main()

old = OUT / "01_all_six_2gpu_grpo_raw_ma5_ma10_fixed32.png"
new = OUT / "01_pi05_control_vs_dvac_raw_ma5_ma10_fixed32.png"
old.replace(new)
print(new)
