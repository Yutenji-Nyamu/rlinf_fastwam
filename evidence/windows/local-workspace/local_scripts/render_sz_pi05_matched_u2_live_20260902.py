"""Render the live matched-update pi0.5 GRPO pair."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(r"C:\Users\86136\Documents\rl")
OUT = ROOT / "docs/rlinf-shenzhen-pi05-robotwin/evidence/pi05-grpo-matched-u2-live-20260902-brief"
SOURCE = ROOT / "local_scripts/render_sz_six_2gpu_grpo_comparison_20260829.py"

spec = importlib.util.spec_from_file_location("grpo_renderer", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("renderer unavailable")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)

renderer.OUT = OUT
renderer.TITLE = "pi0.5 GRPO — matched pi0 update budget"
renderer.SUBTITLE = "Both: 64 env x rollout4, G8, GB1024/MB32/update2, M5; fixed-32 every 5 steps."
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
        "color": "#D55E00",
        "pattern": (14, 7),
        "marker": "diamond",
        "path": OUT / "raw/dvac/runtime/driver.log",
    },
}
renderer.main()

old = OUT / "01_all_six_2gpu_grpo_raw_ma5_ma10_fixed32.png"
new = OUT / "01_pi05_matched_u2_control_vs_dvac.png"
old.replace(new)
print(new)
