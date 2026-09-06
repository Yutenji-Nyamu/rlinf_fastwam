"""Render a compact current-pair view using the established GRPO renderer."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(r"C:\Users\86136\Documents\rl")
OUT = ROOT / "docs/rlinf-shenzhen-grpo-dvac-action-adv/evidence/current-action-mid-st-narrow-live-20260830-1635"
SOURCE = ROOT / "local_scripts/render_sz_six_2gpu_grpo_comparison_20260829.py"

spec = importlib.util.spec_from_file_location("grpo_renderer", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("renderer unavailable")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)

renderer.OUT = OUT
renderer.RUNS = {
    "action_mid": {
        "name": "Action-Adv [0.5,1.5]",
        "color": "#E69F00",
        "pattern": None,
        "marker": "diamond",
        "path": OUT / "raw/action_mid/runtime/driver.log",
    },
    "st_narrow": {
        "name": "ST-DVAC [0.8,1.2]",
        "color": "#00796B",
        "pattern": (14, 7),
        "marker": "square",
        "path": OUT / "raw/st_narrow/runtime/driver.log",
    },
}
renderer.main()

old = OUT / "01_all_six_2gpu_grpo_raw_ma5_ma10_fixed32.png"
new = OUT / "01_current_pair_raw_ma5_ma10_fixed32.png"
old.replace(new)
print(new)
