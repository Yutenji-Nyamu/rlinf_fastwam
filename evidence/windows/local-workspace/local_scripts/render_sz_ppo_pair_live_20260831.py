"""Render current PPO Control versus PPO-DVAC with clearly separated colors."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(r"C:\Users\86136\Documents\rl")
OUT = ROOT / "docs/rlinf-shenzhen-experiment-expansion/evidence/ppo-control-dvac-live-20260831"
SOURCE = ROOT / "local_scripts/render_sz_six_2gpu_grpo_comparison_20260829.py"

spec = importlib.util.spec_from_file_location("ppo_renderer", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("renderer unavailable")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)

renderer.OUT = OUT
renderer.TITLE = "SZ-H100 current matched two-GPU PPO pair"
renderer.SUBTITLE = "Black = PPO Control; vermillion = PPO-DVAC Action-Adv [0.5,1.5]. Same 64x4/B1024/update2 shell."
renderer.RUNS = {
    "control": {
        "name": "PPO Control",
        "color": "#111827",
        "pattern": None,
        "marker": "circle",
        "path": OUT / "raw/control/runtime/driver.log",
    },
    "dvac": {
        "name": "PPO-DVAC [0.5,1.5]",
        "color": "#D55E00",
        "pattern": (14, 7),
        "marker": "diamond",
        "path": OUT / "raw/dvac/runtime/driver.log",
    },
}
renderer.main()

old = OUT / "01_all_six_2gpu_grpo_raw_ma5_ma10_fixed32.png"
new = OUT / "01_ppo_control_vs_dvac_raw_ma5_ma10_fixed32.png"
old.replace(new)
print(new)
