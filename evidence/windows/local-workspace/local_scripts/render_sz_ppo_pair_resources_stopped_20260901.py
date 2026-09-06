"""Render compact resource figures for the stopped SZ PPO pair."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(r"C:\Users\86136\Documents\rl")
SNAPSHOT = ROOT / "docs/rlinf-shenzhen-experiment-expansion/evidence/ppo-control-dvac-stopped-20260901"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot_base = load(ROOT / "local_scripts/render_shenzhen_grpo_vs_ppo_20260823.py", "plot_base")
package_base = load(ROOT / "local_scripts/package_sz_action_adv_prism_stopped_20260828.py", "package_base")

results = {
    "control": package_base.render_resources(
        plot_base,
        "PPO Control",
        SNAPSHOT / "raw/control/runtime/resource.csv",
        (4, 5),
        SNAPSHOT / "02_ppo_control_resources.png",
    ),
    "dvac": package_base.render_resources(
        plot_base,
        "PPO-DVAC Action-Adv [0.5,1.5]",
        SNAPSHOT / "raw/dvac/runtime/resource.csv",
        (6, 7),
        SNAPSHOT / "03_ppo_dvac_resources.png",
    ),
}
print(results)
