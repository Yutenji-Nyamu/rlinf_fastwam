"""Download lightweight final evidence for the stopped SZ PPO pair."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(r"C:\Users\86136\Documents\rl")
BASE = ROOT / "local_scripts" / "download_sz_action_adv_prism_stopped_20260828.py"


def load_base():
    spec = importlib.util.spec_from_file_location("download_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()
base.DEFAULT_DEST = (
    ROOT / "docs" / "rlinf-shenzhen-experiment-expansion" / "evidence"
    / "ppo-control-dvac-stopped-20260901"
)
base.REMOTE_ROOT = "/data/chenyiteng/results/rlinf-shenzhen/ppo/runs"
base.RUNS = {
    "control": "ppo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-localshard-v1",
    "dvac": "ppo-dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v1",
}


if __name__ == "__main__":
    base.main()
