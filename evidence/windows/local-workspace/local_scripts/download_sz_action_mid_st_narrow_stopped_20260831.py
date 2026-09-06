"""Download lightweight final evidence for the two GRPO-DVAC runs stopped on 2026-08-31."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(r"C:\Users\86136\Documents\rl")
BASE = ROOT / "local_scripts" / "download_sz_action_fix_st_half_stopped_20260830.py"


def load_base():
    spec = importlib.util.spec_from_file_location("download_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()
base.DEFAULT_DEST = (
    ROOT / "docs" / "rlinf-shenzhen-grpo-dvac-action-adv" / "evidence"
    / "action-mid-st-narrow-stopped-20260831"
)
base.RUNS = {
    "action_mid": "dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1",
    "st_narrow": "dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v2",
}


if __name__ == "__main__":
    base.main()
