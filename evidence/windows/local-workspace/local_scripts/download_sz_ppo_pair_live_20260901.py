"""Refresh the current SZ PPO pair into a dated, non-overwriting snapshot."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(r"C:\Users\86136\Documents\rl")
SOURCE = ROOT / "local_scripts/download_sz_ppo_pair_live_20260831.py"

spec = importlib.util.spec_from_file_location("ppo_live_downloader", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("PPO live downloader unavailable")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)
downloader.DEST = (
    ROOT
    / "docs/rlinf-shenzhen-experiment-expansion/evidence/ppo-control-dvac-live-20260901"
)
downloader.main()
