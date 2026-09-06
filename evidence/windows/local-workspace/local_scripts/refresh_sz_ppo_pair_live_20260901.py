"""Refresh the existing dated PPO snapshot without creating another figure set."""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
from pathlib import Path


ROOT = Path(r"C:\Users\86136\Documents\rl")
DEST = ROOT / "docs/rlinf-shenzhen-experiment-expansion/evidence/ppo-control-dvac-live-20260901"
SOURCE = ROOT / "local_scripts/download_sz_ppo_pair_live_20260831.py"

spec = importlib.util.spec_from_file_location("ppo_live_downloader", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("PPO live downloader unavailable")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)

with tempfile.TemporaryDirectory(prefix="ppo-live-refresh-") as temp:
    fresh = Path(temp) / "snapshot"
    downloader.DEST = fresh
    downloader.main()
    for relative in (
        "raw/control/runtime/driver.log",
        "raw/control/runtime/resource.csv",
        "raw/dvac/runtime/driver.log",
        "raw/dvac/runtime/resource.csv",
        "live_status.txt",
        "download_manifest.json",
    ):
        source = fresh / relative
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
