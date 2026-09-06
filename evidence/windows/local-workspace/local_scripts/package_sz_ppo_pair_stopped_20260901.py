"""Build one small, high-information ZIP for the stopped SZ PPO pair."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(r"C:\Users\86136\Documents\rl")
SNAPSHOT = ROOT / "docs/rlinf-shenzhen-experiment-expansion/evidence/ppo-control-dvac-stopped-20260901"
NAME = "shenzhen_pi0_ppo_control_dvac_w0p5to1p5_stopped_pair_light_evidence_20260901"
ZIP_PATH = ROOT / "exports" / f"{NAME}.zip"


def main() -> None:
    if ZIP_PATH.exists():
        raise FileExistsError(f"refusing to overwrite {ZIP_PATH}")
    required = [
        SNAPSHOT / "summary.json",
        SNAPSHOT / "curves.csv",
        SNAPSHOT / "01_ppo_control_vs_dvac_final_raw_ma5_ma10_fixed32.png",
        SNAPSHOT / "raw/control/runtime/driver.log",
        SNAPSHOT / "raw/dvac/runtime/driver.log",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in SNAPSHOT.rglob("*") if path.is_file())
    manifest = [
        {
            "path": str(path.relative_to(SNAPSHOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in files
    ]
    manifest_path = SNAPSHOT / "file_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files.append(manifest_path)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, arcname=f"{NAME}/{path.relative_to(SNAPSHOT)}")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity failure at {bad}")
    print(json.dumps({"zip": str(ZIP_PATH), "bytes": ZIP_PATH.stat().st_size, "files": len(files)}))


if __name__ == "__main__":
    main()
