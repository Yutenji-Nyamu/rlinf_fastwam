#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-w0to5-formal100-grpo-matched-4gpu128x4-b2048-eval5-phys4567-v1
EXPORTS=/data/chenyiteng/results/rlinf-shenzhen/grpo/exports
NAME=dvac-global-z-w0to5-step45-high-info-20260826
OUT=$EXPORTS/$NAME
ZIP=$EXPORTS/$NAME.zip
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin

test -f "$RUN/runtime/stopped_by_user_for_dual_2gpu.txt"
test ! -e "$OUT"
test ! -e "$ZIP"
install -d -m 755 "$EXPORTS"

"$VENV/bin/python" - "$RUN" "$OUT" "$ZIP" <<'PY'
from __future__ import annotations
import csv, json, shutil, sys, zipfile
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = Path(sys.argv[1])
out = Path(sys.argv[2])
zip_path = Path(sys.argv[3])
out.mkdir(parents=True)
(out / "runtime").mkdir()
(out / "tensorboard").mkdir()

runtime_names = [
    "driver.log", "resource.csv", "resolved.yaml", "contract.json", "parity.json",
    "command.txt", "launch_manifest.txt", "started_at.txt",
    "stopped_by_user_for_dual_2gpu.txt",
]
copied = []
for name in runtime_names:
    src = run / "runtime" / name
    if src.is_file():
        dst = out / "runtime" / name
        shutil.copy2(src, dst)
        copied.append(dst)

tb_config = run / "tensorboard" / "config.yaml"
if tb_config.is_file():
    dst = out / "tensorboard" / "config.yaml"
    shutil.copy2(tb_config, dst)
    copied.append(dst)
events = sorted((run / "tensorboard").glob("events.out.tfevents.*"))
for src in events:
    dst = out / "tensorboard" / src.name
    shutil.copy2(src, dst)
    copied.append(dst)

rows = []
latest = {}
if events:
    ea = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    for tag in sorted(ea.Tags().get("scalars", [])):
        values = ea.Scalars(tag)
        for value in values:
            rows.append((tag, value.step, value.wall_time, value.value))
        if values:
            value = values[-1]
            latest[tag] = {"step": value.step, "value": value.value, "wall_time": value.wall_time}

with (out / "tensorboard_scalars.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["tag", "step", "wall_time", "value"])
    writer.writerows(rows)

marker = (run / "runtime" / "stopped_by_user_for_dual_2gpu.txt").read_text(encoding="utf-8")
summary = {
    "source_run": str(run),
    "terminal_marker": marker.splitlines(),
    "last_complete_step": 45,
    "termination": "user-requested exact PGID and namespace stop before paired 2-GPU experiments",
    "tensorboard_scalar_rows": len(rows),
    "tensorboard_tags": sorted(latest),
    "latest_scalar_values": latest,
    "contents_exclude": ["checkpoints", "videos", "RoboTwin bulk data"],
}
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(out / "README.md").write_text(
    "# GRPO-DVAC weights [0,5] Step 45 high-information closeout\n\n"
    "The formal run was intentionally stopped after complete global Step 45 to make way for the paired two-GPU control and weights [0,2] runs.\n\n"
    "Included: full driver log, one-minute resource telemetry, resolved config, exact command and parity/contract records, TensorBoard event/config, a flat scalar CSV, and the exact stop marker.\n\n"
    "Excluded to keep this package small: checkpoints, videos, and RoboTwin bulk data. The retained files are sufficient to reproduce the configuration and redraw training/evaluation/resource figures.\n",
    encoding="utf-8",
)

manifest = []
for path in sorted(p for p in out.rglob("*") if p.is_file()):
    manifest.append({"path": str(path.relative_to(out)), "bytes": path.stat().st_size})
(out / "file_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        zf.write(path, arcname=f"{out.name}/{path.relative_to(out)}")
print(json.dumps({"zip": str(zip_path), "bytes": zip_path.stat().st_size, "files": len(manifest) + 1, "scalar_rows": len(rows)}))
PY

echo SZ_W0TO5_STEP45_HIGH_INFO_PACKAGE_OK
