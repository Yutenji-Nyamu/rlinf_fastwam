#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-smoke2-4gpu128train64eval-v1
RUNTIME="$RUN/runtime"
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
OUT="$RUNTIME/final_summary"
ARCHIVE="$RUNTIME/dvac-global-z-smoke2-light-evidence.tar.gz"

printf 'MARKER=SZ_FINALIZE_CURRENT_DVAC_GRPO_SMOKE2_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
pid=$(cat "$RUNTIME/wrapper.pid")
if kill -0 "$pid" 2>/dev/null; then
  printf '%s\n' 'wrapper still alive; not finalizing' >&2
  exit 2
fi
test "$(cat "$RUNTIME/exit_code.txt")" = 0
install -d -m 755 "$OUT"

"$VENV/bin/python" - "$RUN" "$OUT" <<'PY'
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = Path(sys.argv[1])
out = Path(sys.argv[2])

artifact_rows = []
step_weights = {}
step_z = {}
for path in sorted(run.rglob("runner_step_*.pt")):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    weights = payload["weights"].float().reshape(-1)
    z = payload["clipped_z"].float().reshape(-1)
    step = int(payload["runner_step"])
    step_weights.setdefault(step, []).append(weights)
    step_z.setdefault(step, []).append(z)
    current = payload["current_stats"]
    mean_w = float(weights.mean())
    sq_mean_w = float(weights.square().mean())
    artifact_rows.append({
        "runner_step": step,
        "actor_rank": int(payload["actor_rank"]),
        "warmup": int(bool(payload["warmup"])),
        "history_count": int(payload["history"]["history_count"]),
        "history_mean": float(payload["history"]["history_mean"]),
        "history_std": float(payload["history"]["history_std"]),
        "current_count": int(current["count"]),
        "current_mean": float(current["mean"]),
        "current_std": float(current["std"]),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": mean_w,
        "weight_std": float(weights.std(unbiased=False)),
        "weight_ess_fraction": mean_w * mean_w / sq_mean_w,
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "z_mean": float(z.mean()),
        "z_std": float(z.std(unbiased=False)),
        "z_low_clip_fraction": float((z <= -2).float().mean()),
        "z_high_clip_fraction": float((z >= 2).float().mean()),
        "artifact_bytes": path.stat().st_size,
        "artifact_path": str(path.relative_to(run)),
    })
if len(artifact_rows) != 8:
    raise RuntimeError(f"expected 8 rank-step artifacts, got {len(artifact_rows)}")
with (out / "dvac_rank_step_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(artifact_rows[0]))
    writer.writeheader()
    writer.writerows(artifact_rows)

aggregate = []
histogram_rows = []
for step in sorted({row["runner_step"] for row in artifact_rows}):
    rows = [row for row in artifact_rows if row["runner_step"] == step]
    all_weights = torch.cat(step_weights[step])
    all_z = torch.cat(step_z[step])
    quantiles = torch.quantile(
        all_weights, torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    ).tolist()
    aggregate.append({
        "runner_step": step,
        "ranks": len(rows),
        "warmup": rows[0]["warmup"],
        "history_count": rows[0]["history_count"],
        "current_mean": sum(row["current_mean"] for row in rows) / len(rows),
        "current_std": sum(row["current_std"] for row in rows) / len(rows),
        "weight_min": min(row["weight_min"] for row in rows),
        "weight_max": max(row["weight_max"] for row in rows),
        "weight_mean": sum(row["weight_mean"] for row in rows) / len(rows),
        "weight_std": sum(row["weight_std"] for row in rows) / len(rows),
        "weight_ess_fraction": sum(row["weight_ess_fraction"] for row in rows) / len(rows),
        "weight_p01": quantiles[0],
        "weight_p05": quantiles[1],
        "weight_p25": quantiles[2],
        "weight_p50": quantiles[3],
        "weight_p75": quantiles[4],
        "weight_p95": quantiles[5],
        "weight_p99": quantiles[6],
        "z_mean": float(all_z.mean()),
        "z_std": float(all_z.std(unbiased=False)),
        "z_low_clip_fraction": sum(row["z_low_clip_fraction"] for row in rows) / len(rows),
        "z_high_clip_fraction": sum(row["z_high_clip_fraction"] for row in rows) / len(rows),
    })
    counts = torch.histc(all_weights, bins=20, min=0.0, max=2.0).to(torch.int64)
    for index, count in enumerate(counts.tolist()):
        histogram_rows.append({
            "runner_step": step,
            "bin_left": index / 10,
            "bin_right": (index + 1) / 10,
            "count": count,
            "fraction": count / int(all_weights.numel()),
        })
(out / "dvac_step_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
with (out / "dvac_weight_histogram.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["runner_step", "bin_left", "bin_right", "count", "fraction"])
    writer.writeheader()
    writer.writerows(histogram_rows)

events = sorted(run.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
if not events:
    raise RuntimeError("no TensorBoard event file")
acc = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
acc.Reload()
tags = acc.Tags().get("scalars", [])
scalar_rows = []
for tag in tags:
    for event in acc.Scalars(tag):
        scalar_rows.append({"tag": tag, "step": event.step, "value": event.value, "wall_time": event.wall_time})
with (out / "tensorboard_scalars.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["tag", "step", "value", "wall_time"])
    writer.writeheader()
    writer.writerows(scalar_rows)
(out / "tensorboard_tags.json").write_text(json.dumps(tags, indent=2) + "\n")

resource_path = run / "runtime" / "resource.csv"
with resource_path.open(newline="", encoding="utf-8") as f:
    resource_rows = list(csv.DictReader(f))
live_rows = [row for row in resource_rows if row.get("driver_alive") == "1"]
resource = {
    "samples": len(live_rows),
    "host_available_min_gib": min(float(row["host_mem_available_kib"]) for row in live_rows) / 1024**2,
    "host_available_start_gib": float(live_rows[0]["host_mem_available_kib"]) / 1024**2,
    "host_available_end_gib": float(live_rows[-1]["host_mem_available_kib"]) / 1024**2,
}
for gpu in range(4):
    resource[f"gpu{gpu}_peak_used_gib"] = max(float(row[f"gpu{gpu}_used_mib"]) for row in live_rows) / 1024
    resource[f"gpu{gpu}_mean_util_pct"] = sum(float(row[f"gpu{gpu}_util_pct"]) for row in live_rows) / len(live_rows)
(out / "resource_summary.json").write_text(json.dumps(resource, indent=2) + "\n")

sidecars = sorted(run.rglob("dvac_state_rank*.json"))
checkpoint_dirs = sorted(run.rglob("global_step_2"))
inventory = {
    "event_file": str(events[-1].relative_to(run)),
    "scalar_tags": len(tags),
    "scalar_points": len(scalar_rows),
    "dvac_artifacts": len(artifact_rows),
    "dvac_sidecars": [str(path.relative_to(run)) for path in sidecars],
    "global_step_2_dirs": [str(path.relative_to(run)) for path in checkpoint_dirs],
}
(out / "artifact_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
print(json.dumps({"dvac": aggregate, "resource": resource, "inventory": inventory}, indent=2))
PY

find "$RUN" -maxdepth 10 -type f -printf '%s %p\n' | sort -n > "$OUT/file_inventory.txt"
grep -aE 'Global Step:|actor/dvac_|success|reward=|eval=|Traceback|OutOfMemory|WorkerCrashed|nonfinite|ERROR' \
  "$RUNTIME/driver.log" > "$OUT/high_signal_log.txt" || true
tar -czf "$ARCHIVE" -C "$RUN" \
  runtime/resolved.yaml runtime/contract.json runtime/command.txt runtime/launch_manifest.txt \
  runtime/started_at.txt runtime/finished_at.txt runtime/exit_code.txt runtime/driver.log \
  runtime/resource.csv runtime/final_summary

printf '%s\n' '=== CHECKPOINT AND SIDECARS ==='
find "$RUN" -maxdepth 12 \( -type d -name 'global_step_2' -o -type f -name 'dvac_state_rank*.json' -o -type f -name 'complete.json' \) -printf '%s %p\n' | sort
printf '%s\n' '=== ARCHIVE ==='
stat -c 'archive=%n bytes=%s' "$ARCHIVE"
printf 'MARKER=SZ_FINALIZE_CURRENT_DVAC_GRPO_SMOKE2_OK\n'
