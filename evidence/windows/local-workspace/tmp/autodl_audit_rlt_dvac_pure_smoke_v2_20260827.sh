#!/usr/bin/env bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
run=/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s0p5_smoke_20260827_v2
runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s0p5_smoke_20260827_v2/runtime
venv=/root/autodl-tmp/RLinf/.venv

echo PROVENANCE
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
echo RUNTIME
for name in started_at.txt finished_at.txt exit_code.txt source_head.txt; do printf '%s=' "$name"; cat "$runtime/$name"; done

echo CHECKPOINT
checkpoint=$(find "$run" -type d -path '*/checkpoints/global_step_1' | head -n 1)
test -n "$checkpoint"
echo "path=$checkpoint"
find "$checkpoint" -type f | wc -l
du -sh "$checkpoint"
find "$checkpoint" -type f \( -name '*complete*.json' -o -name '_SUCCESS' \) -printf '%s %p\n' | sort

echo TENSORBOARD
event=$(find "$run/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents.*' | head -n 1)
"$venv/bin/python" - "$event" <<'PY'
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

acc = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
acc.Reload()
for tag in sorted(acc.Tags().get("scalars", [])):
    if "rlt_dvac" in tag or tag.endswith("success_once"):
        value = acc.Scalars(tag)[-1]
        print(f"{tag} step={value.step} value={value.value:.9g}")
PY

echo RESOURCES
"$venv/bin/python" - "$runtime/resources.csv" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
memory = [int(row["memory_current"]) / 2**30 for row in rows]
gpu0 = [float(row["gpu0_mib"]) / 1024 for row in rows]
gpu1 = [float(row["gpu1_mib"]) / 1024 for row in rows]
print(f"samples={len(rows)} ram_peak_gib={max(memory):.3f} gpu0_peak_gib={max(gpu0):.3f} gpu1_peak_gib={max(gpu1):.3f}")
PY
for pattern in 'CUDA out of memory' 'OutOfMemoryError' 'WorkerCrashedError' 'ActorDiedError' 'NCCL error'; do
  printf '%s=%s ' "$pattern" "$(grep -cF "$pattern" "$runtime/foreground.log" || true)"
done
echo
cat "$runtime/resources_after.txt"
