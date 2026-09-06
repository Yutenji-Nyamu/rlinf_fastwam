#!/usr/bin/env bash
set -euo pipefail
run=/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s0p5_smoke_20260827_v2
runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s0p5_smoke_20260827_v2/runtime
out=/root/autodl-tmp/experiment_exports/rlt_dvac_pure_smoke_v2_evidence_20260827
archive=/root/autodl-tmp/experiment_exports/rlt_dvac_pure_smoke_v2_evidence_20260827.tar.gz
venv=/root/autodl-tmp/RLinf/.venv

test ! -e "$out"
test ! -e "$archive"
test "$(cat "$runtime/exit_code.txt")" = 0
mkdir -p "$out/traces"
cp "$run/metrics.log" "$out/metrics.log"
cp "$run/tensorboard/config.yaml" "$out/resolved_runtime_config.yaml"
for name in exact_command.txt source_head.txt started_at.txt finished_at.txt exit_code.txt resources.csv resources_after.txt; do
  cp "$runtime/$name" "$out/$name"
done
tail -n 1000 "$runtime/foreground.log" >"$out/foreground_tail_1000.log"
find "$run" -type f -name 'update_*.npz' -exec cp '{}' "$out/traces/" \;
checkpoint=$(find "$run" -type d -path '*/checkpoints/global_step_1' | head -n 1)
test -n "$checkpoint"
complete=$(find "$checkpoint" -type f -name '*complete*.json' | head -n 1)
test -n "$complete"
cp "$complete" "$out/rlt_trainer_state_complete.json"
printf 'checkpoint_path=%s\nfile_count=%s\nbytes=%s\n' \
  "$checkpoint" \
  "$(find "$checkpoint" -type f | wc -l)" \
  "$(du -sb "$checkpoint" | awk '{print $1}')" \
  >"$out/checkpoint_summary.txt"

event=$(find "$run/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents.*' | head -n 1)
"$venv/bin/python" - "$event" "$runtime/resources.csv" >"$out/smoke_summary.json" <<'PY'
import csv
import json
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

acc = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
acc.Reload()
suffixes = {
    "env/success_once",
    "train/actor/rlt_dvac/weight_mean",
    "train/actor/rlt_dvac/weight_p05",
    "train/actor/rlt_dvac/weight_p95",
    "train/actor/rlt_dvac/weight_ess_ratio",
    "train/actor/rlt_dvac/top20_weight_mass",
    "train/actor/rlt_dvac/success_target_reference",
    "train/actor/rlt_dvac_bc/executed_target_ratio",
    "train/actor/rlt_dvac_bc/success_query_count",
    "train/actor/rlt_dvac_bc/success_weight_mean",
    "train/actor/rlt_dvac_bc/success_weight_ess_ratio",
    "train/actor/rlt_dvac_bc/success_unweighted_loss",
    "train/actor/rlt_dvac_bc/success_weighted_loss",
}
scalars = {}
for tag in suffixes:
    values = acc.Scalars(tag)
    if values:
        scalars[tag] = values[-1].value
rows = list(csv.DictReader(open(sys.argv[2], encoding="utf-8")))
result = {
    "exit_code": 0,
    "global_step": 1,
    "scalars": scalars,
    "resources": {
        "samples": len(rows),
        "ram_peak_gib": max(int(row["memory_current"]) for row in rows) / 2**30,
        "gpu0_peak_gib": max(float(row["gpu0_mib"]) for row in rows) / 1024,
        "gpu1_peak_gib": max(float(row["gpu1_mib"]) for row in rows) / 1024,
    },
}
print(json.dumps(result, indent=2, sort_keys=True))
PY

find "$out" -type f -printf '%s %P\n' | sort -k2 >"$out/member_manifest.txt"
tar -czf "$archive" -C "$(dirname "$out")" "$(basename "$out")"
sha256sum "$archive"
du -h "$archive"
cat "$out/smoke_summary.json"
