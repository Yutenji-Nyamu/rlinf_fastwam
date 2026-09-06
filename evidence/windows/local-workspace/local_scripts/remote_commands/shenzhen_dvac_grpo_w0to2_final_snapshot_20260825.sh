#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v5-resume30
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
EVENT=$(find "$RUN/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents*' | head -n 1)
OUT="$RUN/runtime/tensorboard_scalars_snapshot.json"
TMP="$OUT.partial"

"$PY" -B - "$EVENT" "$TMP" <<'PY'
import json, pathlib, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event, output = sys.argv[1:]
ea = EventAccumulator(event, size_guidance={"scalars": 0})
ea.Reload()
all_tags = sorted(ea.Tags().get("scalars", []))
selected_suffixes = (
    "success_once", "approx_kl", "clip_fraction", "grad_norm",
    "policy_loss_abs", "dvac_weight_mean", "dvac_weight_sq_mean",
    "dvac_weight_std", "dvac_weight_ess_fraction",
    "dvac_z_low_clip_fraction", "dvac_z_high_clip_fraction", "dvac_warmup",
)
selected = [tag for tag in all_tags if tag.endswith(selected_suffixes)]
payload = {"__all_tags__": all_tags, "__event__": event}
for tag in selected:
    payload[tag] = [
        {"step": int(x.step), "wall_time": float(x.wall_time), "value": float(x.value)}
        for x in ea.Scalars(tag)
    ]
pathlib.Path(output).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
PY
mv -- "$TMP" "$OUT"

TZ=Asia/Shanghai date --iso-8601=seconds
printf 'fatal_matches='; grep -aEci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$RUN/runtime/driver.log" || true
grep -aE 'Global Step:|Generating Rollout Epochs:' "$RUN/runtime/driver.log" | tail -n 12 || true
stat -c '%s %y %n' \
  "$RUN/runtime/driver.log" \
  "$RUN/runtime/resource.csv" \
  "$RUN/runtime/resolved.yaml" \
  "$RUN/runtime/launch_manifest.txt" \
  "$RUN/runtime/resume_parity.json" \
  "$RUN/runtime/contract.json" \
  "$RUN/runtime/stopped_by_user_for_w0to5.txt" \
  "$OUT" "$EVENT"
echo SZ_DVAC_GRPO_W0TO2_FINAL_SNAPSHOT_OK
