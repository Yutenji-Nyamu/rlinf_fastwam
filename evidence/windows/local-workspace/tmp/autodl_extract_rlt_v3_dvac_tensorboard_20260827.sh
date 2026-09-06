#!/usr/bin/env bash
set -euo pipefail

run=/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3
event_file=$(find "$run/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents.*' | head -n 1)
/root/autodl-tmp/RLinf/.venv/bin/python - "$event_file" <<'PY'
import json
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_file = sys.argv[1]
acc = EventAccumulator(event_file, size_guidance={"scalars": 0})
acc.Reload()
wanted_suffixes = {
    "rlt_dvac/weight_p05",
    "rlt_dvac/weight_median",
    "rlt_dvac/weight_mean",
    "rlt_dvac/weight_p95",
    "rlt_dvac/downweighted_fraction",
    "rlt_dvac/upweighted_fraction",
    "rlt_dvac/min_weight_fraction",
    "rlt_dvac/max_weight_fraction",
    "rlt_dvac/weight_ess_ratio",
    "rlt_dvac/top20_weight_mass",
    "rlt_dvac/z_std",
    "rlt_dvac/query_mean_z_std",
    "rlt_dvac/within_query_z_std",
    "rlt_dvac_bc/success_unweighted_loss",
    "rlt_dvac_bc/success_weighted_loss",
    "rlt_dvac_bc/success_weight_mean",
    "rlt_dvac_bc/success_weight_ess_ratio",
    "rlt_dvac_bc/success_executed_ref_mse",
    "rlt_dvac_bc/executed_target_ratio",
    "actor/bc_loss",
    "actor/bc_unweighted_loss",
    "actor/bc_ref_loss",
    "actor/weighted_bc",
    "actor/q_pi",
    "actor/weighted_q",
    "actor/bc_weight",
    "actor/q_weight",
}
result = {}
for tag in sorted(acc.Tags().get("scalars", [])):
    if not any(tag.endswith(suffix) for suffix in wanted_suffixes):
        continue
    values = acc.Scalars(tag)
    if not values:
        continue
    latest = values[-1]
    tail = values[-20:]
    result[tag] = {
        "count": len(values),
        "first_step": values[0].step,
        "latest_step": latest.step,
        "latest": latest.value,
        "tail20_mean": sum(item.value for item in tail) / len(tail),
        "all_mean": sum(item.value for item in values) / len(values),
    }
print(json.dumps(result, indent=2, sort_keys=True))
PY
