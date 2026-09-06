#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260807_v1
runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime
resolved="$runtime_root/resolved.yaml"

test -f "$resolved"
test ! -e "$run_root"
test ! -e "$runtime_root/started_at.txt"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk 'NF')"

printf 'STATUS_AT\t%s\n' "$(date --iso-8601=seconds)"
printf 'RESOLVED_SHA256\t%s\n' "$(sha256sum "$resolved" | awk '{print $1}')"
printf '%s\n' '=== partial files ==='
find "$runtime_root" -maxdepth 1 -type f -printf '%s\t%y\t%f\n' | sort
printf '%s\n' '=== resolved values ==='
RESOLVED="$resolved" "${venv}/bin/python" -B - <<'PY'
import os
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.environ["RESOLVED"])
print("component_placement_type\t" + type(cfg.cluster.component_placement).__name__)
print("component_placement_repr\t" + repr(OmegaConf.to_container(cfg.cluster.component_placement)))
for key in (
    "start_training_rows",
    "total_online_rows",
    "replay_capacity",
    "utd_q",
    "utd_pi",
    "baseline_eval",
    "final_eval",
    "eval_interval_rows",
    "checkpoint_interval_rows",
):
    print(f"ogpo.{key}\t{getattr(cfg.algorithm.ogpo, key)!r}")
print(f"train_envs\t{cfg.env.train.total_num_envs!r}")
print(f"train_rollout_epoch\t{cfg.env.train.rollout_epoch!r}")
print(f"train_step_cap\t{cfg.env.train.max_steps_per_rollout_epoch!r}")
print(f"eval_envs\t{cfg.env.eval.total_num_envs!r}")
print(f"eval_rollout_epoch\t{cfg.env.eval.rollout_epoch!r}")
print(f"actor_total_training_steps\t{cfg.actor.optim.total_training_steps!r}")
PY
printf '%s\n' OGPO_FORMAL_PARTIAL_INSPECTED
