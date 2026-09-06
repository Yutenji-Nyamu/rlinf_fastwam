#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260808_v2
runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime

printf 'INSPECTED_AT\t%s\n' "$(date --iso-8601=seconds)"
if [[ -e "$run_root" || -L "$run_root" ]]; then
  printf 'RUN_ROOT\tEXISTS\n'
else
  printf 'RUN_ROOT\tABSENT\n'
fi
find "$runtime_root" -maxdepth 1 -type f -printf '%f\t%s\n' | sort
sha256sum "$runtime_root/source_config.yaml" "$runtime_root/resolved.yaml"

RESOLVED="$runtime_root/resolved.yaml" "${venv}/bin/python" -B - <<'PY'
import os
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.environ["RESOLVED"])
ogpo = cfg.algorithm.ogpo
print("PLACEMENT", OmegaConf.to_container(cfg.cluster.component_placement), sep="\t")
for key in (
    "total_online_rows", "start_training_rows", "utd_q", "utd_pi",
    "replay_capacity", "baseline_eval", "final_eval",
    "eval_interval_rows", "checkpoint_interval_rows",
):
    print(key.upper(), getattr(ogpo, key), sep="\t")
print("RESUME_DIR", cfg.runner.resume_dir, sep="\t")
print("CKPT_PATH", cfg.runner.ckpt_path, sep="\t")
print("TRAIN_AUTO_RESET", cfg.env.train.auto_reset, sep="\t")
PY

printf '%s\n' OGPO_FORMAL_V2_PARTIAL_INSPECT_COMPLETE
