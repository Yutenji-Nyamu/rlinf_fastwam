#!/usr/bin/env bash
set -euo pipefail

venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
stage1_root=/data/chenyiteng/results/rlinf-rlt/smoke-stage1-current-ar-2step-20260823
stage1_experiment=robotwin_adjust_bottle_rlt_stage1_current_ar_smoke2_v1
stage1_checkpoint=$stage1_root/$stage1_experiment/checkpoints/global_step_2
stage1_manifest=$stage1_root/artifacts/stage1_artifact_manifest.json
stage2_root=/data/chenyiteng/results/rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823
fresh_checkpoint=$stage2_root/fresh/robotwin_adjust_bottle_rlt_stage2_current_ar_smoke_fresh1_v1/checkpoints/global_step_1
resume_checkpoint=$stage2_root/resume/robotwin_adjust_bottle_rlt_stage2_current_ar_smoke_resume1_v1/checkpoints/global_step_2

printf '%s\n' '=== exits ==='
printf 'stage1='; cat "$stage1_root/runtime/exit_code.txt"
printf 'stage2_fresh='; cat "$stage2_root/fresh/exit_code.txt"
printf 'stage2_resume='; cat "$stage2_root/resume/exit_code.txt"

printf '%s\n' '=== manifest ==='
cat "$stage1_manifest"

printf '%s\n' '=== sidecars ==='
"$venv/bin/python" -B - "$fresh_checkpoint" "$resume_checkpoint" <<'PY'
import json
import sys
from pathlib import Path

import torch

for checkpoint in map(Path, sys.argv[1:]):
    base = checkpoint / "actor/sac_components/rlt_trainer_state"
    print(checkpoint)
    print((base / "complete.json").read_text())
    for rank in range(2):
        state = torch.load(base / f"checkpoint_rank_{rank}.pt", map_location="cpu", weights_only=False)
        compact = {
            key: state.get(key)
            for key in (
                "rank",
                "actor_world_size",
                "saved_runner_step",
                "update_step",
                "local_total_transitions_added",
                "global_warmup_ready_total_transitions",
                "pending_update_budget",
            )
        }
        print(json.dumps(compact, sort_keys=True))
PY

printf '%s\n' '=== sizes ==='
du -sh "$stage1_checkpoint" "$fresh_checkpoint" "$resume_checkpoint"

printf '%s\n' '=== resources ==='
for csv in "$stage1_root/runtime/resource.csv" "$stage2_root/fresh/resource.csv" "$stage2_root/resume/resource.csv"; do
  printf '%s ' "$csv"
  wc -l < "$csv"
  head -n 2 "$csv"
  tail -n 2 "$csv"
done

printf '%s\n' 'RLT_CURRENT_SMOKE_POSTFLIGHT_PROBE_OK'
