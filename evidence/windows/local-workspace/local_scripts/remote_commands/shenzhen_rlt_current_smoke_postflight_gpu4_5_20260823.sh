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
summary=$stage2_root/postflight.json

test "$(cat "$stage1_root/runtime/exit_code.txt")" = 0
test "$(cat "$stage2_root/fresh/exit_code.txt")" = 0
test "$(cat "$stage2_root/resume/exit_code.txt")" = 0
test -s "$stage1_checkpoint/actor/model_state_dict/full_weights.pt"
test -s "$stage1_checkpoint/actor/dcp_checkpoint/.metadata"
test -s "$stage1_manifest"

"$venv/bin/python" -B - \
  "$stage1_manifest" "$fresh_checkpoint" "$resume_checkpoint" "$summary" \
  "$stage1_root/runtime/resource.csv" \
  "$stage2_root/fresh/resource.csv" \
  "$stage2_root/resume/resource.csv" <<'PY'
import csv
import json
import sys
from pathlib import Path

import torch

manifest_path, fresh, resume, output, *resource_paths = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
assert manifest["model_identity"] == "exact_pi0"
assert manifest["reconstruction"] == "current_causal_ar"
assert manifest["train_vla"] is False
assert manifest["dataset"]["episodes"] == 50
assert manifest["dataset"]["frames"] == 7188
assert manifest["dataset"]["action_dim"] == 14
assert manifest["full_weights"]["size_bytes"] == 9556454857

result = {
    "stage1": {
        "manifest_id": manifest["id"],
        "dataset": manifest["dataset"],
        "full_weights": manifest["full_weights"],
        "model_identity": manifest["model_identity"],
        "reconstruction": manifest["reconstruction"],
        "train_vla": manifest["train_vla"],
    },
    "stage2": {},
    "resources": {},
}

for label, checkpoint, expected in (
    ("fresh", fresh, {"runner_step": 1, "update_step": 8, "local": 4}),
    ("resume", resume, {"runner_step": 2, "update_step": 28, "local": 8}),
):
    base = checkpoint / "actor/sac_components/rlt_trainer_state"
    complete = json.loads((base / "complete.json").read_text())
    assert complete["complete"] is True
    assert complete["actor_world_size"] == 2
    assert complete["saved_runner_step"] == expected["runner_step"]
    assert complete["update_step"] == expected["update_step"]
    ranks = []
    for rank in range(2):
        state = torch.load(
            base / f"checkpoint_rank_{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert state["rank"] == rank
        assert state["actor_world_size"] == 2
        assert state["saved_runner_step"] == expected["runner_step"]
        assert state["update_step"] == expected["update_step"]
        assert state["local_total_transitions_added"] == expected["local"]
        assert state["global_warmup_ready_total_transitions"] == 8
        ranks.append(
            {
                "rank": rank,
                "local_total_transitions_added": state["local_total_transitions_added"],
                "saved_runner_step": state["saved_runner_step"],
                "update_step": state["update_step"],
            }
        )
    result["stage2"][label] = {
        "checkpoint": str(checkpoint),
        "global_total_transitions_added": sum(
            item["local_total_transitions_added"] for item in ranks
        ),
        "ranks": ranks,
    }

for path in resource_paths:
    rows = list(csv.DictReader(path.open(newline="")))
    live = [row for row in rows if row.get("driver_alive") == "1"]
    assert live
    assert all(int(row.get("cgroup_oom") or 0) == 0 for row in live)
    assert all(int(row.get("cgroup_oom_kill") or 0) == 0 for row in live)
    result["resources"][str(path)] = {
        "samples": len(live),
        "min_host_available_gib": min(
            int(row["host_mem_available_kib"]) for row in live
        ) / 1024 / 1024,
        "max_cgroup_gib": max(
            int(row["cgroup_memory_current_bytes"]) for row in live
        ) / 1024**3,
        "max_gpu4_mib": max(int(row.get("gpu4_used_mib") or 0) for row in live),
        "max_gpu5_mib": max(int(row.get("gpu5_used_mib") or 0) for row in live),
    }

output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
PY

du -sh "$stage1_checkpoint" "$fresh_checkpoint" "$resume_checkpoint"
printf '%s\n' 'RLT_CURRENT_SMOKE_POSTFLIGHT_OK'
