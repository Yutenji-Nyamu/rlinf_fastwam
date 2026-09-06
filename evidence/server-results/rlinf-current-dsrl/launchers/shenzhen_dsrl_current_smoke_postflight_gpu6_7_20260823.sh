#!/usr/bin/env bash
set -euo pipefail

venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
run_root=/data/chenyiteng/results/rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823
fresh_log=$run_root/fresh
resume_log=$run_root/resume
checkpoint1=$fresh_log/dsrl-fresh-step1/checkpoints/global_step_1
checkpoint2=$resume_log/dsrl-resume-step2/checkpoints/global_step_2
summary=$run_root/postflight.json

test "$(cat "$fresh_log/exit_code.txt")" = 0
test "$(cat "$resume_log/exit_code.txt")" = 0
test -s "$checkpoint1/actor/local_shard_checkpoint/checkpoint_rank_0.pt"
test -s "$checkpoint1/actor/local_shard_checkpoint/checkpoint_rank_1.pt"
test -s "$checkpoint2/actor/local_shard_checkpoint/checkpoint_rank_0.pt"
test -s "$checkpoint2/actor/local_shard_checkpoint/checkpoint_rank_1.pt"
test -d "$checkpoint1/actor/sac_components/alpha"
test -d "$checkpoint2/actor/sac_components/alpha"

for checkpoint in "$checkpoint1" "$checkpoint2"; do
  for rank in 0 1; do
    test -s "$checkpoint/actor/sac_components/dsrl_trainer_state_rank_${rank}.pt"
    test -s "$checkpoint/actor/sac_components/target_model/checkpoint_rank_${rank}.pt"
    test -s "$checkpoint/actor/sac_components/replay_buffer/rank_${rank}/dsrl_transition_replay.pt"
  done
done

"$venv/bin/python" -B - "$checkpoint1" "$checkpoint2" "$summary" <<'PY'
import json
import sys
from pathlib import Path

import torch

checkpoint1, checkpoint2, output = map(Path, sys.argv[1:])
expected = {
    checkpoint1: {"update_step": 800, "global_resident": 40},
    checkpoint2: {"update_step": 1520, "global_resident": 76},
}
result = {}
for checkpoint, contract in expected.items():
    ranks = []
    for rank in range(2):
        base = checkpoint / "actor/sac_components"
        state = torch.load(
            base / f"dsrl_trainer_state_rank_{rank}.pt",
            map_location="cpu",
            weights_only=True,
        )
        replay = torch.load(
            base / f"replay_buffer/rank_{rank}/dsrl_transition_replay.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert state["schema_version"] == 1
        assert state["rank"] == rank
        assert state["world_size"] == 2
        assert state["policy_phase"] == 1
        assert state["flat_replay"] is True
        assert state["pending_local_new_transitions"] == 0
        assert state["update_step"] == contract["update_step"]
        assert replay["local_capacity"] == 12500
        resident = int(replay["resident_size"])
        assert resident > 0
        assert replay["total_inserted"] == resident
        assert replay["write_cursor"] == resident
        shadows = state["target_shadow_f32"]
        assert shadows
        assert all(value.dtype == torch.float32 for value in shadows.values())
        ranks.append(
            {
                "rank": rank,
                "update_step": state["update_step"],
                "policy_phase": state["policy_phase"],
                "resident": replay["resident_size"],
                "target_shadow_tensors": len(shadows),
            }
        )
    assert sum(item["resident"] for item in ranks) == contract["global_resident"], (
        checkpoint,
        ranks,
    )
    result[checkpoint.name] = ranks

output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True))
PY

du -sh "$checkpoint1" "$checkpoint2"
printf '%s\n' 'DSRL_CURRENT_SMOKE_POSTFLIGHT_OK'
