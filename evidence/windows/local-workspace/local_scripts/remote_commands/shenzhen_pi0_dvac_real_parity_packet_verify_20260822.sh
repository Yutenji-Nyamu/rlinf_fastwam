#!/usr/bin/env bash
set -euo pipefail

root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-observe-7d07a421
packet=/data/chenyiteng/results/dvac-observation/packets/pi0-adjust_bottle-real-query-gate-a-800baf80-v1
output=/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-real-query-gate-a-800baf80-v1
python=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
head=800baf80d6eab64169cf0e691eb04a681a093ee9

test "$(git -C "$root" rev-parse HEAD)" = "$head"
test "$(git -C "$root" rev-parse '@{upstream}')" = "$head"
test -z "$(git -C "$root" status --porcelain=v1)"
test "$(git -C "$root" ls-remote personal refs/heads/codex/sz-current-pi0-dvac-observe | awk '{print $1}')" = "$head"
test -d "$packet"
test ! -e "$output"

cd "$packet"
sha256sum -c SHA256SUMS
bash -n compose_command.sh
bash -n launch_gate_a.sh
test -x compose_command.sh
test -x launch_gate_a.sh
grep -Fq 'export CUDA_VISIBLE_DEVICES=2' launch_gate_a.sh
grep -Fq 'timeout --signal=INT --kill-after=120s 900s' launch_gate_a.sh
grep -Fq -- '--expected-head "$EXPECTED_HEAD"' launch_gate_a.sh
grep -Fq -- '--reset-state-id 100100052' launch_gate_a.sh
grep -Fq -- '--inference-seed 0' launch_gate_a.sh

CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none \
  "$python" - "$packet" "$output" <<'PY'
import json
import pathlib
import sys

from omegaconf import OmegaConf

packet = pathlib.Path(sys.argv[1])
output = sys.argv[2]
cfg = OmegaConf.load(packet / "resolved_config.yaml")
manifest = json.loads((packet / "manifest.json").read_text())
budget = json.loads((packet / "budget.json").read_text())
assert cfg.env.eval.total_num_envs == 1
assert cfg.env.eval.use_fixed_reset_state_ids is True
assert cfg.rollout.dvac_telemetry.enabled is False
assert cfg.runner.logger.log_path == output
assert manifest["status"] == "READY_FOR_REVIEW_NOT_EXECUTED"
assert manifest["source"]["rlinf_commit"] == "800baf80d6eab64169cf0e691eb04a681a093ee9"
assert manifest["gate"]["reset_state_id"] == 100100052
assert manifest["gate"]["physical_gpu"] == 2
assert manifest["gate"]["timeout_seconds"] == 900
assert budget["simulator_resets"] == 1
assert budget["policy_queries"] == 2
assert budget["environment_action_slots"] == 0
assert budget["total_denoising_steps"] == 8
print("PACKET_SEMANTIC_VERIFY_OK=1")
PY

find "$packet" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
echo 'REAL_OUTPUT_EXISTS=0'
echo 'GPU_RAY_MODEL_SIM_USED=0'
echo 'PI0_REAL_PARITY_PACKET_VERIFY_OK=1'
