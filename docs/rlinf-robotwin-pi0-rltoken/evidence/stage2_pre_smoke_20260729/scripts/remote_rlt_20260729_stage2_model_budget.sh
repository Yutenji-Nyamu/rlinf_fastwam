#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
evidence_root=/root/autodl-tmp/experiment_exports/rlt_stage2_pre_smoke_20260729_v1
output="${evidence_root}/model_replay_budget.json"

test "$(git -C "${repo}" branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git -C "${repo}" rev-parse HEAD)" = \
  6df42bf488ef10d9c7eb2f89584bc5ab7543a08a
test -d "${evidence_root}"
test ! -e "${output}"

if pgrep -af \
  'train_embodied_agent|ray::|raylet|gcs_server|robotwin_adjust_bottle_rlt_stage2' \
  | grep -v -E 'pgrep -af|stage2_model_budget' >/dev/null
then
  printf '%s\n' "Refusing model-budget audit while an RLT/Ray process is active." >&2
  exit 1
fi

PYTHONPATH="${repo}" PYTHONDONTWRITEBYTECODE=1 \
"${venv}/bin/python" -B - "${output}" <<'PY'
import json
import sys
from pathlib import Path

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy

output = Path(sys.argv[1])
model = RLTMLPPolicy(
    z_dim=2048,
    proprio_dim=14,
    action_dim=14,
    num_action_chunks=10,
    ref_num_action_chunks=10,
    add_q_head=True,
    q_head_type="default",
    fixed_std=0.002,
)

named = dict(model.named_parameters())
critic_params = sum(
    parameter.numel()
    for name, parameter in named.items()
    if "q_head" in name.split(".")
)
actor_params = sum(
    parameter.numel()
    for name, parameter in named.items()
    if "q_head" not in name.split(".")
)
total_params = sum(parameter.numel() for parameter in named.values())
buffers = sum(buffer.numel() for buffer in model.buffers())

# Compact replay keeps current/next (z_rl, proprio, ref_chunk), the executed
# action chunk, reward chunk, intervention flags, and terminal flags.
float32_values_per_row = (
    2 * 2048  # current + next z_rl
    + 2 * 14  # current + next proprio
    + 2 * 10 * 14  # current + next reference chunks
    + 10 * 14  # executed action chunk
    + 10  # primitive reward chunk
)
bool_values_per_row = 10 * 14 + 3
raw_row_bytes = float32_values_per_row * 4 + bool_values_per_row

payload = {
    "passed": True,
    "model_contract": {
        "z_dim": 2048,
        "proprio_dim": 14,
        "action_dim": 14,
        "action_chunk": 10,
        "actor_input_dim": 2048 + 14 + 10 * 14,
        "critic_input_dim_before_action": 2048 + 14,
        "hidden_dims": [256, 256, 256],
        "q_heads": 2,
        "dtype": "fp32",
    },
    "parameters": {
        "actor_optimizer_group": actor_params,
        "critic_optimizer_group": critic_params,
        "model_total": total_params,
        "target_model_total": total_params,
        "persistent_buffers": buffers,
        "raw_model_plus_target_bytes_fp32": total_params * 4 * 2,
        "upper_bound_model_target_grad_adam_bytes_fp32": total_params * 4 * 5,
    },
    "compact_replay_raw_estimate": {
        "float32_values_per_row": float32_values_per_row,
        "bool_values_per_row": bool_values_per_row,
        "raw_bytes_per_row": raw_row_bytes,
        "smoke_64_rows_per_rank_bytes": raw_row_bytes * 64,
        "formal_15000_rows_per_rank_bytes": raw_row_bytes * 15000,
        "formal_two_rank_raw_bytes": raw_row_bytes * 15000 * 2,
        "scope": (
            "Tensor payload estimate only; Python objects, allocator overhead, "
            "trajectory staging, dataloader batches, and frozen pi0 are excluded."
        ),
    },
}
output.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

sha256sum "${output}" >"${output}.sha256"
printf '%s\n' STAGE2_MODEL_REPLAY_BUDGET_OK
cat "${output}"
