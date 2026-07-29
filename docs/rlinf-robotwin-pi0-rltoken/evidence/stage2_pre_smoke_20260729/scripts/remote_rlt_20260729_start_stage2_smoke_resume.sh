#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
smoke_root=/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1
evidence_root=/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1
fresh_runtime="${evidence_root}/fresh_runtime"
runtime_root="${evidence_root}/resume_runtime"
fresh_name=robotwin_adjust_bottle_rlt_stage2_smoke_fresh_v1
resume_name=robotwin_adjust_bottle_rlt_stage2_smoke_resume_v1
fresh_checkpoint="${smoke_root}/${fresh_name}/checkpoints/global_step_1"
resume_checkpoint="${smoke_root}/${resume_name}/checkpoints/global_step_2"
monitor=/root/autodl-tmp/tmp/rlt_stage2_resource_monitor_20260729.sh

stage1_model=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
stage1_manifest=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
stage1_manifest_id=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
stage1_manifest_sha256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
norm_stats=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
norm_stats_sha256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

code_commit=3b610cb4685a1d41c97da64df67ab86561697dfd
formal_config_sha256=f089f333839c99b87d546e8bcf0d5bddbb7da380e8cc1597e1de4c4450592850
smoke_config_sha256=02715a69ac5ff76fb6d3d7250b447dd0131f3621ae87271d54e9fe4ef6712aa8
worker_sha256=71cccde9b7f18ab63a10817f75b7d5a4d5f5c8d9cadfef99da20690d327c4766
expected_resolved_sha256=f91688d21c7d6180dacb169824210b415e49f1c7d26a27d2a917f562ab24c82a
monitor_sha256=925cb515a4ecd6dbfcb192168c63644e1b2b2d691f6a4d50fdc3ddd8a5bbd96b

cd "${repo}"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test -z "$(git status --porcelain)"
test "$(git rev-list --left-right --count HEAD...@{upstream})" = $'0\t0'
git merge-base --is-ancestor "${code_commit}" HEAD
git diff --quiet \
  "${code_commit}" -- . ':(exclude)docs/**' ':(exclude)HANDOFF.md'
test "$(sha256sum examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml | cut -d' ' -f1)" = "${formal_config_sha256}"
test "$(sha256sum examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke.yaml | cut -d' ' -f1)" = "${smoke_config_sha256}"
test "$(sha256sum rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py | cut -d' ' -f1)" = "${worker_sha256}"
test "$(sha256sum "${stage1_manifest}" | cut -d' ' -f1)" = "${stage1_manifest_sha256}"
test "$(sha256sum "${norm_stats}" | cut -d' ' -f1)" = "${norm_stats_sha256}"
test -f "${fresh_runtime}/exit_code.txt"
test "$(cat "${fresh_runtime}/exit_code.txt")" = 0
test -d "${fresh_checkpoint}/actor/dcp_checkpoint"
test -f "${fresh_checkpoint}/actor/dcp_checkpoint/.metadata"
test -d "${fresh_checkpoint}/actor/sac_components/rlt_trainer_state"
test -d "${fresh_checkpoint}/actor/sac_components/replay_buffer/rank_0"
test -d "${fresh_checkpoint}/actor/sac_components/replay_buffer/rank_1"
test -f "${monitor}"
test "$(sha256sum "${monitor}" | cut -d' ' -f1)" = "${monitor_sha256}"
test ! -e "${smoke_root}/${resume_name}"
test ! -L "${smoke_root}/${resume_name}"
test ! -e "${runtime_root}"
test ! -L "${runtime_root}"

if pgrep -af \
  'train_embodied_agent|ray::|raylet|gcs_server|robotwin_adjust_bottle_rlt_stage2' \
  | grep -v -E 'pgrep -af|start_stage2_smoke_resume' >/dev/null
then
  printf '%s\n' "Refusing resume smoke while an RLT/Ray process is active." >&2
  exit 40
fi
test -z "$(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | awk 'NF'
)"
while IFS=, read -r index used util; do
  used=${used// /}
  util=${util// /}
  test "${used}" -le 16
  test "${util}" -eq 0
done < <(
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
)
host_available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
disk_available=$(df -B1 --output=avail /root/autodl-tmp | tail -n 1)
test "${host_available_kib}" -ge 419430400
test "${disk_available}" -ge 214748364800

mkdir -p "${runtime_root}"
PYTHONPATH="${repo}" PYTHONDONTWRITEBYTECODE=1 \
"${venv}/bin/python" -B - \
  "${fresh_checkpoint}" \
  "${stage1_manifest_sha256}" \
  "${runtime_root}/fresh_checkpoint_preflight.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

checkpoint = Path(sys.argv[1]).resolve(strict=True)
expected_manifest_sha = sys.argv[2]
output = Path(sys.argv[3])
state_dir = checkpoint / "actor" / "sac_components" / "rlt_trainer_state"
completion_path = state_dir / "rlt_trainer_state_complete.json"
completion = json.loads(completion_path.read_text(encoding="utf-8"))

assert completion["complete"] is True
assert completion["schema_version"] == 1
assert completion["actor_world_size"] == 2
assert completion["saved_runner_step"] == 1
assert completion["update_step"] == 8
assert [entry["rank"] for entry in completion["files"]] == [0, 1]

states = []
replay = []
for rank in range(2):
    entry = completion["files"][rank]
    state_path = state_dir / f"checkpoint_rank_{rank}.pt"
    digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
    assert digest == entry["sha256"]
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    assert state["rank"] == rank
    assert state["actor_world_size"] == 2
    assert state["saved_runner_step"] == 1
    assert state["update_step"] == 8
    assert state["local_total_transitions_added"] >= 2
    assert state["global_warmup_ready_total_transitions"] is not None
    contract = json.loads(state["rlt_resume_contract"])
    assert contract["contract"]["stage1_manifest_sha256"] == expected_manifest_sha
    states.append(
        {
            "rank": rank,
            "update_step": state["update_step"],
            "local_total_transitions_added": (
                state["local_total_transitions_added"]
            ),
            "local_total_episodes_added": state["local_total_episodes_added"],
            "warmup_transition_anchor": (
                state["global_warmup_ready_total_transitions"]
            ),
            "contract_sha256": state["rlt_resume_contract_sha256"],
            "state_sha256": digest,
        }
    )

    replay_path = (
        checkpoint
        / "actor"
        / "sac_components"
        / "replay_buffer"
        / f"rank_{rank}"
        / "metadata.json"
    )
    replay_metadata = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay_metadata["total_samples"] == state["local_total_transitions_added"]
    assert 2 <= replay_metadata["total_samples"] <= 64
    replay.append({"rank": rank, **replay_metadata})

assert len({state["update_step"] for state in states}) == 1
assert len({state["contract_sha256"] for state in states}) == 1
payload = {
    "passed": True,
    "checkpoint": str(checkpoint),
    "completion": completion,
    "rank_states": states,
    "replay": replay,
}
output.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

(
  cd "${fresh_checkpoint}"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) >"${runtime_root}/fresh_checkpoint_sha256_before_resume.txt"

export PYTHONPATH="${repo}:${assets}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export EMBODIED_PATH="${repo}/examples/embodiment"
export REPO_PATH="${repo}"
export ROBOTWIN_PATH="${assets}"
export ROBOTWIN_ASSETS_PATH="${assets}"
export ROBOT_PLATFORM=ALOHA
export CUDA_VISIBLE_DEVICES=0,1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RLT_LOG_ROOT="${smoke_root}"
export ROBOTWIN_PI0_NORM_STATS_PATH="${norm_stats}"
export RLT_STAGE1_MODEL_PATH="${stage1_model}"
export RLT_STAGE1_MANIFEST_PATH="${stage1_manifest}"
export RLT_STAGE1_MANIFEST_ID="${stage1_manifest_id}"
export RLT_STAGE1_MANIFEST_SHA256="${stage1_manifest_sha256}"
export RLT_NORM_STATS_SHA256="${norm_stats_sha256}"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY

"${venv}/bin/python" -B \
  examples/embodiment/train_embodied_agent.py \
  --config-path "${repo}/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke \
  "runner.max_steps=2" \
  "runner.resume_dir=${fresh_checkpoint}" \
  "runner.logger.experiment_name=${resume_name}" \
  --cfg job \
  --resolve >"${runtime_root}/resolved.yaml"
test "$(sha256sum "${runtime_root}/resolved.yaml" | cut -d' ' -f1)" = \
  "${expected_resolved_sha256}"

{
  date --iso-8601=seconds
  git status --short --branch
  git rev-parse HEAD
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
  free -b
  cat /sys/fs/cgroup/memory.current
  cat /sys/fs/cgroup/memory.stat
  cat /sys/fs/cgroup/memory.events
  df -B1 /root/autodl-tmp
} >"${runtime_root}/resources_before.txt"

resume_cmd=(
  "${venv}/bin/python" -B
  examples/embodiment/train_embodied_agent.py
  --config-path "${repo}/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke
  "runner.max_steps=2"
  "runner.resume_dir=${fresh_checkpoint}"
  "runner.logger.experiment_name=${resume_name}"
)
printf '%q ' "${resume_cmd[@]}" >"${runtime_root}/exact_command.txt"
printf '\n' >>"${runtime_root}/exact_command.txt"

cat >"${runtime_root}/run_foreground.sh" <<EOF
#!/usr/bin/env bash
set +e
cd "${repo}"
export PYTHONPATH="${repo}:${assets}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export EMBODIED_PATH="${repo}/examples/embodiment"
export REPO_PATH="${repo}"
export ROBOTWIN_PATH="${assets}"
export ROBOTWIN_ASSETS_PATH="${assets}"
export ROBOT_PLATFORM=ALOHA
export CUDA_VISIBLE_DEVICES=0,1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RLT_LOG_ROOT="${smoke_root}"
export ROBOTWIN_PI0_NORM_STATS_PATH="${norm_stats}"
export RLT_STAGE1_MODEL_PATH="${stage1_model}"
export RLT_STAGE1_MANIFEST_PATH="${stage1_manifest}"
export RLT_STAGE1_MANIFEST_ID="${stage1_manifest_id}"
export RLT_STAGE1_MANIFEST_SHA256="${stage1_manifest_sha256}"
export RLT_NORM_STATS_SHA256="${norm_stats_sha256}"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
date --iso-8601=seconds >"${runtime_root}/started_at.txt"
timeout --signal=TERM --kill-after=120s 5400s \\
  "${resume_cmd[@]}"
rc=\$?
printf '%s\n' "\${rc}" >"${runtime_root}/exit_code.txt"
date --iso-8601=seconds >"${runtime_root}/finished_at.txt"
exit "\${rc}"
EOF
chmod 700 "${runtime_root}/run_foreground.sh"

{
  printf 'branch\t%s\n' "$(git branch --show-current)"
  printf 'head\t%s\n' "$(git rev-parse HEAD)"
  printf 'code_commit\t%s\n' "${code_commit}"
  printf 'stage1_manifest_sha256\t%s\n' "${stage1_manifest_sha256}"
  printf 'norm_stats_sha256\t%s\n' "${norm_stats_sha256}"
  printf 'source_config_sha256\t%s\n' "${smoke_config_sha256}"
  printf 'resolved_config_sha256\t%s\n' "${expected_resolved_sha256}"
  printf 'resume_from\t%s\n' "${fresh_checkpoint}"
  printf 'experiment_name\t%s\n' "${resume_name}"
  printf 'checkpoint\t%s\n' "${resume_checkpoint}"
  printf 'timeout_seconds\t5400\n'
} >"${runtime_root}/run_provenance.tsv"

nohup "${runtime_root}/run_foreground.sh" \
  >"${runtime_root}/driver.log" \
  2>&1 \
  </dev/null &
driver_pid=$!
printf '%s\n' "${driver_pid}" >"${runtime_root}/driver_pid.txt"

nohup bash "${monitor}" \
  "${driver_pid}" \
  "${runtime_root}/resources.csv" \
  2 \
  >"${runtime_root}/monitor.log" \
  2>&1 \
  </dev/null &
monitor_pid=$!
printf '%s\n' "${monitor_pid}" >"${runtime_root}/monitor_pid.txt"

sleep 2
kill -0 "${driver_pid}"
printf 'RESUME_DRIVER_PID\t%s\n' "${driver_pid}"
printf 'RESUME_MONITOR_PID\t%s\n' "${monitor_pid}"
printf 'RUNTIME_ROOT\t%s\n' "${runtime_root}"
printf 'RESUME_FROM\t%s\n' "${fresh_checkpoint}"
printf 'EXPECTED_CHECKPOINT\t%s\n' "${resume_checkpoint}"
