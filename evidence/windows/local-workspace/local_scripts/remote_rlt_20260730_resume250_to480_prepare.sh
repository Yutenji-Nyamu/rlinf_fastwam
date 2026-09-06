#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
source_run=/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1
source_evidence=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1
source_experiment=robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1
source_runtime="${source_evidence}/runtime"
source_checkpoint="${source_run}/${source_experiment}/checkpoints/global_step_250"
source_completion="${source_checkpoint}/actor/sac_components/rlt_trainer_state/rlt_trainer_state_complete.json"

run_root=/root/autodl-tmp/experiments/rlt_stage2_formal_resume250_to480_20260730_v1
evidence_root=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_resume250_to480_20260730_v1
runtime_root="${evidence_root}/runtime"
experiment_name=robotwin_adjust_bottle_rlt_stage2_formal_resume250_to480_v1
monitor=/root/autodl-tmp/tmp/rlt_stage2_resource_monitor_20260729.sh

stage1_model=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
stage1_manifest=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
stage1_manifest_id=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
stage1_manifest_sha256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
norm_stats=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
norm_stats_sha256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

formal_config=examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250.yaml
formal_config_sha256=d9ee30f8c776b349cc3e5e08cb17c97f46151c9c62bffada7f514e9522ffb315
seed_bank=rlinf/envs/robotwin/seeds/eval_seeds_adjust_bottle_rlt_periodic20_v1.json
seed_bank_sha256=fb9c3353e27b83aad6fe7ff778437d960b084de9d981c2af68615d52769952a7
worker=rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py
worker_sha256=71cccde9b7f18ab63a10817f75b7d5a4d5f5c8d9cadfef99da20690d327c4766
preflight=toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py
preflight_sha256=3278a8cbdf766d30309856eac2a4eb5f8cc3c792986e230c2ef022b615553bb6
monitor_sha256=925cb515a4ecd6dbfcb192168c63644e1b2b2d691f6a4d50fdc3ddd8a5bbd96b
source_resolved_sha256=586644cd69461016c1dd8c653da0eea12b01c61f2d0a9b4901654d90800f2a3e
source_completion_sha256=5d4185b3782aa227c22be257328302fbe6a05f7bc0aede54114f1303b444ae15
head=46a2d19bae629eaa57830f5faeac71ac81a1a494
hard_timeout_seconds=43200

cd "$repo"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git rev-parse HEAD)" = "$head"
test -z "$(git status --porcelain)"
test "$(sha256sum "$formal_config" | awk '{print $1}')" = "$formal_config_sha256"
test "$(sha256sum "$seed_bank" | awk '{print $1}')" = "$seed_bank_sha256"
test "$(sha256sum "$worker" | awk '{print $1}')" = "$worker_sha256"
test "$(sha256sum "$preflight" | awk '{print $1}')" = "$preflight_sha256"
test "$(sha256sum "$monitor" | awk '{print $1}')" = "$monitor_sha256"
test "$(sha256sum "$stage1_manifest" | awk '{print $1}')" = "$stage1_manifest_sha256"
test "$(sha256sum "$norm_stats" | awk '{print $1}')" = "$norm_stats_sha256"
test "$(sha256sum "${source_runtime}/resolved.yaml" | awk '{print $1}')" = \
  "$source_resolved_sha256"
test "$(sha256sum "$source_completion" | awk '{print $1}')" = \
  "$source_completion_sha256"
test "$(cat "${source_runtime}/exit_code.txt")" = 0
test -f "${source_runtime}/finished_at.txt"
test -d "$source_checkpoint"
test ! -L "$source_checkpoint"
test ! -e "$run_root"
test ! -L "$run_root"
if test -e "$evidence_root"; then
  test -d "$runtime_root"
  test ! -L "$evidence_root"
  test ! -L "$runtime_root"
  test ! -e "$runtime_root/driver.log"
  test ! -e "$runtime_root/driver_pid.txt"
  test ! -e "$runtime_root/started_at.txt"
  test ! -e "$runtime_root/exit_code.txt"
  test ! -e "$runtime_root/resources.csv"
  mapfile -t existing_runtime_files < <(
    find "$runtime_root" -maxdepth 1 -type f -printf '%f\n' | sort
  )
  for existing_file in "${existing_runtime_files[@]}"; do
    case "$existing_file" in
      budget.json|config_parity.json|eval_seed_bank.json|exact_command.txt|\
      launch_background.sh|resolved.yaml|resources_before.txt|\
      run_foreground.sh|run_provenance.tsv|source_checkpoint_preflight.json|\
      source_completion.json|source_config.yaml|source_resolved.yaml|\
      stage1_binding_preflight.json|stage1_binding_preflight.stdout|\
      stop_command.txt|stop_conditions.txt)
        ;;
      *)
        printf 'Unexpected pre-launch runtime file: %s\n' "$existing_file" >&2
        exit 41
        ;;
    esac
  done
else
  test ! -L "$evidence_root"
fi

mapfile -t process_rows < <(
  {
    ps -eo pid=,comm=,args= \
      | awk '$2 ~ /^python/ && $0 ~ /train_embodied_agent[.]py/ {print}'
    pgrep -ax raylet || true
    pgrep -ax gcs_server || true
  }
)
test "${#process_rows[@]}" = 0
mapfile -t compute_rows < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk 'NF'
)
test "${#compute_rows[@]}" = 0
test "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -ge 419430400
test "$(
  df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' '
)" -ge 214748364800
test "$(awk '$1 == "anon" {print $2}' /sys/fs/cgroup/memory.stat)" -le 10737418240
test "$(awk '$1 == "oom" {print $2}' /sys/fs/cgroup/memory.events)" = 0
test "$(awk '$1 == "oom_kill" {print $2}' /sys/fs/cgroup/memory.events)" = 0

mkdir -p "$runtime_root"

PYTHONPATH="$repo" PYTHONDONTWRITEBYTECODE=1 \
  "${venv}/bin/python" -B - \
  "$source_checkpoint" \
  "$stage1_manifest_sha256" \
  "$runtime_root/source_checkpoint_preflight.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

checkpoint = Path(sys.argv[1]).resolve(strict=True)
expected_stage1_manifest_sha = sys.argv[2]
output = Path(sys.argv[3])
state_dir = checkpoint / "actor" / "sac_components" / "rlt_trainer_state"
completion_path = state_dir / "rlt_trainer_state_complete.json"
completion = json.loads(completion_path.read_text(encoding="utf-8"))

assert completion["complete"] is True
assert completion["schema_version"] == 1
assert completion["actor_world_size"] == 2
assert completion["saved_runner_step"] == 250
assert completion["update_step"] == 102260
assert completion["rlt_resume_contract_sha256"] == (
    "82cd409bef1549afb3feb41fa5a80ed08d207d112e4dfe8020af14f49cad1fc9"
)

entries = {int(entry["rank"]): entry for entry in completion["files"]}
assert sorted(entries) == [0, 1]
states = []
replays = []
for rank in range(2):
    entry = entries[rank]
    state_path = state_dir / f"checkpoint_rank_{rank}.pt"
    digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
    assert digest == entry["sha256"]
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    assert state["rank"] == rank
    assert state["actor_world_size"] == 2
    assert state["saved_runner_step"] == 250
    assert state["update_step"] == 102260
    assert state["global_warmup_ready_total_transitions"] == 20399
    assert state["global_warmup_ready_total_episodes"] == 1088
    assert state["rlt_resume_contract_sha256"] == (
        completion["rlt_resume_contract_sha256"]
    )
    assert hashlib.sha256(
        state["rlt_resume_contract"].encode("utf-8")
    ).hexdigest() == state["rlt_resume_contract_sha256"]
    contract = json.loads(state["rlt_resume_contract"])
    assert contract["contract"]["stage1_manifest_sha256"] == (
        expected_stage1_manifest_sha
    )

    replay_path = (
        checkpoint
        / "actor"
        / "sac_components"
        / "replay_buffer"
        / f"rank_{rank}"
        / "metadata.json"
    )
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay["total_samples"] == state["local_total_transitions_added"]
    assert replay["size"] == replay["total_samples"]
    assert replay["trajectory_counter"] == replay["total_samples"]
    states.append(
        {
            "rank": rank,
            "state_sha256": digest,
            "update_step": int(state["update_step"]),
            "local_total_transitions_added": int(
                state["local_total_transitions_added"]
            ),
            "local_total_episodes_added": int(state["local_total_episodes_added"]),
            "warmup_transitions": int(
                state["global_warmup_ready_total_transitions"]
            ),
            "warmup_episodes": int(state["global_warmup_ready_total_episodes"]),
            "contract_sha256": state["rlt_resume_contract_sha256"],
        }
    )
    replays.append({"rank": rank, **replay})

assert len({state["update_step"] for state in states}) == 1
assert len({state["contract_sha256"] for state in states}) == 1
assert [item["total_samples"] for item in replays] == [17170, 17681]

payload = {
    "passed": True,
    "checkpoint": str(checkpoint),
    "completion": completion,
    "rank_states": states,
    "replay": replays,
}
output.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

export PYTHONPATH="${repo}:${assets}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export EMBODIED_PATH="${repo}/examples/embodiment"
export REPO_PATH="$repo"
export ROBOTWIN_PATH="$assets"
export ROBOTWIN_ASSETS_PATH="$assets"
export ROBOT_PLATFORM=ALOHA
export CUDA_VISIBLE_DEVICES=0,1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RLT_LOG_ROOT="$run_root"
export ROBOTWIN_PI0_NORM_STATS_PATH="$norm_stats"
export RLT_STAGE1_MODEL_PATH="$stage1_model"
export RLT_STAGE1_MANIFEST_PATH="$stage1_manifest"
export RLT_STAGE1_MANIFEST_ID="$stage1_manifest_id"
export RLT_STAGE1_MANIFEST_SHA256="$stage1_manifest_sha256"
export RLT_NORM_STATS_SHA256="$norm_stats_sha256"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY

if test -e "$runtime_root/stage1_binding_preflight.json"; then
  test -f "$runtime_root/stage1_binding_preflight.stdout"
  "${venv}/bin/python" -B - \
    "$runtime_root/stage1_binding_preflight.json" \
    "$stage1_manifest" \
    "$stage1_manifest_sha256" \
    "$stage1_model" \
    "$norm_stats" \
    "$norm_stats_sha256" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["passed"] is True
assert payload["manifest"]["id"] == (
    "robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1"
)
assert payload["manifest"]["path"] == sys.argv[2]
assert payload["manifest"]["sha256"] == sys.argv[3]
assert payload["stage1_model"]["path"] == sys.argv[4]
assert payload["norm_stats"]["path"] == sys.argv[5]
assert payload["norm_stats"]["sha256"] == sys.argv[6]
assert payload["model_contract"] == {
    "action_chunk": 10,
    "action_dim": 14,
    "action_horizon": 50,
    "canonical_adapter_version": "robotwin_aloha_canonical_v1",
    "image_prefix_shape": [768, 2048],
    "norm_stats_sha256": sys.argv[6],
    "z_rl_dim": 2048,
}
print("REUSED_STAGE1_BINDING_PREFLIGHT_PASS")
PY
else
  "${venv}/bin/python" -B "$preflight" \
    --manifest-path "$stage1_manifest" \
    --manifest-id "$stage1_manifest_id" \
    --manifest-sha256 "$stage1_manifest_sha256" \
    --stage1-model-path "$stage1_model" \
    --norm-stats-path "$norm_stats" \
    --norm-stats-sha256 "$norm_stats_sha256" \
    --canonical-adapter-version robotwin_aloha_canonical_v1 \
    --action-horizon 50 \
    --action-chunk 10 \
    --action-dim 14 \
    --z-rl-dim 2048 \
    --prefix-seq-len 768 \
    --prefix-dim 2048 \
    --output "$runtime_root/stage1_binding_preflight.json" \
    >"$runtime_root/stage1_binding_preflight.stdout"
fi

cp "$formal_config" "$runtime_root/source_config.yaml"
cp "$seed_bank" "$runtime_root/eval_seed_bank.json"
cp "${source_runtime}/resolved.yaml" "$runtime_root/source_resolved.yaml"
cp "$source_completion" "$runtime_root/source_completion.json"

"${venv}/bin/python" -B \
  examples/embodiment/train_embodied_agent.py \
  --config-path "${repo}/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250 \
  "runner.logger.log_path=${run_root}" \
  "runner.logger.experiment_name=${experiment_name}" \
  "runner.max_steps=480" \
  "runner.resume_dir=${source_checkpoint}" \
  --cfg job \
  --resolve >"$runtime_root/resolved.yaml"
resolved_sha256="$(sha256sum "$runtime_root/resolved.yaml" | awk '{print $1}')"

SOURCE_RESOLVED="$runtime_root/source_resolved.yaml" \
RESOLVED="$runtime_root/resolved.yaml" \
RUN_ROOT="$run_root" \
EXPERIMENT_NAME="$experiment_name" \
SOURCE_CHECKPOINT="$source_checkpoint" \
PARITY_OUTPUT="$runtime_root/config_parity.json" \
  "${venv}/bin/python" -B - <<'PY'
import json
import os
from pathlib import Path

from omegaconf import OmegaConf


def load(path):
    return OmegaConf.to_container(
        OmegaConf.load(path),
        resolve=True,
        enum_to_str=True,
    )


def walk_diff(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        paths = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(walk_diff(left[key], right[key], path))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        paths = []
        if len(left) != len(right):
            return [prefix]
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.extend(
                walk_diff(left_item, right_item, f"{prefix}[{index}]")
            )
        return paths
    return [] if left == right else [prefix]


source = load(os.environ["SOURCE_RESOLVED"])
resume = load(os.environ["RESOLVED"])
changed = walk_diff(source, resume)
allowed = {
    "env.eval.video_cfg.video_base_dir",
    "env.train.video_cfg.video_base_dir",
    "runner.logger.experiment_name",
    "runner.logger.log_path",
    "runner.max_steps",
    "runner.resume_dir",
}
assert set(changed) == allowed, (changed, allowed)
assert source["runner"]["max_steps"] == 250
assert source["runner"]["resume_dir"] is None
assert resume["runner"]["max_steps"] == 480
assert resume["runner"]["resume_dir"] == os.environ["SOURCE_CHECKPOINT"]
assert resume["runner"]["logger"]["log_path"] == os.environ["RUN_ROOT"]
assert resume["runner"]["logger"]["experiment_name"] == (
    os.environ["EXPERIMENT_NAME"]
)
assert os.environ["RUN_ROOT"] in resume["env"]["train"]["video_cfg"][
    "video_base_dir"
]
assert os.environ["RUN_ROOT"] in resume["env"]["eval"]["video_cfg"][
    "video_base_dir"
]
assert resume["runner"]["val_check_interval"] == 25
assert resume["runner"]["save_interval"] == 25
assert resume["runner"]["weight_sync_interval"] == 1
assert resume["env"]["train"]["total_num_envs"] == 8
assert resume["env"]["eval"]["total_num_envs"] == 4
assert resume["env"]["eval"]["rollout_epoch"] == 5
assert resume["algorithm"]["update_epoch"] == 5
assert resume["algorithm"]["critic_actor_ratio"] == 2
assert resume["algorithm"]["rlt_schedule"]["max_updates_per_train_step"] == 1600
assert resume["algorithm"]["rlt_schedule"]["warmup_min_size"] == 10000
assert resume["algorithm"]["rlt_schedule"]["warmup_post_collect_updates"] == 30000
assert resume["algorithm"]["rlt_schedule"]["train_every_transitions"] == 1
assert resume["algorithm"]["actor_weight_schedule"]["warmup_updates"] == 20000
assert resume["algorithm"]["actor_weight_schedule"]["ramp_updates"] == 50000
assert resume["algorithm"]["replay_buffer"]["cache_size"] == 50000
assert resume["algorithm"]["replay_buffer"]["sample_window_size"] == 50000
assert resume["actor"]["micro_batch_size"] == 128
assert resume["actor"]["global_batch_size"] == 512
assert resume["actor"]["model"]["num_action_chunks"] == 10
assert resume["rollout"]["rlt_feature_model"]["openpi"]["action_horizon"] == 50
assert resume["rollout"]["rlt_feature_model"]["openpi"]["action_chunk"] == 10
assert resume["algorithm"]["rlt_route"]["type"] == "full_task"
assert resume["weight_syncer"]["patch"]["init_sync"]["enabled"] is True
assert "UNRESOLVED" not in Path(os.environ["RESOLVED"]).read_text()

payload = {
    "passed": True,
    "changed_paths": changed,
    "source_max_steps": source["runner"]["max_steps"],
    "resume_max_steps": resume["runner"]["max_steps"],
    "resume_dir": resume["runner"]["resume_dir"],
}
Path(os.environ["PARITY_OUTPUT"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

formal_cmd=(
  "${venv}/bin/python" -B
  examples/embodiment/train_embodied_agent.py
  --config-path "${repo}/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250
  "runner.logger.log_path=${run_root}"
  "runner.logger.experiment_name=${experiment_name}"
  "runner.max_steps=480"
  "runner.resume_dir=${source_checkpoint}"
)
printf '%q ' "${formal_cmd[@]}" >"$runtime_root/exact_command.txt"
printf '\n' >>"$runtime_root/exact_command.txt"

cat >"$runtime_root/budget.json" <<'EOF'
{
  "actor_updates_expected_increment": 62675,
  "checkpoints_increment": 10,
  "critic_updates_expected_increment": 125350,
  "eval_episodes_increment": 200,
  "eval_events_increment": 10,
  "expected_macro_transitions_increment": 25070,
  "hard_timeout_seconds": 43200,
  "max_action_slots_increment": 368000,
  "outer_cycles_increment": 230,
  "source_actor_updates": 51130,
  "source_critic_updates": 102260,
  "source_global_step": 250,
  "target_global_step": 480,
  "train_envs": 8,
  "train_episodes_increment": 1840
}
EOF

{
  printf 'prepared_at\t%s\n' "$(date --iso-8601=seconds)"
  printf 'branch\t%s\n' "$(git branch --show-current)"
  printf 'head\t%s\n' "$head"
  printf 'ahead_behind_head_vs_upstream\t%s\n' "$(
    git rev-list --left-right --count HEAD...@{upstream}
  )"
  printf 'formal_config_sha256\t%s\n' "$formal_config_sha256"
  printf 'seed_bank_sha256\t%s\n' "$seed_bank_sha256"
  printf 'worker_sha256\t%s\n' "$worker_sha256"
  printf 'preflight_sha256\t%s\n' "$preflight_sha256"
  printf 'monitor_sha256\t%s\n' "$monitor_sha256"
  printf 'stage1_model\t%s\n' "$stage1_model"
  printf 'stage1_manifest\t%s\n' "$stage1_manifest"
  printf 'stage1_manifest_sha256\t%s\n' "$stage1_manifest_sha256"
  printf 'norm_stats\t%s\n' "$norm_stats"
  printf 'norm_stats_sha256\t%s\n' "$norm_stats_sha256"
  printf 'source_run\t%s\n' "$source_run"
  printf 'source_resolved_sha256\t%s\n' "$source_resolved_sha256"
  printf 'source_checkpoint\t%s\n' "$source_checkpoint"
  printf 'source_completion_sha256\t%s\n' "$source_completion_sha256"
  printf 'source_global_step\t250\n'
  printf 'source_update_step\t102260\n'
  printf 'resolved_config_sha256\t%s\n' "$resolved_sha256"
  printf 'run_root\t%s\n' "$run_root"
  printf 'experiment_name\t%s\n' "$experiment_name"
  printf 'runtime_root\t%s\n' "$runtime_root"
  printf 'target_global_step\t480\n'
  printf 'increment_cycles\t230\n'
  printf 'timeout_seconds\t%s\n' "$hard_timeout_seconds"
} >"$runtime_root/run_provenance.tsv"

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
  cat /proc/pressure/memory
  df -B1 /root/autodl-tmp
} >"$runtime_root/resources_before.txt"

cat >"$runtime_root/run_foreground.sh" <<EOF
#!/usr/bin/env bash
set +e
cd "$repo"
export PYTHONPATH="$repo:$assets"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export EMBODIED_PATH="$repo/examples/embodiment"
export REPO_PATH="$repo"
export ROBOTWIN_PATH="$assets"
export ROBOTWIN_ASSETS_PATH="$assets"
export ROBOT_PLATFORM=ALOHA
export CUDA_VISIBLE_DEVICES=0,1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RLT_LOG_ROOT="$run_root"
export ROBOTWIN_PI0_NORM_STATS_PATH="$norm_stats"
export RLT_STAGE1_MODEL_PATH="$stage1_model"
export RLT_STAGE1_MANIFEST_PATH="$stage1_manifest"
export RLT_STAGE1_MANIFEST_ID="$stage1_manifest_id"
export RLT_STAGE1_MANIFEST_SHA256="$stage1_manifest_sha256"
export RLT_NORM_STATS_SHA256="$norm_stats_sha256"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
date --iso-8601=seconds >"$runtime_root/started_at.txt"
timeout --signal=TERM --kill-after=180s ${hard_timeout_seconds}s \\
  "${venv}/bin/python" -B \\
  examples/embodiment/train_embodied_agent.py \\
  --config-path "${repo}/examples/embodiment/config" \\
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250 \\
  "runner.logger.log_path=${run_root}" \\
  "runner.logger.experiment_name=${experiment_name}" \\
  "runner.max_steps=480" \\
  "runner.resume_dir=${source_checkpoint}"
rc=\$?
printf '%s\n' "\$rc" >"$runtime_root/exit_code.txt"
date --iso-8601=seconds >"$runtime_root/finished_at.txt"
{
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \\
    --format=csv,noheader,nounits
  cat /sys/fs/cgroup/memory.current
  cat /sys/fs/cgroup/memory.events
  cat /proc/pressure/memory
} >"$runtime_root/resources_after.txt"
exit "\$rc"
EOF
chmod 700 "$runtime_root/run_foreground.sh"

cat >"$runtime_root/launch_background.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
cd "$repo"
test "\$(git branch --show-current)" = codex/rlt-pi0-robotwin
test "\$(git rev-parse HEAD)" = "$head"
test -z "\$(git status --porcelain)"
test "\$(sha256sum "$formal_config" | awk '{print \$1}')" = "$formal_config_sha256"
test "\$(sha256sum "$seed_bank" | awk '{print \$1}')" = "$seed_bank_sha256"
test "\$(sha256sum "$preflight" | awk '{print \$1}')" = "$preflight_sha256"
test "\$(sha256sum "$monitor" | awk '{print \$1}')" = "$monitor_sha256"
test "\$(sha256sum "$runtime_root/resolved.yaml" | awk '{print \$1}')" = "$resolved_sha256"
test "\$(sha256sum "$source_completion" | awk '{print \$1}')" = "$source_completion_sha256"
test "\$(cat "$source_runtime/exit_code.txt")" = 0
test -d "$source_checkpoint"
test ! -e "$run_root"
mapfile -t active_rows < <(
  {
    ps -eo pid=,comm=,args= \\
      | awk '\$2 ~ /^python/ && \$0 ~ /train_embodied_agent[.]py/ {print}'
    pgrep -ax raylet || true
    pgrep -ax gcs_server || true
  }
)
test "\${#active_rows[@]}" = 0
mapfile -t compute_rows < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk 'NF'
)
test "\${#compute_rows[@]}" = 0
test "\$(awk '/MemAvailable:/ {print \$2}' /proc/meminfo)" -ge 419430400
test "\$(
  df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' '
)" -ge 214748364800
test "\$(awk '\$1 == "anon" {print \$2}' /sys/fs/cgroup/memory.stat)" -le 10737418240
test "\$(awk '\$1 == "oom" {print \$2}' /sys/fs/cgroup/memory.events)" = 0
test "\$(awk '\$1 == "oom_kill" {print \$2}' /sys/fs/cgroup/memory.events)" = 0
nohup "$runtime_root/run_foreground.sh" \\
  >"$runtime_root/driver.log" \\
  2>&1 \\
  </dev/null &
driver_pid=\$!
printf '%s\n' "\$driver_pid" >"$runtime_root/driver_pid.txt"
nohup bash "$monitor" \\
  "\$driver_pid" \\
  "$runtime_root/resources.csv" \\
  2 \\
  >"$runtime_root/monitor.log" \\
  2>&1 \\
  </dev/null &
monitor_pid=\$!
printf '%s\n' "\$monitor_pid" >"$runtime_root/monitor_pid.txt"
sleep 2
kill -0 "\$driver_pid"
printf 'RESUME_DRIVER_PID\t%s\n' "\$driver_pid"
printf 'RESUME_MONITOR_PID\t%s\n' "\$monitor_pid"
printf 'RUNTIME_ROOT\t%s\n' "$runtime_root"
printf 'RUN_ROOT\t%s\n' "$run_root"
printf 'EXPERIMENT_NAME\t%s\n' "$experiment_name"
printf 'SOURCE_CHECKPOINT\t%s\n' "$source_checkpoint"
printf 'TARGET_GLOBAL_STEP\t480\n'
printf 'RESOLVED_SHA256\t%s\n' "$resolved_sha256"
EOF
chmod 700 "$runtime_root/launch_background.sh"

cat >"$runtime_root/stop_command.txt" <<EOF
kill -TERM \$(cat "$runtime_root/driver_pid.txt")
EOF

cat >"$runtime_root/stop_conditions.txt" <<'EOF'
Do not stop on noisy intermediate success rate.
Stop only for CUDA OOM, NaN/Inf, NCCL fatal, Ray rank death, cgroup OOM,
sustained memory pressure/anon growth, disk below 200 GiB, no effective
progress for about 30 minutes, or the 12-hour hard timeout.
EOF

bash -n "$runtime_root/run_foreground.sh"
bash -n "$runtime_root/launch_background.sh"
grep -A10 '^timeout --signal=TERM' "$runtime_root/run_foreground.sh"

printf 'PREPARED_AT\t%s\n' "$(date --iso-8601=seconds)"
printf 'HEAD\t%s\n' "$head"
printf 'RUNTIME_ROOT\t%s\n' "$runtime_root"
printf 'RUN_ROOT\t%s\n' "$run_root"
printf 'EXPERIMENT_NAME\t%s\n' "$experiment_name"
printf 'SOURCE_CHECKPOINT\t%s\n' "$source_checkpoint"
printf 'SOURCE_GLOBAL_STEP\t250\n'
printf 'TARGET_GLOBAL_STEP\t480\n'
printf 'INCREMENT_CYCLES\t230\n'
printf 'RESOLVED_SHA256\t%s\n' "$resolved_sha256"
printf 'EXACT_COMMAND_FILE\t%s\n' "$runtime_root/exact_command.txt"
printf 'LAUNCH_COMMAND\tbash %s\n' "$runtime_root/launch_background.sh"
printf 'TIMEOUT_SECONDS\t%s\n' "$hard_timeout_seconds"
printf '%s\n' RLT_STAGE2_RESUME250_TO480_PREPARED
