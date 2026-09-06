#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
run_root=/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1
evidence_root=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1
runtime_root="${evidence_root}/runtime"
experiment_name=robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1
monitor=/root/autodl-tmp/tmp/rlt_stage2_resource_monitor_20260729.sh
smoke_summary=/root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1/runtime/postflight_summary.json

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

cd "$repo"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git rev-parse --short=8 HEAD)" = 46a2d19b
head="$(git rev-parse HEAD)"
test -z "$(git status --short)"
test "$(sha256sum "$formal_config" | awk '{print $1}')" = "$formal_config_sha256"
test "$(sha256sum "$seed_bank" | awk '{print $1}')" = "$seed_bank_sha256"
test "$(sha256sum "$worker" | awk '{print $1}')" = "$worker_sha256"
test "$(sha256sum "$preflight" | awk '{print $1}')" = "$preflight_sha256"
test "$(sha256sum "$monitor" | awk '{print $1}')" = "$monitor_sha256"
test "$(sha256sum "$stage1_manifest" | awk '{print $1}')" = "$stage1_manifest_sha256"
test "$(sha256sum "$norm_stats" | awk '{print $1}')" = "$norm_stats_sha256"
test ! -e "$run_root"
test ! -L "$run_root"
test ! -e "$evidence_root"
test ! -L "$evidence_root"

SMOKE_SUMMARY="$smoke_summary" \
  "${venv}/bin/python" -B - <<'PY'
import json
import os
from pathlib import Path

summary = json.loads(Path(os.environ["SMOKE_SUMMARY"]).read_text())
assert summary["status"] == "pass"
assert summary["exit_code"] == 0
assert summary["seed_count"] == 20
assert summary["seed_unique_count"] == 20
assert summary["eval_episodes"] == 20
assert summary["checkpoint_complete"] is True
assert summary["gpu0_peak_mib"] < 40960
assert summary["gpu1_peak_mib"] < 40960
assert summary["cgroup_anon_peak_gib"] < 100
assert summary["cgroup_high_delta"] == 0
assert summary["cgroup_oom_delta"] == 0
assert summary["cgroup_oom_kill_delta"] == 0
PY

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
test "$(awk '$1 == "oom" {print $2}' /sys/fs/cgroup/memory.events)" = 0
test "$(awk '$1 == "oom_kill" {print $2}' /sys/fs/cgroup/memory.events)" = 0

mkdir -p "$runtime_root"
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

cp "$formal_config" "$runtime_root/source_config.yaml"
cp "$seed_bank" "$runtime_root/eval_seed_bank.json"
cp "$smoke_summary" "$runtime_root/resource_gate_summary.json"
"${venv}/bin/python" -B \
  examples/embodiment/train_embodied_agent.py \
  --config-path "${repo}/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250 \
  "runner.logger.log_path=${run_root}" \
  "runner.logger.experiment_name=${experiment_name}" \
  "runner.resume_dir=null" \
  --cfg job \
  --resolve >"$runtime_root/resolved.yaml"
resolved_sha256="$(sha256sum "$runtime_root/resolved.yaml" | awk '{print $1}')"
smoke_summary_sha256="$(sha256sum "$smoke_summary" | awk '{print $1}')"

RESOLVED="$runtime_root/resolved.yaml" \
RUN_ROOT="$run_root" \
EXPERIMENT_NAME="$experiment_name" \
STAGE1_MODEL="$stage1_model" \
STAGE1_MANIFEST="$stage1_manifest" \
SEED_BANK="$repo/$seed_bank" \
  "${venv}/bin/python" -B - <<'PY'
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

cfg = OmegaConf.load(os.environ["RESOLVED"])
assert cfg.runner.max_steps == 250
assert cfg.runner.val_check_interval == 25
assert cfg.runner.save_interval == 25
assert cfg.runner.resume_dir is None
assert cfg.runner.logger.log_path == os.environ["RUN_ROOT"]
assert cfg.runner.logger.experiment_name == os.environ["EXPERIMENT_NAME"]
assert cfg.runner.weight_sync_interval == 1
assert cfg.env.train.total_num_envs == 8
assert cfg.env.train.rollout_epoch == 1
assert cfg.env.train.auto_reset is False
assert cfg.env.train.max_episode_steps == 200
assert cfg.env.train.max_steps_per_rollout_epoch == 200
assert cfg.env.eval.total_num_envs == 4
assert cfg.env.eval.rollout_epoch == 5
assert cfg.env.eval.auto_reset is False
assert cfg.env.eval.ignore_terminations is True
assert cfg.env.eval.group_size == 1
assert cfg.env.eval.use_fixed_reset_state_ids is False
assert cfg.env.eval.seeds_path == os.environ["SEED_BANK"]
seeds = json.loads(Path(os.environ["SEED_BANK"]).read_text())[
    "adjust_bottle"
]["success_seeds"]
assert len(seeds) == len(set(seeds)) == 20
assert cfg.algorithm.update_epoch == 5
assert cfg.algorithm.rlt_schedule.max_updates_per_train_step == 1600
assert cfg.algorithm.rlt_schedule.warmup_min_size == 10000
assert cfg.algorithm.rlt_schedule.warmup_post_collect_updates == 30000
assert cfg.algorithm.rlt_schedule.train_every_transitions == 1
assert cfg.algorithm.critic_actor_ratio == 2
assert cfg.algorithm.actor_weight_schedule.warmup_updates == 20000
assert cfg.algorithm.actor_weight_schedule.ramp_updates == 50000
assert cfg.algorithm.reference_dropout_prob == 0.5
assert cfg.algorithm.replay_buffer.cache_size == 50000
assert cfg.algorithm.replay_buffer.sample_window_size == 50000
assert cfg.actor.micro_batch_size == 128
assert cfg.actor.global_batch_size == 512
assert cfg.actor.model.z_dim == 2048
assert cfg.actor.model.proprio_dim == 14
assert cfg.actor.model.action_dim == 14
assert cfg.actor.model.num_action_chunks == 10
assert cfg.actor.model.fixed_std == 0.002
assert cfg.actor.optim.lr == 1e-4
assert cfg.actor.critic_optim.lr == 1e-4
assert cfg.actor.optim.clip_grad == 10
assert cfg.actor.critic_optim.clip_grad == 10
assert cfg.rollout.rlt_feature_model.openpi.action_horizon == 50
assert cfg.rollout.rlt_feature_model.openpi.action_chunk == 10
assert cfg.rollout.rlt_feature_model.model_path == os.environ["STAGE1_MODEL"]
assert cfg.algorithm.rlt_route.type == "full_task"
assert cfg.weight_syncer.patch.init_sync.enabled is True
assert cfg.algorithm.rlt_resume.contract.stage1_manifest_path == os.environ["STAGE1_MANIFEST"]
assert "UNRESOLVED" not in Path(os.environ["RESOLVED"]).read_text()
PY

formal_cmd=(
  "${venv}/bin/python" -B
  examples/embodiment/train_embodied_agent.py
  --config-path "${repo}/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250
  "runner.logger.log_path=${run_root}"
  "runner.logger.experiment_name=${experiment_name}"
  "runner.resume_dir=null"
)
printf '%q ' "${formal_cmd[@]}" >"$runtime_root/exact_command.txt"
printf '\n' >>"$runtime_root/exact_command.txt"

cat >"$runtime_root/budget.json" <<'EOF'
{
  "actor_updates_expected": 62762,
  "checkpoints": 10,
  "critic_updates_expected": 125525,
  "eval_episodes_periodic": 200,
  "eval_events": 10,
  "expected_macro_transitions": 39105,
  "hard_timeout_seconds": 64800,
  "max_action_slots": 400000,
  "max_macro_transitions": 40000,
  "outer_cycles": 250,
  "train_envs": 8,
  "train_episodes": 2000
}
EOF

{
  printf 'prepared_at\t%s\n' "$(date --iso-8601=seconds)"
  printf 'branch\t%s\n' "$(git branch --show-current)"
  printf 'head\t%s\n' "$head"
  printf 'ahead_behind_head_vs_upstream\t%s\n' "$(
    git rev-list --left-right --count HEAD...@{upstream}
  )"
  printf 'push_status\tdeferred_after_github_main_timeout\n'
  printf 'formal_config_sha256\t%s\n' "$formal_config_sha256"
  printf 'seed_bank_sha256\t%s\n' "$seed_bank_sha256"
  printf 'worker_sha256\t%s\n' "$worker_sha256"
  printf 'preflight_sha256\t%s\n' "$preflight_sha256"
  printf 'monitor_sha256\t%s\n' "$monitor_sha256"
  printf 'smoke_summary_sha256\t%s\n' "$smoke_summary_sha256"
  printf 'stage1_model\t%s\n' "$stage1_model"
  printf 'stage1_manifest\t%s\n' "$stage1_manifest"
  printf 'stage1_manifest_sha256\t%s\n' "$stage1_manifest_sha256"
  printf 'norm_stats\t%s\n' "$norm_stats"
  printf 'norm_stats_sha256\t%s\n' "$norm_stats_sha256"
  printf 'resolved_config_sha256\t%s\n' "$resolved_sha256"
  printf 'run_root\t%s\n' "$run_root"
  printf 'experiment_name\t%s\n' "$experiment_name"
  printf 'runtime_root\t%s\n' "$runtime_root"
  printf 'max_steps\t250\n'
  printf 'timeout_seconds\t64800\n'
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
timeout --signal=TERM --kill-after=180s 64800s \\
  "${venv}/bin/python" -B \\
  examples/embodiment/train_embodied_agent.py \\
  --config-path "${repo}/examples/embodiment/config" \\
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250 \\
  "runner.logger.log_path=${run_root}" \\
  "runner.logger.experiment_name=${experiment_name}" \\
  "runner.resume_dir=null"
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
test -z "\$(git status --short)"
test "\$(sha256sum "$formal_config" | awk '{print \$1}')" = "$formal_config_sha256"
test "\$(sha256sum "$seed_bank" | awk '{print \$1}')" = "$seed_bank_sha256"
test "\$(sha256sum "$preflight" | awk '{print \$1}')" = "$preflight_sha256"
test "\$(sha256sum "$monitor" | awk '{print \$1}')" = "$monitor_sha256"
test "\$(sha256sum "$runtime_root/resolved.yaml" | awk '{print \$1}')" = "$resolved_sha256"
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
printf 'FORMAL_DRIVER_PID\t%s\n' "\$driver_pid"
printf 'FORMAL_MONITOR_PID\t%s\n' "\$monitor_pid"
printf 'RUNTIME_ROOT\t%s\n' "$runtime_root"
printf 'RUN_ROOT\t%s\n' "$run_root"
printf 'EXPERIMENT_NAME\t%s\n' "$experiment_name"
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
progress for about 30 minutes, or the 18-hour hard timeout.
EOF

bash -n "$runtime_root/run_foreground.sh"
bash -n "$runtime_root/launch_background.sh"
grep -A8 '^timeout --signal=TERM' "$runtime_root/run_foreground.sh"

printf 'PREPARED_AT\t%s\n' "$(date --iso-8601=seconds)"
printf 'HEAD\t%s\n' "$head"
printf 'RUNTIME_ROOT\t%s\n' "$runtime_root"
printf 'RUN_ROOT\t%s\n' "$run_root"
printf 'EXPERIMENT_NAME\t%s\n' "$experiment_name"
printf 'RESOLVED_SHA256\t%s\n' "$resolved_sha256"
printf 'SMOKE_SUMMARY_SHA256\t%s\n' "$smoke_summary_sha256"
printf 'EXACT_COMMAND_FILE\t%s\n' "$runtime_root/exact_command.txt"
printf 'LAUNCH_COMMAND\tbash %s\n' "$runtime_root/launch_background.sh"
printf 'TIMEOUT_SECONDS\t64800\n'
printf '%s\n' RLT_STAGE2_FORMAL_8ENV250_PREPARED
