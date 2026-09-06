#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
run_root=/root/autodl-tmp/experiments/rlt_stage2_formal_100c_20260730_v1
evidence_root=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1
runtime_root="${evidence_root}/runtime"
experiment_name=robotwin_adjust_bottle_rlt_stage2_formal_100c_v1
monitor=/root/autodl-tmp/tmp/rlt_stage2_resource_monitor_20260729.sh

stage1_model=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
stage1_manifest=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
stage1_manifest_id=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
stage1_manifest_sha256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
norm_stats=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
norm_stats_sha256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

code_commit=3b610cb4685a1d41c97da64df67ab86561697dfd
formal_config=examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml
formal_config_sha256=f089f333839c99b87d546e8bcf0d5bddbb7da380e8cc1597e1de4c4450592850
worker=rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py
worker_sha256=71cccde9b7f18ab63a10817f75b7d5a4d5f5c8d9cadfef99da20690d327c4766
preflight=toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py
preflight_sha256=3278a8cbdf766d30309856eac2a4eb5f8cc3c792986e230c2ef022b615553bb6
monitor_sha256=925cb515a4ecd6dbfcb192168c63644e1b2b2d691f6a4d50fdc3ddd8a5bbd96b

cd "${repo}"
test "$(git branch --show-current)" = codex/rlt-pi0-robotwin
test -z "$(git status --short)"
read -r upstream_only head_only < <(
  git rev-list --left-right --count '@{upstream}...HEAD'
)
test "${upstream_only}" = 0
if test "${head_only}" -gt 0; then
  test -z "$(
    git diff --name-only '@{upstream}..HEAD' \
      | grep -vE '^(docs/|HANDOFF\.md$)' \
      || true
  )"
fi
git merge-base --is-ancestor "${code_commit}" HEAD
test -z "$(
  git diff --name-only "${code_commit}..HEAD" \
    | grep -vE '^(docs/|HANDOFF\.md$)' \
    || true
)"
test "$(sha256sum "${formal_config}" | cut -d' ' -f1)" = \
  "${formal_config_sha256}"
test "$(sha256sum "${worker}" | cut -d' ' -f1)" = "${worker_sha256}"
test "$(sha256sum "${preflight}" | cut -d' ' -f1)" = \
  "${preflight_sha256}"
test "$(sha256sum "${monitor}" | cut -d' ' -f1)" = "${monitor_sha256}"
test "$(sha256sum "${stage1_manifest}" | cut -d' ' -f1)" = \
  "${stage1_manifest_sha256}"
test "$(sha256sum "${norm_stats}" | cut -d' ' -f1)" = \
  "${norm_stats_sha256}"
test ! -e "${run_root}"
test ! -L "${run_root}"
test ! -e "${evidence_root}"
test ! -L "${evidence_root}"

mapfile -t process_rows < <(
  pgrep -af \
    'train_embodied_agent.py|rlt_stage2_formal_100c|raylet|gcs_server' \
    || true
)
test "${#process_rows[@]}" = 0
mapfile -t compute_rows < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | awk 'NF'
)
test "${#compute_rows[@]}" = 0
host_available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
disk_available_bytes="$(
  df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' '
)"
test "${host_available_kib}" -ge 419430400
test "${disk_available_bytes}" -ge 214748364800

mkdir -p "${runtime_root}"
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
export RLT_LOG_ROOT="${run_root}"
export ROBOTWIN_PI0_NORM_STATS_PATH="${norm_stats}"
export RLT_STAGE1_MODEL_PATH="${stage1_model}"
export RLT_STAGE1_MANIFEST_PATH="${stage1_manifest}"
export RLT_STAGE1_MANIFEST_ID="${stage1_manifest_id}"
export RLT_STAGE1_MANIFEST_SHA256="${stage1_manifest_sha256}"
export RLT_NORM_STATS_SHA256="${norm_stats_sha256}"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY

"${venv}/bin/python" -B "${preflight}" \
  --manifest-path "${stage1_manifest}" \
  --manifest-id "${stage1_manifest_id}" \
  --manifest-sha256 "${stage1_manifest_sha256}" \
  --stage1-model-path "${stage1_model}" \
  --norm-stats-path "${norm_stats}" \
  --norm-stats-sha256 "${norm_stats_sha256}" \
  --canonical-adapter-version robotwin_aloha_canonical_v1 \
  --action-horizon 50 \
  --action-chunk 10 \
  --action-dim 14 \
  --z-rl-dim 2048 \
  --prefix-seq-len 768 \
  --prefix-dim 2048 \
  --output "${runtime_root}/stage1_binding_preflight.json" \
  >"${runtime_root}/stage1_binding_preflight.stdout"

cp "${formal_config}" "${runtime_root}/source_config.yaml"
"${venv}/bin/python" -B \
  examples/embodiment/train_embodied_agent.py \
  --config-path "${repo}/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp \
  "runner.max_steps=100" \
  "runner.logger.log_path=${run_root}" \
  "runner.logger.experiment_name=${experiment_name}" \
  "runner.resume_dir=null" \
  --cfg job \
  --resolve >"${runtime_root}/resolved.yaml"
resolved_sha256="$(
  sha256sum "${runtime_root}/resolved.yaml" | cut -d' ' -f1
)"

RESOLVED="${runtime_root}/resolved.yaml" \
RUN_ROOT="${run_root}" \
EXPERIMENT_NAME="${experiment_name}" \
STAGE1_MODEL="${stage1_model}" \
STAGE1_MANIFEST="${stage1_manifest}" \
STAGE1_MANIFEST_ID="${stage1_manifest_id}" \
STAGE1_MANIFEST_SHA256="${stage1_manifest_sha256}" \
NORM_STATS="${norm_stats}" \
NORM_STATS_SHA256="${norm_stats_sha256}" \
  "${venv}/bin/python" -B - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf

text = Path(os.environ["RESOLVED"]).read_text()
assert "UNRESOLVED" not in text
cfg = OmegaConf.load(os.environ["RESOLVED"])
assert cfg.runner.max_epochs == 1000
assert cfg.runner.max_steps == 100
assert cfg.runner.val_check_interval == 10
assert cfg.runner.save_interval == 10
assert cfg.runner.resume_dir is None
assert cfg.runner.logger.log_path == os.environ["RUN_ROOT"]
assert cfg.runner.logger.experiment_name == os.environ["EXPERIMENT_NAME"]
assert cfg.env.train.total_num_envs == 4
assert cfg.env.eval.total_num_envs == 4
assert cfg.env.train.max_steps_per_rollout_epoch == 200
assert cfg.env.eval.max_steps_per_rollout_epoch == 200
assert cfg.env.train.task_config.task_name == "adjust_bottle"
assert cfg.env.train.task_config.embodiment == ["aloha-agilex"]
assert cfg.actor.micro_batch_size == 128
assert cfg.actor.global_batch_size == 512
assert cfg.actor.model.z_dim == 2048
assert cfg.actor.model.proprio_dim == 14
assert cfg.actor.model.action_dim == 14
assert cfg.actor.model.num_action_chunks == 10
assert cfg.actor.model.fixed_std == 0.002
assert cfg.actor.model.precision == "fp32"
assert cfg.algorithm.update_epoch == 5
assert cfg.algorithm.gamma == 0.99
assert cfg.algorithm.tau == 0.005
assert cfg.algorithm.rlt_route.type == "full_task"
assert cfg.algorithm.rlt_transition_replay.compact is True
assert cfg.algorithm.rlt_schedule.max_updates_per_train_step == 400
assert cfg.algorithm.rlt_schedule.warmup_min_size == 500
assert cfg.algorithm.rlt_schedule.warmup_post_collect_updates == 5000
assert cfg.algorithm.rlt_schedule.train_every_transitions == 1
assert cfg.algorithm.actor_weight_schedule.warmup_updates == 5000
assert cfg.algorithm.actor_weight_schedule.ramp_updates == 10000
assert cfg.algorithm.reference_dropout_prob == 0.5
assert cfg.algorithm.replay_buffer.cache_size == 15000
assert cfg.algorithm.replay_buffer.sample_window_size == 15000
assert cfg.algorithm.critic_actor_ratio == 2
assert cfg.rollout.rlt_feature_model.model_path == os.environ["STAGE1_MODEL"]
assert (
    cfg.rollout.rlt_feature_model.openpi_data.norm_stats_path
    == os.environ["NORM_STATS"]
)
contract = cfg.algorithm.rlt_resume.contract
assert contract.stage1_manifest_path == os.environ["STAGE1_MANIFEST"]
assert contract.stage1_manifest_id == os.environ["STAGE1_MANIFEST_ID"]
assert contract.stage1_manifest_sha256 == os.environ["STAGE1_MANIFEST_SHA256"]
assert contract.norm_stats_sha256 == os.environ["NORM_STATS_SHA256"]
PY

formal_cmd=(
  "${venv}/bin/python" -B
  examples/embodiment/train_embodied_agent.py
  --config-path "${repo}/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp
  "runner.max_steps=100"
  "runner.logger.log_path=${run_root}"
  "runner.logger.experiment_name=${experiment_name}"
  "runner.resume_dir=null"
)
printf '%q ' "${formal_cmd[@]}" >"${runtime_root}/exact_command.txt"
printf '\n' >>"${runtime_root}/exact_command.txt"

{
  printf 'prepared_at\t%s\n' "$(date --iso-8601=seconds)"
  printf 'branch\t%s\n' "$(git branch --show-current)"
  printf 'head\t%s\n' "$(git rev-parse HEAD)"
  printf 'upstream_only_head_only\t%s/%s\n' \
    "${upstream_only}" "${head_only}"
  printf 'code_commit\t%s\n' "${code_commit}"
  printf 'formal_config_sha256\t%s\n' "${formal_config_sha256}"
  printf 'worker_sha256\t%s\n' "${worker_sha256}"
  printf 'preflight_sha256\t%s\n' "${preflight_sha256}"
  printf 'monitor_sha256\t%s\n' "${monitor_sha256}"
  printf 'stage1_model\t%s\n' "${stage1_model}"
  printf 'stage1_manifest\t%s\n' "${stage1_manifest}"
  printf 'stage1_manifest_sha256\t%s\n' "${stage1_manifest_sha256}"
  printf 'norm_stats\t%s\n' "${norm_stats}"
  printf 'norm_stats_sha256\t%s\n' "${norm_stats_sha256}"
  printf 'resolved_config_sha256\t%s\n' "${resolved_sha256}"
  printf 'run_root\t%s\n' "${run_root}"
  printf 'experiment_name\t%s\n' "${experiment_name}"
  printf 'runtime_root\t%s\n' "${runtime_root}"
  printf 'max_steps\t100\n'
  printf 'timeout_seconds\t50400\n'
} >"${runtime_root}/run_provenance.tsv"

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
export RLT_LOG_ROOT="${run_root}"
export ROBOTWIN_PI0_NORM_STATS_PATH="${norm_stats}"
export RLT_STAGE1_MODEL_PATH="${stage1_model}"
export RLT_STAGE1_MANIFEST_PATH="${stage1_manifest}"
export RLT_STAGE1_MANIFEST_ID="${stage1_manifest_id}"
export RLT_STAGE1_MANIFEST_SHA256="${stage1_manifest_sha256}"
export RLT_NORM_STATS_SHA256="${norm_stats_sha256}"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
date --iso-8601=seconds >"${runtime_root}/started_at.txt"
timeout --signal=TERM --kill-after=180s 50400s \\
  "${venv}/bin/python" -B \\
  examples/embodiment/train_embodied_agent.py \\
  --config-path "${repo}/examples/embodiment/config" \\
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp \\
  "runner.max_steps=100" \\
  "runner.logger.log_path=${run_root}" \\
  "runner.logger.experiment_name=${experiment_name}" \\
  "runner.resume_dir=null"
rc=\$?
printf '%s\n' "\${rc}" >"${runtime_root}/exit_code.txt"
date --iso-8601=seconds >"${runtime_root}/finished_at.txt"
exit "\${rc}"
EOF
chmod 700 "${runtime_root}/run_foreground.sh"

cat >"${runtime_root}/launch_background.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
cd "${repo}"
test "\$(git branch --show-current)" = codex/rlt-pi0-robotwin
test -z "\$(git status --short)"
test "\$(sha256sum "${formal_config}" | cut -d' ' -f1)" = "${formal_config_sha256}"
test "\$(sha256sum "${worker}" | cut -d' ' -f1)" = "${worker_sha256}"
test "\$(sha256sum "${preflight}" | cut -d' ' -f1)" = "${preflight_sha256}"
test "\$(sha256sum "${monitor}" | cut -d' ' -f1)" = "${monitor_sha256}"
test "\$(sha256sum "${runtime_root}/resolved.yaml" | cut -d' ' -f1)" = "${resolved_sha256}"
test ! -e "${run_root}"
mapfile -t active_rows < <(
  pgrep -af 'train_embodied_agent.py|rlt_stage2_formal_100c|raylet|gcs_server' \\
    || true
)
test "\${#active_rows[@]}" = 0
mapfile -t compute_rows < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \\
    | awk 'NF'
)
test "\${#compute_rows[@]}" = 0
test "\$(awk '/MemAvailable:/ {print \$2}' /proc/meminfo)" -ge 419430400
test "\$(
  df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' '
)" -ge 214748364800
nohup "${runtime_root}/run_foreground.sh" \\
  >"${runtime_root}/driver.log" \\
  2>&1 \\
  </dev/null &
driver_pid=\$!
printf '%s\n' "\${driver_pid}" >"${runtime_root}/driver_pid.txt"
nohup bash "${monitor}" \\
  "\${driver_pid}" \\
  "${runtime_root}/resources.csv" \\
  2 \\
  >"${runtime_root}/monitor.log" \\
  2>&1 \\
  </dev/null &
monitor_pid=\$!
printf '%s\n' "\${monitor_pid}" >"${runtime_root}/monitor_pid.txt"
sleep 2
kill -0 "\${driver_pid}"
printf 'FORMAL_DRIVER_PID\t%s\n' "\${driver_pid}"
printf 'FORMAL_MONITOR_PID\t%s\n' "\${monitor_pid}"
printf 'RUNTIME_ROOT\t%s\n' "${runtime_root}"
printf 'RUN_ROOT\t%s\n' "${run_root}"
printf 'EXPERIMENT_NAME\t%s\n' "${experiment_name}"
printf 'RESOLVED_SHA256\t%s\n' "${resolved_sha256}"
EOF
chmod 700 "${runtime_root}/launch_background.sh"

cat >"${runtime_root}/stop_command.txt" <<EOF
kill -TERM \$(cat "${runtime_root}/driver_pid.txt")
EOF

printf 'PREPARED_AT\t%s\n' "$(date --iso-8601=seconds)"
printf 'RUNTIME_ROOT\t%s\n' "${runtime_root}"
printf 'RUN_ROOT\t%s\n' "${run_root}"
printf 'EXPERIMENT_NAME\t%s\n' "${experiment_name}"
printf 'RESOLVED_SHA256\t%s\n' "${resolved_sha256}"
printf 'EXACT_COMMAND_FILE\t%s\n' "${runtime_root}/exact_command.txt"
printf 'LAUNCH_COMMAND\tbash %s\n' "${runtime_root}/launch_background.sh"
printf 'TIMEOUT_SECONDS\t50400\n'
printf '%s\n' RLT_STAGE2_FORMAL100_PREPARED
