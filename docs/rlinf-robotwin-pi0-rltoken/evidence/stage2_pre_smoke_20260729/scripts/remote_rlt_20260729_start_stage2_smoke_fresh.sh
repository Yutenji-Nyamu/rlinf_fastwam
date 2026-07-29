#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
smoke_root=/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1
evidence_root=/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1
runtime_root="${evidence_root}/fresh_runtime"
fresh_name=robotwin_adjust_bottle_rlt_stage2_smoke_fresh_v1
checkpoint="${smoke_root}/${fresh_name}/checkpoints/global_step_1"
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
preflight_sha256=3278a8cbdf766d30309856eac2a4eb5f8cc3c792986e230c2ef022b615553bb6
expected_resolved_sha256=c45743c1c797a9010d9a0f0c36a41c4cbabf4fd8f69e39707cb501e7b3d5c229
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
test "$(sha256sum toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py | cut -d' ' -f1)" = "${preflight_sha256}"
test -x "${venv}/bin/python"
test -d "${assets}"
test -d "${stage1_model}"
test -f "${stage1_manifest}"
test -f "${norm_stats}"
test -f "${monitor}"
test "$(sha256sum "${monitor}" | cut -d' ' -f1)" = "${monitor_sha256}"
test ! -e "${smoke_root}"
test ! -L "${smoke_root}"
test ! -e "${evidence_root}"
test ! -L "${evidence_root}"

if pgrep -af \
  'train_embodied_agent|ray::|raylet|gcs_server|robotwin_adjust_bottle_rlt_stage2' \
  | grep -v -E 'pgrep -af|start_stage2_smoke_fresh' >/dev/null
then
  printf '%s\n' "Refusing fresh smoke while an RLT/Ray process is active." >&2
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
  toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py \
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

"${venv}/bin/python" -B \
  examples/embodiment/train_embodied_agent.py \
  --config-path "${repo}/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke \
  "runner.logger.experiment_name=${fresh_name}" \
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

fresh_cmd=(
  "${venv}/bin/python" -B
  examples/embodiment/train_embodied_agent.py
  --config-path "${repo}/examples/embodiment/config"
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke
  "runner.logger.experiment_name=${fresh_name}"
  "runner.resume_dir=null"
)
printf '%q ' "${fresh_cmd[@]}" >"${runtime_root}/exact_command.txt"
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
  "${fresh_cmd[@]}"
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
  printf 'stage1_model\t%s\n' "${stage1_model}"
  printf 'stage1_manifest_sha256\t%s\n' "${stage1_manifest_sha256}"
  printf 'norm_stats_sha256\t%s\n' "${norm_stats_sha256}"
  printf 'source_config_sha256\t%s\n' "${smoke_config_sha256}"
  printf 'resolved_config_sha256\t%s\n' "${expected_resolved_sha256}"
  printf 'smoke_root\t%s\n' "${smoke_root}"
  printf 'experiment_name\t%s\n' "${fresh_name}"
  printf 'checkpoint\t%s\n' "${checkpoint}"
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
printf 'FRESH_DRIVER_PID\t%s\n' "${driver_pid}"
printf 'FRESH_MONITOR_PID\t%s\n' "${monitor_pid}"
printf 'RUNTIME_ROOT\t%s\n' "${runtime_root}"
printf 'SMOKE_ROOT\t%s\n' "${smoke_root}"
printf 'EXPECTED_CHECKPOINT\t%s\n' "${checkpoint}"
