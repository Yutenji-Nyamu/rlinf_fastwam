#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
run_root=/root/autodl-tmp/experiments/rlt_stage2_formal_100c_20260730_v1
evidence_root=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1
stage1_model=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
stage1_manifest=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
norm_stats=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
formal_config=examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml
worker=rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py
preflight=toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py
monitor=/root/autodl-tmp/tmp/rlt_stage2_resource_monitor_20260729.sh

cd "${repo}"
branch="$(git branch --show-current)"
head="$(git rev-parse HEAD)"
test "${branch}" = codex/rlt-pi0-robotwin
test -z "$(git status --short)"
read -r upstream_only head_only < <(
  git rev-list --left-right --count '@{upstream}...HEAD'
)
# One terminal docs-only closeout may remain unpublished while GitHub main is
# unavailable. Runtime code/config must still match the approved code commit.
test "${upstream_only}" = 0
if test "${head_only}" -gt 0; then
  test -z "$(
    git diff --name-only '@{upstream}..HEAD' \
      | grep -vE '^(docs/|HANDOFF\.md$)' \
      || true
  )"
fi
git merge-base --is-ancestor \
  3b610cb4685a1d41c97da64df67ab86561697dfd HEAD
test -z "$(
  git diff --name-only \
    3b610cb4685a1d41c97da64df67ab86561697dfd..HEAD \
    | grep -vE '^(docs/|HANDOFF\.md$)' \
    || true
)"

test -x "${venv}/bin/python"
test -d "${assets}"
test -d "${stage1_model}"
test -f "${stage1_manifest}"
test -f "${norm_stats}"
test -f "${formal_config}"
test -f "${worker}"
test -f "${preflight}"
test -f "${monitor}"
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
mapfile -t gpu_compute < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | awk 'NF'
)
test "${#gpu_compute[@]}" = 0
mapfile -t gpu_rows < <(
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
)
test "${#gpu_rows[@]}" = 2
for row in "${gpu_rows[@]}"; do
  IFS=, read -r index used total util <<<"${row}"
  used="${used// /}"
  util="${util// /}"
  test "${used}" -le 16
  test "${util}" -le 5
done

host_available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
disk_available_bytes="$(
  df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' '
)"
test "${host_available_kib}" -ge 419430400
test "${disk_available_bytes}" -ge 214748364800

FORMAL="${formal_config}" "${venv}/bin/python" -B - <<'PY'
import os
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.environ["FORMAL"])
assert cfg.runner.max_steps == 0
assert cfg.runner.max_epochs == 1000
assert cfg.runner.val_check_interval == 10
assert cfg.runner.save_interval == 10
assert cfg.env.train.total_num_envs == 4
assert cfg.env.eval.total_num_envs == 4
assert cfg.env.train.max_steps_per_rollout_epoch == 200
assert cfg.actor.micro_batch_size == 128
assert cfg.actor.global_batch_size == 512
assert cfg.actor.model.num_action_chunks == 10
assert cfg.actor.model.action_dim == 14
assert cfg.algorithm.rlt_route.type == "full_task"
assert cfg.algorithm.rlt_schedule.warmup_min_size == 500
assert cfg.algorithm.rlt_schedule.warmup_post_collect_updates == 5000
assert cfg.algorithm.rlt_schedule.max_updates_per_train_step == 400
assert cfg.algorithm.rlt_schedule.train_every_transitions == 1
assert cfg.algorithm.update_epoch == 5
assert cfg.algorithm.critic_actor_ratio == 2
assert cfg.algorithm.replay_buffer.sample_window_size == 15000
PY

printf 'audit_time\t%s\n' "$(date --iso-8601=seconds)"
printf 'branch\t%s\n' "${branch}"
printf 'head\t%s\n' "${head}"
printf 'upstream_only_head_only\t%s/%s\n' \
  "${upstream_only}" "${head_only}"
printf 'formal_config_sha256\t%s\n' "$(
  sha256sum "${formal_config}" | cut -d' ' -f1
)"
printf 'worker_sha256\t%s\n' "$(sha256sum "${worker}" | cut -d' ' -f1)"
printf 'preflight_sha256\t%s\n' "$(
  sha256sum "${preflight}" | cut -d' ' -f1
)"
printf 'monitor_sha256\t%s\n' "$(sha256sum "${monitor}" | cut -d' ' -f1)"
printf 'stage1_manifest_sha256\t%s\n' "$(
  sha256sum "${stage1_manifest}" | cut -d' ' -f1
)"
printf 'norm_stats_sha256\t%s\n' "$(
  sha256sum "${norm_stats}" | cut -d' ' -f1
)"
printf 'process_count\t%s\n' "${#process_rows[@]}"
printf 'gpu_rows\t%s | %s\n' "${gpu_rows[0]}" "${gpu_rows[1]}"
printf 'host_available_kib\t%s\n' "${host_available_kib}"
printf 'cgroup_current_bytes\t%s\n' "$(cat /sys/fs/cgroup/memory.current)"
printf 'cgroup_anon_bytes\t%s\n' "$(
  awk '$1 == "anon" {print $2}' /sys/fs/cgroup/memory.stat
)"
printf 'cgroup_file_bytes\t%s\n' "$(
  awk '$1 == "file" {print $2}' /sys/fs/cgroup/memory.stat
)"
printf 'memory_events\t%s\n' "$(
  tr '\n' ' ' </sys/fs/cgroup/memory.events
)"
printf 'disk_available_bytes\t%s\n' "${disk_available_bytes}"
printf 'run_root_absent\tyes\n'
printf 'evidence_root_absent\tyes\n'
printf '%s\n' RLT_STAGE2_FORMAL100_LIVE_AUDIT_OK
