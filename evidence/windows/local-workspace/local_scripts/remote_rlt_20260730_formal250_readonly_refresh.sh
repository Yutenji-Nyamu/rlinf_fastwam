#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
stage1_model=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
stage1_manifest=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
pilot_root=/root/autodl-tmp/experiments/rlt_stage2_formal_100c_20260730_v1
formal_root=/root/autodl-tmp/experiments/rlt_stage2_formal_8env250_20260730_v1
smoke_root=/root/autodl-tmp/experiments/rlt_stage2_resource_smoke_8env3c_20260730_v1

printf 'observed_at\t%s\n' "$(date --iso-8601=seconds)"
printf 'hostname\t%s\n' "$(hostname)"
printf 'pwd\t%s\n' "$(pwd)"
printf 'uid\t%s\n' "$(id -u)"

cd "${repo}"
printf 'branch\t%s\n' "$(git branch --show-current)"
printf 'head\t%s\n' "$(git rev-parse HEAD)"
printf 'upstream\t%s\n' "$(git rev-parse '@{upstream}')"
printf 'left_right\t%s\n' "$(
  git rev-list --left-right --count '@{upstream}...HEAD' | tr '\t' '/'
)"
printf 'status_begin\n'
git status --short --branch
printf 'status_end\n'

printf 'source_hashes_begin\n'
sha256sum \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml \
  rlinf/envs/robotwin/robotwin_env.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_robotwin_seed_partition.py \
  toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py
printf 'source_hashes_end\n'

printf 'processes_begin\n'
pgrep -af \
  'train_embodied_agent.py|RLinf_rlt_pi0_robotwin|raylet|gcs_server|rlt_stage2_resource_monitor' \
  || true
printf 'processes_end\n'

printf 'gpu_begin\n'
nvidia-smi \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
printf 'gpu_compute_begin\n'
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits \
  || true
printf 'gpu_compute_end\n'

printf 'host_memory_begin\n'
free -b
printf 'host_memory_end\n'
printf 'cgroup_current\t%s\n' "$(cat /sys/fs/cgroup/memory.current)"
printf 'cgroup_max\t%s\n' "$(cat /sys/fs/cgroup/memory.max)"
printf 'cgroup_stat_begin\n'
grep -E '^(anon|file|kernel|shmem|inactive_file|active_file) ' \
  /sys/fs/cgroup/memory.stat
printf 'cgroup_stat_end\n'
printf 'cgroup_events_begin\n'
cat /sys/fs/cgroup/memory.events
printf 'cgroup_events_end\n'
printf 'memory_psi_begin\n'
cat /proc/pressure/memory
printf 'memory_psi_end\n'
printf 'disk_begin\n'
df -B1 /root/autodl-tmp /dev/shm
printf 'disk_end\n'

for path in \
  "${stage1_model}" \
  "${stage1_manifest}" \
  "${pilot_root}" \
  "${smoke_root}" \
  "${formal_root}"
do
  if test -e "${path}"; then
    printf 'path\texists\t%s\t%s\n' "$(du -sb "${path}" | cut -f1)" "${path}"
  else
    printf 'path\tmissing\t0\t%s\n' "${path}"
  fi
done

printf 'stage1_manifest_sha256\t%s\n' "$(sha256sum "${stage1_manifest}" | cut -d' ' -f1)"
printf 'stage1_model_completion\t%s\n' "$(
  find "${stage1_model}" -maxdepth 3 -type f \
    -name 'completion_manifest.json' -print -quit
)"
printf 'READONLY_REFRESH_OK\n'
