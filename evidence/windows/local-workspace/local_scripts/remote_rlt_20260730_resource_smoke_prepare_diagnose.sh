#!/usr/bin/env bash
set -u
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
run_root=/root/autodl-tmp/experiments/rlt_stage2_resource_smoke_8env3c_20260730_v1
evidence_root=/root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1
cd "$repo"

printf 'time\t%s\n' "$(date --iso-8601=seconds)"
printf 'branch\t%s\n' "$(git branch --show-current)"
printf 'head\t%s\n' "$(git rev-parse HEAD)"
printf 'short_head\t%s\n' "$(git rev-parse --short=8 HEAD)"
printf 'status_begin\n%s\nstatus_end\n' "$(git status --short)"
printf 'formal_sha\t%s\n' "$(sha256sum examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250.yaml | awk '{print $1}')"
printf 'smoke_sha\t%s\n' "$(sha256sum examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_resource_smoke.yaml | awk '{print $1}')"
printf 'seed_sha\t%s\n' "$(sha256sum rlinf/envs/robotwin/seeds/eval_seeds_adjust_bottle_rlt_periodic20_v1.json | awk '{print $1}')"
printf 'worker_sha\t%s\n' "$(sha256sum rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py | awk '{print $1}')"
printf 'preflight_sha\t%s\n' "$(sha256sum toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py | awk '{print $1}')"
printf 'monitor_sha\t%s\n' "$(sha256sum /root/autodl-tmp/tmp/rlt_stage2_resource_monitor_20260729.sh | awk '{print $1}')"
printf 'manifest_sha\t%s\n' "$(sha256sum /root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json | awk '{print $1}')"
printf 'stats_sha\t%s\n' "$(sha256sum /root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json | awk '{print $1}')"
printf 'run_exists\t%s\n' "$(test -e "$run_root" && echo yes || echo no)"
printf 'evidence_exists\t%s\n' "$(test -e "$evidence_root" && echo yes || echo no)"
printf 'process_begin\n'
pgrep -af 'train_embodied_agent.py|raylet|gcs_server' || true
printf 'process_end\n'
printf 'compute_begin\n'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
printf 'compute_end\n'
printf 'host_available_kib\t%s\n' "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
printf 'disk_available_bytes\t%s\n' "$(
  df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' '
)"
printf 'memory_events_begin\n'
cat /sys/fs/cgroup/memory.events
printf 'memory_events_end\n'
