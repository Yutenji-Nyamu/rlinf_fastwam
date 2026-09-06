#!/usr/bin/env bash

# Read-only source/resource/artifact refresh before Idea2 implementation.
export GIT_OPTIONAL_LOCKS=0

section() {
  printf '\n## %s\n' "$1"
}

section "identity"
date --iso-8601=seconds 2>/dev/null || date
hostname
pwd
id -u

section "gpu_memory_processes"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits 2>&1 || true
printf 'compute_processes:\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>&1 || true
free -h
for metric in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.high /sys/fs/cgroup/memory.events; do
  if [[ -r "$metric" ]]; then
    printf '%s: ' "$metric"
    cat "$metric"
  fi
done
df -h /root/autodl-tmp 2>&1
ps -eo pid,ppid,lstart,etime,%cpu,%mem,rss,stat,comm,args --sort=-rss | head -n 18

section "common_repo_and_worktrees"
repo=/root/autodl-tmp/RLinf
git -C "$repo" rev-parse HEAD
git -C "$repo" symbolic-ref --quiet --short HEAD || true
printf 'tracked_status:\n'
git -C "$repo" status --short --untracked-files=no
printf 'untracked_files:\n'
git -C "$repo" ls-files --others --exclude-standard
printf 'worktrees:\n'
git -C "$repo" worktree list --porcelain
printf 'base_object:\n'
git -C "$repo" cat-file -t 6d0db56bf26f972cd27fa29535f5eb939e80e5bf
git -C "$repo" show -s --format='%H%n%cI%n%s' 6d0db56bf26f972cd27fa29535f5eb939e80e5bf
printf 'target_branch_ref:\n'
git -C "$repo" show-ref --verify refs/heads/codex/idea2-dvac-pi0-robotwin 2>&1 || true
printf 'target_path:\n'
if [[ -e /root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin ]]; then
  stat -c '%F %U:%G %y %n' /root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
else
  printf 'ABSENT\n'
fi

section "runtime_and_sft_contract"
for path in \
  /root/autodl-tmp/RLinf/.venv/bin/python \
  /root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle \
  /root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/assets/physical-intelligence/robotwin/norm_stats.json; do
  if [[ -e "$path" ]]; then
    stat -c '%F %s bytes %y %n' "$path"
  else
    printf 'MISSING %s\n' "$path"
  fi
done
if [[ -f /root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/assets/physical-intelligence/robotwin/norm_stats.json ]]; then
  sha256sum /root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/assets/physical-intelligence/robotwin/norm_stats.json
fi

section "robotwin_top_level_candidates"
find /root/autodl-tmp -mindepth 1 -maxdepth 2 \
  \( -iname '*robotwin*' -o -iname '*rlinf*' \) \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %y %p\n' 2>/dev/null | sort

section "historical_rlinf_eval_logs_and_videos"
for logs in /root/autodl-tmp/RLinf*/logs; do
  [[ -d "$logs" ]] || continue
  printf 'logs_root=%s\n' "$logs"
  find "$logs" -mindepth 1 -maxdepth 1 -type d \
    \( -iname '*robotwin*' -o -iname '*openpi*' -o -iname '*pi0*' \) \
    -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -nr | head -n 30
  find "$logs" -maxdepth 7 -type f -iname '*.mp4' \
    -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort -nr | head -n 40
done

section "historical_robotwin_native_artifacts"
for root in /root/autodl-tmp/RoboTwin* /root/autodl-tmp/robotwin*; do
  [[ -d "$root" ]] || continue
  printf 'robotwin_root=%s\n' "$root"
  find "$root" -maxdepth 7 -type f \
    \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.gif' -o -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.log' -o -iname '*result*.json' \) \
    -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort -nr | head -n 100
done

section "robotwin_video_switch_sources"
while IFS= read -r source; do
  printf 'source=%s\n' "$source"
  grep -nE 'eval_video_log|render_freq|save_path|video|mp4|ffmpeg' "$source" 2>/dev/null | head -n 100 || true
done < <(find /root/autodl-tmp -maxdepth 6 -type f \
  \( -path '*/robotwin/envs/vector_env.py' -o -path '*/envs/_base_task.py' -o -path '*/script/eval_policy.py' \) \
  -print 2>/dev/null | sort -u)

section "rlinf_video_and_eval_config"
grep -nE 'save_video|video_dir|total_num_envs|eval_rollout_epoch|use_fixed_reset_state_ids|model_path|action_chunks_len' \
  /root/autodl-tmp/RLinf/evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml 2>/dev/null || true
grep -nE 'RecordVideo|save_video|video_dir' /root/autodl-tmp/RLinf/rlinf/envs/robotwin/robotwin_env.py 2>/dev/null || true

