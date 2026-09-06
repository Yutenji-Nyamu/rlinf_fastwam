#!/usr/bin/env bash

# Focused read-only follow-up after the broad AUTODL-A800 inventory.
export GIT_OPTIONAL_LOCKS=0

section() {
  printf '\n## %s\n' "$1"
}

repo_state() {
  local repo=$1
  printf '\nrepo=%s\n' "$repo"
  if [[ ! -e "$repo/.git" ]]; then
    printf 'missing_git_worktree\n'
    return
  fi
  git -C "$repo" symbolic-ref --quiet --short HEAD 2>/dev/null || printf 'DETACHED\n'
  git -C "$repo" rev-parse HEAD 2>/dev/null || true
  git -C "$repo" log -1 --format='%cI %s' 2>/dev/null || true
  printf 'tracked_status:\n'
  git -C "$repo" status --short --untracked-files=no 2>/dev/null || true
  printf 'untracked_count='
  git -C "$repo" ls-files --others --exclude-standard 2>/dev/null | wc -l
  local branch
  branch=$(git -C "$repo" symbolic-ref --quiet --short HEAD 2>/dev/null || true)
  if [[ -n "$branch" ]] && git -C "$repo" show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    printf 'origin_ahead_behind='
    git -C "$repo" rev-list --left-right --count "origin/$branch...$branch" 2>/dev/null || true
  fi
}

section "live_idle_recheck"
date --iso-8601=seconds 2>/dev/null || date
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits 2>&1
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>&1
ps -eo pid,ppid,lstart,etime,%cpu,%mem,rss,stat,comm,args --sort=-rss | head -n 18

section "relevant_repo_state"
for repo in \
  /root/autodl-tmp/RLinf \
  /root/autodl-tmp/RLinf_fastwam_rlinf \
  /root/autodl-tmp/RLinf_rlt_pi0_robotwin \
  /root/autodl-tmp/RLinf_qam_pi0_robotwin \
  /root/autodl-tmp/RLinf_ogpo_pi0_robotwin; do
  repo_state "$repo"
done

section "identifier_8cde1ff"
for repo in \
  /root/autodl-tmp \
  /root/autodl-tmp/RLinf \
  /root/autodl-tmp/RLinf_fastwam_rlinf \
  /root/autodl-tmp/RLinf_rlt_pi0_robotwin \
  /root/autodl-tmp/RLinf_qam_pi0_robotwin \
  /root/autodl-tmp/RLinf_ogpo_pi0_robotwin; do
  if [[ -e "$repo/.git" ]] && git -C "$repo" cat-file -e '8cde1ff^{commit}' 2>/dev/null; then
    printf 'found_in=%s\n' "$repo"
    git -C "$repo" show -s --format='%H%n%cI%n%s' 8cde1ff
  fi
done

section "experiment_roots_by_mtime"
find /root/autodl-tmp/experiments -mindepth 1 -maxdepth 1 -type d \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -r | head -n 50

section "ogpo_experiment_sizes_and_checkpoints"
while IFS= read -r exp; do
  printf '\nexperiment=%s\n' "$exp"
  du -sh --apparent-size "$exp" 2>/dev/null || true
  find "$exp" -type d -name 'global_step_*' \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -r
done < <(find /root/autodl-tmp/experiments -mindepth 1 -maxdepth 1 -type d -name '*ogpo*' -print 2>/dev/null | sort)

section "latest_ogpo_logs"
while IFS= read -r log_record; do
  log_path=${log_record#* }
  printf '\nlog=%s\n' "$log_path"
  tail -n 30 "$log_path" 2>/dev/null || true
done < <(
  find /root/autodl-tmp -maxdepth 6 -type f \
    \( -iname '*ogpo*.log' -o -path '*ogpo*/*.log' \) \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 8
)

section "latest_ogpo_checkpoint_metadata"
while IFS= read -r checkpoint_record; do
  checkpoint=${checkpoint_record#* }
  printf '\ncheckpoint=%s\n' "$checkpoint"
  du -sh --apparent-size "$checkpoint" 2>/dev/null || true
  find "$checkpoint" -maxdepth 2 -type f \
    -printf '%P %s bytes %TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort | head -n 80
done < <(
  find /root/autodl-tmp/experiments -type d -path '*ogpo*' -name 'global_step_*' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 4
)

section "top_level_ogpo_files"
find /root/autodl-tmp -maxdepth 2 -type f -iname '*ogpo*' \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort -r
