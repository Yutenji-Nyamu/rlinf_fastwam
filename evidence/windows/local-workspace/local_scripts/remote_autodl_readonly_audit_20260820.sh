#!/usr/bin/env bash

# Read-only live inventory for the known AUTODL-A800 container.
# Missing optional commands are reported but do not stop later sections.

export GIT_OPTIONAL_LOCKS=0

section() {
  printf '\n## %s\n' "$1"
}

section "identity_and_clock"
date --iso-8601=seconds 2>/dev/null || date
hostname
pwd
id
uptime
uname -a
if [[ -r /etc/os-release ]]; then
  sed -n '1,8p' /etc/os-release
fi

section "cpu_and_memory"
printf 'nproc='; nproc
if command -v lscpu >/dev/null 2>&1; then
  lscpu | awk -F: '/^(CPU\(s\)|On-line CPU|Model name|Thread|Core|Socket|NUMA node\(s\))/{gsub(/^[ \t]+/, "", $2); print $1 ": " $2}'
fi
free -h

section "cgroup"
cat /proc/self/cgroup
for metric in \
  /sys/fs/cgroup/memory.current \
  /sys/fs/cgroup/memory.peak \
  /sys/fs/cgroup/memory.max \
  /sys/fs/cgroup/memory.high \
  /sys/fs/cgroup/memory.events \
  /sys/fs/cgroup/memory/memory.usage_in_bytes \
  /sys/fs/cgroup/memory/memory.max_usage_in_bytes \
  /sys/fs/cgroup/memory/memory.limit_in_bytes \
  /sys/fs/cgroup/memory/memory.failcnt; do
  if [[ -r "$metric" ]]; then
    printf '%s: ' "$metric"
    cat "$metric"
  fi
done

section "storage"
df -hT / /root /root/autodl-tmp 2>&1
df -ih / /root /root/autodl-tmp 2>&1
if command -v findmnt >/dev/null 2>&1; then
  findmnt -T /root/autodl-tmp -o TARGET,SOURCE,FSTYPE,OPTIONS 2>&1
fi

section "gpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits
  printf '\ncompute_processes:\n'
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>&1
else
  printf 'nvidia-smi unavailable\n'
fi

section "processes_by_rss"
ps -eo pid,ppid,user,lstart,etime,%cpu,%mem,rss,stat,comm,args --sort=-rss | head -n 40

section "sessions"
if command -v tmux >/dev/null 2>&1; then
  tmux list-sessions 2>&1 || true
else
  printf 'tmux unavailable\n'
fi
if command -v screen >/dev/null 2>&1; then
  screen -ls 2>&1 || true
else
  printf 'screen unavailable\n'
fi

section "autodl_top_level"
if [[ -d /root/autodl-tmp ]]; then
  find /root/autodl-tmp -mindepth 1 -maxdepth 1 -printf '%TY-%Tm-%TdT%TH:%TM:%TS %y %f\n' | sort -r
else
  printf '/root/autodl-tmp missing\n'
fi

section "git_worktrees"
if [[ -d /root/autodl-tmp ]]; then
  while IFS= read -r marker; do
    repo=${marker%/.git}
    if ! git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      continue
    fi
    head=$(git -C "$repo" rev-parse HEAD 2>/dev/null || printf 'NO_HEAD')
    branch=$(git -C "$repo" symbolic-ref --quiet --short HEAD 2>/dev/null || printf 'DETACHED')
    tracked_dirty=$(git -C "$repo" status --porcelain=v1 --untracked-files=no 2>/dev/null | wc -l)
    untracked=$(git -C "$repo" ls-files --others --exclude-standard 2>/dev/null | wc -l)
    last_commit=$(git -C "$repo" log -1 --format='%cI %s' 2>/dev/null || true)
    printf 'repo=%s\nbranch=%s\nhead=%s\ntracked_dirty=%s untracked=%s\nlast_commit=%s\n--\n' \
      "$repo" "$branch" "$head" "$tracked_dirty" "$untracked" "$last_commit"
  done < <(find /root/autodl-tmp -maxdepth 4 -name .git -print 2>/dev/null | sort)
fi

section "recent_project_files_30d"
if [[ -d /root/autodl-tmp ]]; then
  find /root/autodl-tmp \
    -path '*/.git' -prune -o \
    -path '*/.venv' -prune -o \
    -path '*/site-packages' -prune -o \
    -path '*/__pycache__' -prune -o \
    -path '*/.cache' -prune -o \
    -type f -mtime -30 -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null \
    | sort -nr | head -n 80
fi

section "checkpoint_directories"
if [[ -d /root/autodl-tmp ]]; then
  find /root/autodl-tmp -maxdepth 8 -type d \
    \( -name 'global_step_*' -o -name 'checkpoint-*' -o -name 'checkpoints' \) \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -r | head -n 100
fi
