#!/usr/bin/env bash
set -u
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
MODEL=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab

date --iso-8601=seconds
echo WORKTREE
if test -d "$WT/.git" || test -f "$WT/.git"; then
  git -C "$WT" rev-parse --abbrev-ref HEAD
  git -C "$WT" rev-parse HEAD
  git -C "$WT" status --short
  git -C "$WT" diff --stat
  git -C "$WT" log -3 --date=iso --format='%h %ad %s'
  find "$WT" -type f -mmin -180 -not -path '*/.git/*' -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -80
else
  echo MISSING_WORKTREE
fi

echo MODEL_OUTPUT
if test -d "$MODEL"; then
  du -sh "$MODEL"
  find "$MODEL" -maxdepth 3 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -50
else
  echo MISSING_MODEL_OUTPUT
fi

echo RELEVANT_PROCESSES
ps -eo user,pid,ppid,pgid,etimes,rss,stat,args --sort=pid | grep -E '[s]idney|[p]i05|[c]onvert|[p]ytest|[r]uff|[t]rain_embodied_agent.py' || true

echo GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
