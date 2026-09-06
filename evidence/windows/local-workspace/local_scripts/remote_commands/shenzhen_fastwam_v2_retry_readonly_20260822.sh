#!/usr/bin/env bash
set -uo pipefail

meta='/data/chenyiteng/results/dvac-observation/run-metadata/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2'
payload='/data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2'
official='/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2'

date --iso-8601=seconds
hostname
id

for path in "$meta" "$payload" "$official"; do
  if [[ -e "$path" ]]; then
    stat -c 'exists type=%F size=%s mtime=%y path=%n' "$path"
    find "$path" -maxdepth 5 -printf '%TY-%Tm-%TdT%TH:%TM:%TS %y %s %p\n' 2>/dev/null | sort
  else
    printf 'absent=%s\n' "$path"
  fi
done

printf 'metadata_text_files\n'
for file in "$meta"/*; do
  [[ -f "$file" ]] || continue
  case "$file" in
    *.log|*.txt|*.yaml|*.json|*.sha256|*.pid|*.rc)
      printf 'file=%s\n' "$file"
      tail -n 160 "$file"
      ;;
  esac
done

printf 'official_logs\n'
find "$official" -type f -name '*.log' -print -exec tail -n 160 {} \; 2>/dev/null || true

printf 'candidate_processes\n'
ps -u chenyiteng -o user=,pid=,ppid=,etimes=,rss=,stat=,comm=,args= | grep -F 'fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2' | grep -v grep || true

printf 'gpu3\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader | sed -n '4p'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader || true

printf 'fastwam_v2_probe_complete\n'
