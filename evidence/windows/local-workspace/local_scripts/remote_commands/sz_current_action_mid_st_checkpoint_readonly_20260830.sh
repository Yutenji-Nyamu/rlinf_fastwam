#!/usr/bin/env bash
set -euo pipefail

root=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
action=$root/dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1
st=$root/dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1

printf 'now='; TZ=Asia/Shanghai date --iso-8601=seconds
for item in "ACTION:$action" "ST:$st"; do
  label=${item%%:*}; run=${item#*:}
  printf '\n%s\n' "$label"
  stat -c 'driver_bytes=%s driver_mtime=%y' "$run/runtime/driver.log"
  grep -aE 'Global Step:|Saving checkpoint|checkpoint.*saved|Finished saving|Traceback|RayActorError|out of memory' "$run/runtime/driver.log" | tail -n 12 || true
  printf 'step10_dirs='; find "$run" -type d -name global_step_10 | wc -l
  checkpoint=$(find "$run" -type d -name global_step_10 -print -quit)
  if [[ -n "$checkpoint" ]]; then
    printf 'checkpoint=%s\n' "$checkpoint"
    printf 'files=%s bytes=%s metadata=%s complete=%s newest=' \
      "$(find "$checkpoint" -type f | wc -l)" \
      "$(du -sb "$checkpoint" | awk '{print $1}')" \
      "$(find "$checkpoint" -name .metadata -type f | wc -l)" \
      "$(find "$checkpoint" -name complete.json -type f | wc -l)"
    find "$checkpoint" -type f -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS%Tz %s %p\n' | sort -n | tail -n 1 || true
  fi
done

printf '\nST_GPU_PROCESSES\n'
for dev in 6 7; do
  nvidia-smi -i "$dev" --query-compute-apps=pid --format=csv,noheader,nounits | sort -nu | while read -r pid; do
    [[ -n "$pid" ]] || continue
    ps -o pid=,etimes=,stat=,wchan:28=,args= -p "$pid"
  done
done
