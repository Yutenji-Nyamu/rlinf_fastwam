set -eu
base=/data/chenyiteng/results/rlinf-shenzhen

classify_grpo() {
  run="$1"; step="$2"
  case "$run" in
    dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1) [ "$step" = 60 ] && echo KEEP || echo DELETE ;;
    dvac-action-adv-fix-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1) [ "$step" = 80 ] && echo KEEP || echo DELETE ;;
    dvac-action-adv-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1) [ "$step" = 20 ] && echo KEEP || echo DELETE ;;
    dvac-action-adv-w0to2-smoke2-2gpu64x4-b1024-noeval-phys23-v1) echo DELETE ;;
    dvac-global-z-formal100-4gpu32x8-g8-phys2367-v2) echo DELETE ;;
    dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v4) echo DELETE ;;
    dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v5-resume30) [ "$step" = 40 ] && echo KEEP || echo DELETE ;;
    dvac-global-z-smoke2-4gpu128train64eval-v1) echo DELETE ;;
    dvac-global-z-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v2) [ "$step" = 50 ] && echo KEEP || echo DELETE ;;
    dvac-global-z-w0to5-formal100-grpo-matched-4gpu128x4-b2048-eval5-phys4567-v1) [ "$step" = 40 ] && echo KEEP || echo DELETE ;;
    dvac-st-global-z-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1) [ "$step" = 50 ] && echo KEEP || echo DELETE ;;
    dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v2) [ "$step" = 40 ] && echo KEEP || echo DELETE ;;
    dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1) echo DELETE ;;
    grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2) [ "$step" = 90 ] && echo KEEP || echo DELETE ;;
    grpo-formal100-current-4gpu128train64eval-ppo-matched-v2) [ "$step" = 50 ] && echo KEEP || echo DELETE ;;
    prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-localshard-phys67-v2) [ "$step" = 50 ] && echo KEEP || echo DELETE ;;
    prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1) echo DELETE ;;
    prism-dvac-rank-rloo-smoke1-2gpu64x4-b1024-noeval-phys23-v1) echo DELETE ;;
    *) echo REVIEW ;;
  esac
}

classify_ppo() {
  run="$1"; step="$2"
  case "$run" in
    ppo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-localshard-v1) [ "$step" = 60 ] && echo KEEP || echo DELETE ;;
    ppo-dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v1) [ "$step" = 50 ] && echo KEEP || echo DELETE ;;
    ppo-control-smoke1-2gpu64x4-b1024-noeval-localshard-phys23-v1) echo DELETE ;;
    ppo-dvac-action-adv-fix-w0to2-smoke2-2gpu64x4-b1024-noeval-localshard-phys23-v1) echo DELETE ;;
    *) echo REVIEW ;;
  esac
}

emit_ck() {
  family="$1"; run="$2"; ck="$3"; action="$4"
  step=${ck##*/global_step_}
  bytes=$(du -sx -B1 "$ck" | awk '{print $1}')
  files=$(find "$ck" -xdev -type f -printf . | wc -c)
  printf 'CK|%s|%s|step=%s|%s|bytes=%s|files=%s|%s\n' "$family" "$run" "$step" "$action" "$bytes" "$files" "$ck"
  if [ "$action" = DELETE ]; then
    find "$ck" -xdev -type f -printf '%s|%p\n' | sort -t'|' -k1,1nr | head -n 2 | sed 's/^/TARGET|/'
  fi
}

for rundir in "$base/grpo/runs"/*; do
  [ -d "$rundir" ] || continue
  run=$(basename "$rundir")
  find "$rundir" -xdev -type d -name 'global_step_*' -print | sort -V | while IFS= read -r ck; do
    step=${ck##*/global_step_}
    action=$(classify_grpo "$run" "$step")
    emit_ck GRPO "$run" "$ck" "$action"
  done
done

for rundir in "$base/ppo/runs"/*; do
  [ -d "$rundir" ] || continue
  run=$(basename "$rundir")
  find "$rundir" -xdev -type d -name 'global_step_*' -print | sort -V | while IFS= read -r ck; do
    step=${ck##*/global_step_}
    action=$(classify_ppo "$run" "$step")
    emit_ck PPO "$run" "$ck" "$action"
  done
done

for run in ppo-formal100-4gpu128train64eval-official-v1 ppo-oneopt-4gpu128train64eval-v1; do
  rundir="$base/ppo/$run"
  [ -d "$rundir" ] || continue
  find "$rundir" -xdev -type d -name 'global_step_*' -print | sort -V | while IFS= read -r ck; do
    step=${ck##*/global_step_}
    if [ "$run" = ppo-formal100-4gpu128train64eval-official-v1 ] && [ "$step" = 40 ]; then action=KEEP; else action=DELETE; fi
    emit_ck PPO_LEGACY "$run" "$ck" "$action"
  done
done
