#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
ACTION="$ROOT/dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1"
ST="$ROOT/dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1"
PRISM="$ROOT/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-localshard-phys67-v2"
ACTION_WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv-fix
ST_WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
PRISM_WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/prism-dvac-rank-rloo
FIX=306ce2e98a06b6f439a1070d8942e20132e48d49
PYTHON=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python

printf 'now='; TZ=Asia/Shanghai date --iso-8601=seconds

for item in "ACTION:$ACTION:$ACTION_WT" "ST:$ST:$ST_WT" "PRISM:$PRISM:$PRISM_WT"; do
  label=${item%%:*}; rest=${item#*:}; run=${rest%%:*}; wt=${rest#*:}
  printf '\n[%s]\n' "$label"
  printf 'runtime_source_head='; sed -n 's/^source_head=//p' "$run/runtime/launch_manifest.txt" | head -n1
  printf 'worktree_head='; git -C "$wt" rev-parse HEAD
  printf 'worktree_branch='; git -C "$wt" branch --show-current
  printf 'worktree_dirty='; test -z "$(git -C "$wt" status --short)" && echo no || echo yes
  if git -C "$wt" merge-base --is-ancestor "$FIX" HEAD 2>/dev/null; then
    echo local_shard_fix_ancestor=yes
  else
    echo local_shard_fix_ancestor=no
  fi
  "$PYTHON" - "$run/runtime/resolved.yaml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
fsdp = cfg.get("actor", {}).get("fsdp_config", {})
print("resolved_checkpoint_format=" + str(fsdp.get("checkpoint_format", "<missing/default-dcp>")))
print("save_interval=" + str(cfg["runner"]["save_interval"]))
PY
  printf 'manager_checkpoint_format_lines='; grep -nE '_get_checkpoint_format|checkpoint_format=' "$wt/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py" | paste -sd ';' - || true
done

for item in "ACTION:$ACTION" "ST:$ST" "PRISM:$PRISM"; do
  label=${item%%:*}; run=${item#*:}
  printf '\n[%s_STEP10]\n' "$label"
  checkpoint=$(find "$run" -type d -name global_step_10 -print -quit)
  printf 'checkpoint=%s\n' "$checkpoint"
  if [[ -n "$checkpoint" ]]; then
    printf 'files=%s bytes=%s metadata=%s local_shards=%s complete=%s\n' \
      "$(find "$checkpoint" -type f | wc -l)" \
      "$(du -sb "$checkpoint" | awk '{print $1}')" \
      "$(find "$checkpoint" -name .metadata -type f | wc -l)" \
      "$(find "$checkpoint" -path '*/local_shard_checkpoint/checkpoint_rank_*.pt' -type f | wc -l)" \
      "$(find "$checkpoint" -name complete.json -type f | wc -l)"
    find "$checkpoint" -maxdepth 4 -type f -printf '%P %s\n' | sort | head -n 24
  fi
done

printf '\n[PROCESS]\n'
for run in "$ACTION" "$ST"; do
  pid=$(cat "$run/runtime/wrapper.pid")
  printf '%s wrapper=%s alive=' "$(basename "$run")" "$pid"
  kill -0 "$pid" 2>/dev/null && echo yes || echo no
  stat -c 'driver_bytes=%s driver_mtime=%y' "$run/runtime/driver.log"
done
