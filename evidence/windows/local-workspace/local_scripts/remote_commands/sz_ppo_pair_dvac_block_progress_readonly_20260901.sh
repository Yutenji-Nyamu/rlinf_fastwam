#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/chenyiteng/results/rlinf-shenzhen/ppo/runs
CONTROL=$ROOT/ppo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-localshard-v1
DVAC=$ROOT/ppo-dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v1
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python

date --iso-8601=seconds
"$PY" - "$CONTROL/runtime/resolved.yaml" "$DVAC/runtime/resolved.yaml" <<'PY'
import json, sys, yaml
for label, path in (("control",sys.argv[1]),("dvac",sys.argv[2])):
    cfg=yaml.safe_load(open(path, encoding="utf-8"))
    print(label+"_dvac="+json.dumps(cfg["algorithm"]["dvac_gradient_weighting"], ensure_ascii=False, sort_keys=True))
PY

for item in "control:$CONTROL" "dvac:$DVAC"; do
  label=${item%%:*}; run=${item#*:}
  echo "=== $label progress ==="
  grep -aoE 'Global Step:[[:space:]]+[0-9]+/100|Generating Rollout Epochs:[^\r\n]*|Rollout Epoch:[[:space:]]+[0-9]+/4|success_once=[-+0-9.eE]+' "$run/runtime/driver.log" | tail -n 30 || true
  echo 'checkpoint tail:'
  find "$run" -type d -name 'global_step_*' -printf '%p\n' 2>/dev/null | sort -V | tail -n 5 || true
done

echo '=== memory now ==='
awk '/^MemTotal:|^MemAvailable:|^SwapTotal:|^SwapFree:/' /proc/meminfo
cat /proc/pressure/memory
