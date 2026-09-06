#!/usr/bin/env bash
set -euo pipefail

FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
CACHE=/home/chenyiteng/cache/fastwam-7faa

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
printf 'source_head=%s\n' "$(git -C "$FW" rev-parse HEAD)"
git -C "$FW" status --short

printf '%s\n' '=== partial environment boundary ==='
"$ENV/bin/python" --version
"$ENV/bin/python" -m pip show torch torchvision fastwam 2>/dev/null || true
"$ENV/bin/python" -m pip show pip setuptools wheel ninja setuptools-scm | \
  grep -E '^(Name|Version):' || true
du -sh "$FW" "$ENV" "$CACHE"

printf '%s\n' '=== no owned deployment process ==='
ACTIVE="$(ps -u chenyiteng -o pid=,comm=,args= | awk '$2 ~ /^python/ && tolower($0) ~ /fastwam/')"
if [[ -n "$ACTIVE" ]]; then
  printf '%s\n' "$ACTIVE"
  exit 40
fi

printf '%s\n' '=== GPUs and Ray ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
pgrep -a -x raylet || true
pgrep -a -x gcs_server || true
printf '%s\n' FASTWAM_NIGHT_BREAKPOINT_OK
