#!/usr/bin/env bash
set -euo pipefail

PI0_SOURCE=/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1
FASTWAM_SOURCE=/data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2
VIDEO_ROOT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2
OUTPUT=/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-pi0-fixed64-fastwam-adjust-p1-v1
PACKET=/data/chenyiteng/results/dvac-observation/packets/sz-dvac-analysis-v1
FASTWAM_PY=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python

printf 'IDENTITY\n'
id
hostname
date -Is

printf 'SOURCE_PREDICATES\n'
test -d "$PI0_SOURCE"
test -d "$FASTWAM_SOURCE"
test -d "$VIDEO_ROOT"
test ! -e "$OUTPUT"
test ! -e "$PACKET"
test -x "$FASTWAM_PY"
printf 'pi0_source=%s\nfastwam_source=%s\nvideo_root=%s\noutput_absent=true\npacket_absent=true\n' \
  "$PI0_SOURCE" "$FASTWAM_SOURCE" "$VIDEO_ROOT"

printf 'SOURCE_MARKERS\n'
find "$PI0_SOURCE" -type f \( -name 'trace_rollout_rank*.npz' -o -name 'query_index_rollout_rank*.csv' -o -name 'episode_index_env_rank*.csv' \) -printf '%f\n' | sort
find "$FASTWAM_SOURCE" -maxdepth 2 -type f \( -name 'run_manifest.json' -o -name 'queries.csv' -o -name 'episodes.csv' -o -name '*.npz' \) -printf '%P\n' | sort | sed -n '1,12p'

printf 'SOURCE_BYTES\n'
du -sb "$PI0_SOURCE" "$FASTWAM_SOURCE" "$VIDEO_ROOT"

printf 'PYTHON_IMPORT_PREFLIGHT\n'
"$FASTWAM_PY" - <<'PY'
import importlib
import sys

print(f"python={sys.executable}")
for name in ("numpy", "pandas", "PIL", "matplotlib", "cv2"):
    module = importlib.import_module(name)
    print(f"{name}={getattr(module, '__version__', 'import-ok')}")
PY

printf 'PREFLIGHT_OK\n'
