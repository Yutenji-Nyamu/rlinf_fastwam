#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
grep -R -n "make_pi05_pre_post_processors\|policy_preprocessor" "$ROOT/src/lerobot" | head -100
sed -n '1,280p' "$ROOT/src/lerobot/policies/pi05/processor_pi05.py"
sed -n '1210,1270p' "$ROOT/src/lerobot/policies/pi05/modeling_pi05.py"
