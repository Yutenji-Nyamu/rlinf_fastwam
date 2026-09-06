#!/usr/bin/env bash
set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
sed -n '14,95p' "$root/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py"
grep -R -n 'def warmup_optimizer_state' "$root/rlinf/hybrid_engines" "$root/rlinf/utils"
sed -n '1,200p' "$root/rlinf/hybrid_engines/fsdp/strategy/checkpoint.py"
sed -n '640,725p' "$root/rlinf/utils/utils.py"
find "$root/rlinf/hybrid_engines/fsdp/utils" -maxdepth 1 -type f -name '*.py'
grep -R -n -E 'get_fsdp_wrap_policy|register.*hook|_no_split|custom_auto_wrap' "$root/rlinf/hybrid_engines/fsdp/utils" "$root/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py"
