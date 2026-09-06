#!/usr/bin/env bash
set -u
repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac

echo '[ENTRYPOINT]'
sed -n '1,260p' "$repo/examples/embodiment/train_embodied_agent.py"

echo '[RAY_INIT_SEARCH]'
rg -n 'ray\.init|RAY_ADDRESS|RAY_TMPDIR|_temp_dir|include_dashboard|address=' "$repo/rlinf" "$repo/examples" | head -n 240 || true

echo '[PLACEMENT_SEARCH]'
rg -n 'component_placement|get_world_size\("actor"\)|global_batch_size|gradient_accumulation' "$repo/rlinf" | head -n 240 || true

echo '[CURRENT_FOREGROUND_SCRIPT_REDACTED]'
sed -E 's/(PASSWORD|TOKEN|KEY)=.*/\1=<redacted>/' /root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime/run_foreground.sh

echo '[RAY_PROCESSES_PORTS]'
ps -eo pid,pgid,cmd | grep -E 'gcs_server|raylet|dashboard' | grep -v grep || true
ss -ltnp | grep -E 'gcs_server|raylet|python' | head -n 120 || true

