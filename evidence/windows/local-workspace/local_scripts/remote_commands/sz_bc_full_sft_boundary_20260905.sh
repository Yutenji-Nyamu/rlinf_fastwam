#!/usr/bin/env bash
set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
run=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v3
date -Is
tail -n 125 "$run/driver.log"
cat "$run/exit_code.txt" "$run/finished_at.txt"
cat "$root/rlinf/models/embodiment/openpi/__init__.py"
sed -n '250,410p' "$venv/lib/python3.11/site-packages/openpi/models_pytorch/pi0_pytorch.py"
sed -n '145,210p' "$root/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py"
grep -nA 15 'def torch_dtype_from_precision' "$root/rlinf/utils/utils.py"
nvidia-smi -i 6 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
