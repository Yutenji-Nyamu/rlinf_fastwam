#!/usr/bin/env bash
set -eu
date -Is
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
free -b
ps -u chenyiteng -o pid,pgid,etime,rss,args | grep -E 'oidn_toggle_trial.py|ray::EnvWorker|ray::MultiStepRolloutWorker' | cut -c1-550
for mode in oidn none; do
  dir=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/diagnostics/oidn-toggle-20260904/$mode
  test ! -f "$dir/exit_code.txt" || cat "$dir/exit_code.txt"
  test ! -f "$dir/trial.log" || tail -18 "$dir/trial.log"
done
