#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh

ROOT=/data/chenyiteng/projects/robotwin-native/RoboTwin
RUN=/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1
TRAIN="$RUN/06_act_train_smoke_1epoch_not_for_eval"
EVAL="$ROOT/eval_result/adjust_bottle/ACT/demo_clean/demo_clean-50/2026-08-21_12-01-48_486194"

date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds
printf '%s\n' '=== storage ==='
df -h / /home /data
for path in /home/chenyiteng /data/chenyiteng "$ROOT" "$RUN" "$TRAIN" "$EVAL"; do
  du -sh "$path"
done
printf '%s\n' '=== gpu ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
printf '%s\n' '=== owned relevant processes ==='
ps -u chenyiteng -o pid=,args= | grep -E '[c]ollect_data|[i]mitate_episodes|[e]val_policy|[e]val\.sh|[p]olicy_server' || true
printf '%s\n' '=== proxy ==='
systemctl is-active mihomo.service
systemctl is-enabled mihomo.service
curl -L -o /dev/null -sS --retry 2 --retry-all-errors --connect-timeout 5 --max-time 15 -w 'github code=%{http_code} speed_Bps=%{speed_download} total=%{time_total}\n' https://github.com/
curl -L -o /dev/null -sS --retry 2 --retry-all-errors --connect-timeout 5 --max-time 15 -w 'hf code=%{http_code} speed_Bps=%{speed_download} total=%{time_total}\n' https://huggingface.co/
printf '%s\n' '=== source status ==='
cd "$ROOT"
git status --short
git diff --stat
git diff --check
printf '%s\n' '=== output files ==='
find "$TRAIN" -maxdepth 1 -type f -printf '%s\t%f\n' | sort
find "$EVAL" -maxdepth 1 -type f -printf '%s\t%f\n' | sort
