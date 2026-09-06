#!/usr/bin/env bash
set -uo pipefail
RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
printf 'timestamp=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)"
printf '%s\n' '=== metrics tail ==='
tail -n 180 "$RUN/metrics.log"
printf '%s\n' '=== driver scalar/progress tail ==='
grep -aE 'Global Step:|env/success_once|eval/success_once|train/actor/(approx_kl|clip_fraction|grad_norm)|train/critic/(value_loss|explained_variance)|time/(generate_rollouts|actor_training|step)' "$RUN/driver.log" | tail -n 160
exit 0
