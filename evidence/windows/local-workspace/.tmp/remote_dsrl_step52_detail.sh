set -euo pipefail

RUN=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1

find "$RUN" -maxdepth 2 -type f -printf '%s %p\n' | sort -n

echo RECENT_ERRORS
grep -nE 'Traceback|CUDA out of memory|OutOfMemory|OOM|NaN|nan|ERROR' \
  "$RUN/formal_driver.log" | tail -n 40 || true

echo LAST_METRICS
grep -E \
  'Global Step:|success_rate=|sac/global_resident_transitions=|critic_loss=|actor_loss=|alpha=|entropy=|eval' \
  "$RUN/formal_driver.log" | tail -n 160
