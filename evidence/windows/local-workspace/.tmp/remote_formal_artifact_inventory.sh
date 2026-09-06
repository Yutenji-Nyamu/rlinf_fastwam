set -euo pipefail

RUN=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
echo "TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "CHECKPOINT_DIRS"
find "$RUN" -type d -path '*/checkpoints/global_step_*' -print 2>/dev/null || true
echo "TOP_LEVEL"
find "$RUN" -maxdepth 2 -type f -printf '%s %p\n' | sort -n
du -sh "$RUN"
df -h /root/autodl-tmp
