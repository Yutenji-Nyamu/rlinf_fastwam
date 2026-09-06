set -euo pipefail

RUN=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
SESSION=/tmp/ray/session_2026-07-28_18-54-21_566345_70062

echo "RUN_TREE_BEGIN"
find "$RUN" -maxdepth 4 -type d -printf '%p\n' | sort
echo "RUN_TREE_END"
echo "RUN_NON_CHECKPOINT_SIZE=$(du -sh --exclude=checkpoints "$RUN" | awk '{print $1}')"
echo "RAY_SIZE=$(du -sh "$SESSION/logs" | awk '{print $1}')"
echo "RAY_TOP_BEGIN"
find "$SESSION/logs" -maxdepth 1 -type f -printf '%s %f\n' |
  sort -nr | head -n 60
echo "RAY_TOP_END"
