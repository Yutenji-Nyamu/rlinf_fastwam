set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
CKPT_ROOT=$RUN/robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_formal_v1/checkpoints

echo "TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "HEAD=$(git -C "$REPO" rev-parse HEAD)"
echo "UPSTREAM=$(git -C "$REPO" rev-parse '@{upstream}')"
echo "STATUS_BEGIN"
git -C "$REPO" status --short
echo "STATUS_END"

echo "ERROR_SCAN_BEGIN"
grep -nE \
  'Traceback|CUDA out of memory|OutOfMemory|(^|[^[:alpha:]])(NaN|nan|Inf|inf)([^[:alpha:]]|$)|(^|[^[:alpha:]])OOM([^[:alpha:]]|$)|ERROR' \
  "$RUN/formal_driver.log" | tail -n 80 || true
echo "ERROR_SCAN_END"

for step in 65 130
do
  checkpoint=$CKPT_ROOT/global_step_$step
  test -d "$checkpoint"
  test -d "$checkpoint/actor/local_shard_checkpoint"
  test -d "$checkpoint/actor/sac_components/alpha/dcp_checkpoint"
  test -d "$checkpoint/actor/sac_components/target_model"
  test -d "$checkpoint/actor/sac_components/replay_buffer/rank_0"
  test -d "$checkpoint/actor/sac_components/replay_buffer/rank_1"
  test -z "$(find "$checkpoint" -name '*.tmp' -o -name '.metadata.tmp')"
  echo "CHECKPOINT_STEP=$step"
  echo "CHECKPOINT_BYTES=$(du -sb "$checkpoint" | awk '{print $1}')"
  echo "CHECKPOINT_FILES=$(find "$checkpoint" -type f | wc -l)"
  echo "CHECKPOINT_NEWEST=$(find "$checkpoint" -type f -printf '%T@ %p\n' | sort -n | tail -n 1)"
done

echo "CHECKPOINT_VALIDATION=PASS"
echo "LATEST_METRICS_BEGIN"
grep -E \
  'Global Step:|sac/global_resident_transitions=|critic_loss=|actor_loss=|alpha=|entropy=|eval=' \
  "$RUN/formal_driver.log" | tail -n 180
echo "LATEST_METRICS_END"
