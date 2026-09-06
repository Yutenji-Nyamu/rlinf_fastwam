#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
CANON=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
POLICY_REL=third_party/RoboTwin/policy/fastwam_policy

echo "TIME_UTC=$(date -u +%FT%TZ)"
for root in "$WT" "$CANON"; do
  echo "ROOT_BEGIN=$root"
  git -C "$root" rev-parse HEAD
  ls -ld \
    "$root/third_party/RoboTwin" \
    "$root/third_party/RoboTwin/assets" \
    "$root/third_party/RoboTwin/task_config" \
    "$root/$POLICY_REL" 2>&1 || true
  readlink "$root/$POLICY_REL" || true
  readlink -f "$root/$POLICY_REL" || true
  test -f "$root/third_party/RoboTwin/envs/adjust_bottle.py" && echo ADJUST_BOTTLE_SOURCE_OK
  test -f "$root/third_party/RoboTwin/script/eval_policy.py" && echo EVAL_POLICY_SOURCE_OK
  echo "ROOT_END=$root"
done

echo MODEL_ENV_BEGIN
test -x /home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
stat -c '%n bytes=%s' \
  /data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384.pt \
  /data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384_dataset_stats.json
echo MODEL_ENV_END

echo TARGET_ABSENCE_BEGIN
for p in \
  /data/chenyiteng/results/dvac-observation/fastwam-adjust-bottle-p0-off \
  /data/chenyiteng/results/dvac-observation/fastwam-adjust-bottle-p0-on; do
  if test -e "$p"; then echo "EXISTS=$p"; else echo "ABSENT=$p"; fi
done
echo TARGET_ABSENCE_END
echo FASTWAM_DVAC_P0_LAYOUT_PROBE_OK
