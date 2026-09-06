#!/usr/bin/env bash
set -u

PI0=/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-p1-16ep-800baf80-v1
FW=/data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2
FWWT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
FWOFF="$FWWT/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2"
FWCAN=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
FWMODEL=/data/chenyiteng/models/fastwam/release-8eaceeb

echo '=== TIME_IDENTITY ==='
date --iso-8601=seconds
hostname
id

summarize_tree() {
  label="$1"
  root="$2"
  printf '\nTREE label=%s path=%s\n' "$label" "$root"
  if [ ! -e "$root" ]; then
    echo 'absent'
    return
  fi
  du -sb "$root"
  for ext in csv npz png mp4; do
    find "$root" -type f -iname "*.$ext" -printf '%s\n' 2>/dev/null | awk -v e="$ext" '{n++; b+=$1} END {printf "%s_count=%d %s_bytes=%d\n",e,n,e,b}'
  done
  find "$root" -type f -name '*.csv' -printf '%s %p\n' 2>/dev/null | sort
}

echo '=== TELEMETRY_AND_VIDEO_INVENTORY ==='
summarize_tree pi0_p1_root "$PI0"
summarize_tree pi0_dvac_telemetry "$PI0/dvac_telemetry"
summarize_tree fastwam_v2_payload "$FW"
summarize_tree fastwam_v2_official_result "$FWOFF"

echo '=== CSV_ROWS ==='
for f in "$PI0"/*.csv "$PI0"/dvac_telemetry/*.csv "$FW"/*.csv "$FWOFF"/*.csv; do
  [ -f "$f" ] || continue
  printf 'lines='; wc -l < "$f"
  printf 'path=%s\n' "$f"
done

echo '=== FASTWAM_GIT_LOCK ==='
git -C "$FWWT" rev-parse HEAD
git -C "$FWWT" status --short
git -C "$FWWT" branch --show-current

echo '=== FASTWAM_RELEVANT_TRACKED_PATHS ==='
git -C "$FWWT" ls-tree -r --name-only HEAD | grep -Ei '(^|/)(README[^/]*|eval_robotwin_single\.py|.*robotwin.*\.ya?ml|.*robotwin.*config.*\.py)$' | head -n 160 || true

echo '=== FASTWAM_README_TASK_CHECKPOINT_EVIDENCE ==='
for f in "$FWWT/README.md" "$FWWT/README_zh.md" "$FWWT/experiments/robotwin/README.md"; do
  [ -f "$f" ] || continue
  printf '\nFILE=%s\n' "$f"
  grep -niE 'RoboTwin|multi[- ]?task|task|checkpoint|robotwin_uncond_3cam_384|evaluation|inference' "$f" | head -n 180 || true
done

echo '=== FASTWAM_EVAL_AND_CONFIG_TASK_FIELDS ==='
for f in \
  "$FWWT/experiments/robotwin/eval_robotwin_single.py" \
  "$FWWT/experiments/robotwin/eval_robotwin.py" \
  "$FWWT/experiments/robotwin/config/robotwin_uncond_3cam_384.yaml" \
  "$FWWT/experiments/robotwin/configs/robotwin_uncond_3cam_384.yaml" \
  "$FWWT/configs/robotwin_uncond_3cam_384.yaml"; do
  [ -f "$f" ] || continue
  printf '\nFILE=%s\n' "$f"
  grep -niE 'task_name|task_config|eval_num_episodes|checkpoint|dataset_stats|statistics|instruction|sigma_shift|replan|robotwin_uncond' "$f" | head -n 240 || true
done

echo '=== FASTWAM_ALL_RELEVANT_CONFIG_FIELDS ==='
find "$FWWT" -maxdepth 5 -type f \( -name '*.yaml' -o -name '*.yml' -o -name '*.py' \) -print0 2>/dev/null \
  | xargs -0 grep -nIE 'task_name|task_config|robotwin_uncond_3cam_384\.pt|dataset_statistics\.json' 2>/dev/null \
  | head -n 260 || true

echo '=== RELEASE_FILES_AND_STATS_TOPLEVEL ==='
find "$FWMODEL" -maxdepth 3 -type f -printf '%s %p\n' | sort -n
STATS="$(find "$FWMODEL" -maxdepth 3 -type f \( -iname '*stat*.json' -o -iname '*dataset*.json' \) | head -n 1)"
printf 'stats_path=%s\n' "$STATS"
if [ -n "$STATS" ]; then
  python3 - "$STATS" <<'PY'
import json, sys
p = sys.argv[1]
with open(p, encoding='utf-8') as f:
    x = json.load(f)
print('stats_type=', type(x).__name__)
if isinstance(x, dict):
    print('stats_top_level_count=', len(x))
    print('stats_top_level_keys=', sorted(map(str, x.keys())))
    for k, v in x.items():
        if isinstance(v, dict):
            print('stats_child', k, 'count=', len(v), 'keys=', sorted(map(str, v.keys()))[:200])
PY
fi

echo '=== TASK_CONFIG_TREE ==='
for root in "$FWWT/third_party/RoboTwin/task_config" "$FWCAN/third_party/RoboTwin/task_config"; do
  printf '\nTASK_CONFIG_ROOT=%s\n' "$root"
  if [ -d "$root" ]; then
    find -L "$root" -maxdepth 3 -type f -printf '%P\n' | sort
    find -L "$root" -maxdepth 3 -type f \( -name '*.yaml' -o -name '*.yml' \) -print0 \
      | xargs -0 grep -nHE '^(episode_num|step_lim|save_freq|data_type|render_freq|randomized|camera):' 2>/dev/null || true
  fi
done

echo '=== ROBOTWIN_TASK_MODULE_NAMES ==='
ENVROOT="$FWWT/third_party/RoboTwin/envs"
if [ -d "$ENVROOT" ]; then
  find "$ENVROOT" -maxdepth 2 -type f -name '*.py' -printf '%f\n' \
    | sed 's/\.py$//' | grep -vE '^(__init__|_base_task|utils)$' | sort -u
fi

echo 'SZ_DVAC_ROLLOUT_DESIGN_READONLY_INVENTORY_DONE'
