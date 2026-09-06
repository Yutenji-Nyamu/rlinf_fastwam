#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
RUN_ID=fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1
PAYLOAD=/data/chenyiteng/results/dvac-observation/$RUN_ID
META=/data/chenyiteng/results/dvac-observation/run-metadata/$RUN_ID
OFFICIAL="$WT/evaluate_results/robotwin/robotwin_uncond_3cam_384/$RUN_ID"

printf 'TIME=%s\n' "$(date --iso-8601=seconds)"
printf '%s\n' 'EPISODES_CSV_BEGIN'
sed -n '1,17p' "$PAYLOAD/episodes.csv"
printf '%s\n' 'EPISODES_CSV_END'
printf '%s\n' 'ACCEPTED_SEEDS_BEGIN'
grep -oE 'current seed:[[:space:]]*[^[:space:]]+' "$META/driver.log" || true
printf '%s\n' 'ACCEPTED_SEEDS_END'
printf '%s\n' 'VIDEOS_BEGIN'
find "$OFFICIAL" -type f -name '*.mp4' -size +1024c -printf '%f\t%s\n' | sort
printf '%s\n' 'VIDEOS_END'
