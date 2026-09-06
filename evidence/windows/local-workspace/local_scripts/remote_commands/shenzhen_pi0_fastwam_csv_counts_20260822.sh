#!/usr/bin/env bash
set -uo pipefail

pi0='/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-p1-16ep-800baf80-v1'
fastwam='/data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2'
fastwam_official='/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2'

date --iso-8601=seconds
hostname
id

printf 'pi0_csvs_and_media\n'
find "$pi0" -type f \( -name '*.csv' -o -name '*.mp4' -o -name '*.npz' \) -printf '%s %p\n' 2>/dev/null | sort
for csv in $(find "$pi0" -type f -name '*.csv' 2>/dev/null | sort); do
  printf 'csv=%s lines=%s\n' "$csv" "$(wc -l < "$csv")"
  head -n 2 "$csv"
  tail -n 3 "$csv"
done

printf 'fastwam_csvs_and_media\n'
find "$fastwam" "$fastwam_official" -type f \( -name '*.csv' -o -name '*.mp4' -o -name '*.npz' \) -printf '%s %p\n' 2>/dev/null | sort
for csv in $(find "$fastwam" -type f -name '*.csv' 2>/dev/null | sort); do
  printf 'csv=%s lines=%s\n' "$csv" "$(wc -l < "$csv")"
  head -n 2 "$csv"
  tail -n 4 "$csv"
done

printf 'csv_count_probe_done\n'
