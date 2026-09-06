#!/usr/bin/env bash
set -euo pipefail

ROBOTWIN=/data/chenyiteng/projects/robotwin-native/RoboTwin
RUN=/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1
RESULT="$ROBOTWIN/eval_result/adjust_bottle/ACT/demo_clean/demo_clean-50/2026-08-21_12-01-48_486194"
VIDEO="$RESULT/episode0.mp4"
SUMMARY="$RUN/08_act_hf_eval_1ep_scheduler/2026-08-21_12-01-38_159325/summary.json"

test -s "$RESULT/_result.txt"
test -s "$VIDEO"
test -s "$SUMMARY"
printf '%s\n' '=== result ==='
cat "$RESULT/_result.txt"
printf '%s\n' '=== video ==='
stat --printf='path=%n\nsize=%s\n' "$VIDEO"
sha256sum "$VIDEO"
ffprobe -v error -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames -show_entries format=duration,size -of default=noprint_wrappers=1 "$VIDEO"
printf '%s\n' '=== scheduler summary ==='
cat "$SUMMARY"
printf '\n%s\n' '=== final files ==='
find "$RESULT" -maxdepth 1 -type f -printf '%s\t%f\n' | sort
