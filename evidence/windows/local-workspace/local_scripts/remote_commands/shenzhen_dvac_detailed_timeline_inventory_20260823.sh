#!/usr/bin/env bash
set -euo pipefail

date -Is
hostname
id

roots=(
  /data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1/dvac_telemetry
  /data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2
  /data/chenyiteng/results/dvac-observation/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1
  /data/chenyiteng/results/dvac-observation/fastwam-turn_switch-p2-16ep-c63dc9b5-v1
  /data/chenyiteng/results/dvac-observation/fastwam-pick_diverse_bottles-p2-16ep-c63dc9b5-v1
)

for root in "${roots[@]}"; do
  test -d "$root"
  printf 'ROOT\t%s\t' "$root"
  du -sh "$root" | awk '{print $1}'
  printf 'FILES\t%s\t' "$root"
  find "$root" -type f | wc -l
done

videos=(
  /data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1/dvac_telemetry/seed_2/0.mp4
  /data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1/dvac_telemetry/seed_3/0.mp4
  /data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2/adjust_bottle/episode1_randomized-false_success-true.mp4
  /data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1/move_stapler_pad/episode3_randomized-false_success-true.mp4
  /data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1/move_stapler_pad/episode5_randomized-false_success-false.mp4
  /data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-turn_switch-p2-16ep-c63dc9b5-v1/turn_switch/episode0_randomized-false_success-false.mp4
  /data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-turn_switch-p2-16ep-c63dc9b5-v1/turn_switch/episode8_randomized-false_success-true.mp4
  /data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-pick_diverse_bottles-p2-16ep-c63dc9b5-v1/pick_diverse_bottles/episode3_randomized-false_success-true.mp4
  /data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-pick_diverse_bottles-p2-16ep-c63dc9b5-v1/pick_diverse_bottles/episode4_randomized-false_success-false.mp4
)

total=0
for video in "${videos[@]}"; do
  if [[ ! -f "$video" ]]; then
    printf 'VIDEO_MISSING\t%s\n' "$video"
    continue
  fi
  size=$(stat -c '%s' "$video")
  total=$((total + size))
  printf 'VIDEO\t%s\t%s\n' "$size" "$video"
done
printf 'VIDEO_TOTAL_BYTES\t%s\n' "$total"

image_total=0
while IFS= read -r image; do
  size=$(stat -c '%s' "$image")
  image_total=$((image_total + size))
  printf 'PI0_QUERY_IMAGE\t%s\t%s\n' "$size" "$image"
done < <(
  find /data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1/dvac_telemetry/query_images/rollout_rank02 \
    -maxdepth 1 -type f -name 'ep000033_q*_reset100100005__*.png' -print
  find /data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1/dvac_telemetry/query_images/rollout_rank03 \
    -maxdepth 1 -type f -name 'ep000050_q*_reset100100017__*.png' -print
)
printf 'PI0_QUERY_IMAGE_TOTAL_BYTES\t%s\n' "$image_total"

for manifest in \
  /data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1/dvac_telemetry/run_manifest.json \
  /data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2/run_manifest.json \
  /data/chenyiteng/results/dvac-observation/fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1/run_manifest.json \
  /data/chenyiteng/results/dvac-observation/fastwam-turn_switch-p2-16ep-c63dc9b5-v1/run_manifest.json \
  /data/chenyiteng/results/dvac-observation/fastwam-pick_diverse_bottles-p2-16ep-c63dc9b5-v1/run_manifest.json
do
  if [[ ! -f "$manifest" ]]; then
    printf 'MANIFEST_MISSING\t%s\n' "$manifest"
    continue
  fi
  python3 -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); keys=("schema_version","run_id","policy","task","denoising_steps","action_horizon","execution_horizon","action_num_inference_steps","skip_get_obs_within_replan"); print("MANIFEST",p,{k:d.get(k) for k in keys},sep="\t")' "$manifest"
done

printf 'GRPO_PROCESS_PROBE\n'
ps -eo pid,user,etimes,cmd --sort=pid | grep -E 'grpo-pi0-robotwin-formal100|ray::EnvWorker|ray::ActorWorker' | grep -v grep | head -n 30 || true
