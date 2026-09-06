#!/usr/bin/env bash
set -uo pipefail

wt='/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711'
canon='/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711'
env_dir='/home/chenyiteng/venvs/fastwam-7faa-py310-cu128'
ckpt='/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384.pt'
stats='/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384_dataset_stats.json'
meta='/data/chenyiteng/results/dvac-observation/run-metadata/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2'
payload='/data/chenyiteng/results/dvac-observation/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2'
official="$wt/evaluate_results/robotwin/robotwin_uncond_3cam_384/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2"

date --iso-8601=seconds
hostname
id

printf 'head=%s\n' "$(git -C "$wt" rev-parse HEAD 2>&1)"
printf 'status_begin\n'
git -C "$wt" status --porcelain
printf 'status_end\n'

for path in "$ckpt" "$stats" "$env_dir/bin/python"; do
  if [[ -s "$path" ]]; then printf 'nonempty=true path=%s\n' "$path"; else printf 'nonempty=false path=%s\n' "$path"; fi
done
for path in "$meta" "$payload" "$official"; do
  if [[ -e "$path" ]]; then printf 'exists=true path=%s\n' "$path"; else printf 'exists=false path=%s\n' "$path"; fi
done

for name in assets task_config; do
  link="$wt/third_party/RoboTwin/$name"
  target="$canon/third_party/RoboTwin/$name"
  if [[ -L "$link" ]]; then printf 'symlink=true path=%s resolved=%s\n' "$link" "$(readlink -f "$link")"; else printf 'symlink=false path=%s\n' "$link"; fi
  printf 'target_resolved=%s\n' "$(readlink -f "$target")"
done

printf 'gpu3_compute_pids\n'
nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader,nounits || true
printf 'preflight_predicates_complete\n'
