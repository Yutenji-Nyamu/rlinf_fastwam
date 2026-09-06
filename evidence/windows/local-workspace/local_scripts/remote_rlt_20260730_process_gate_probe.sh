#!/usr/bin/env bash
set -u

printf 'self=%s parent=%s\n' "$$" "$PPID"
printf 'train_matches_raw_begin\n'
pgrep -af '[t]rain_embodied_agent.py' || true
printf 'train_matches_raw_end\n'
mapfile -t process_rows < <(
  {
    pgrep -af '[t]rain_embodied_agent.py' \
      | awk -v self="$$" '$1 != self' || true
    pgrep -ax raylet || true
    pgrep -ax gcs_server || true
  }
)
printf 'filtered_count=%s\n' "${#process_rows[@]}"
printf 'filtered_begin\n%s\nfiltered_end\n' "${process_rows[*]-}"
mapfile -t compute_rows < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk 'NF'
)
printf 'compute_count=%s\n' "${#compute_rows[@]}"
