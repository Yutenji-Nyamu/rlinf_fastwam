#!/usr/bin/env bash
set -euo pipefail
source_dir=/root/autodl-tmp/experiment_exports/rlt_step475_c10_episode_probe_light_20260828_v1
archive=/root/autodl-tmp/experiment_exports/rlt_step475_c10_episode_probe_light_20260828_v1.tar.gz
tar -C "$source_dir" -czf "$archive" .
du -h "$archive"
sha256sum "$archive"
