#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
python=/root/autodl-tmp/RLinf/.venv/bin/python

printf 'NOW=%s\n' "$(date --iso-8601=seconds)"
printf 'memory.before='; cat /sys/fs/cgroup/memory.current
cd "$repo"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$repo" timeout 60 "$python" -m pytest -q \
  tests/unit_tests/test_rlt_dvac_weighting.py
printf 'memory.after='; cat /sys/fs/cgroup/memory.current
grep -E '^oom |^oom_kill ' /sys/fs/cgroup/memory.events
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
