#!/usr/bin/env bash
set -euo pipefail
venv=/root/autodl-tmp/RLinf/.venv
echo '[RAY_VERSION]'
"$venv/bin/python" -c 'import ray; print(ray.__version__, ray.__commit__)'
echo '[RAY_START_HELP]'
"$venv/bin/ray" start --help
echo '[CURRENT_RAYLET_ARGS]'
ps -eo pid,args | grep '[r]aylet'
echo '[SHM_AND_CPU]'
df -h /dev/shm
nproc
