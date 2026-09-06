#!/usr/bin/env bash
set -euo pipefail
venv=/root/autodl-tmp/RLinf/.venv
RAY_ADDRESS=auto "$venv/bin/ray" memory --stats-only 2>&1 | sed -n '1,220p'
