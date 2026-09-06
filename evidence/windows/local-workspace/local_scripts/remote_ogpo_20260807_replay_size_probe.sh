#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo"

/root/autodl-tmp/RLinf/.venv/bin/python -
