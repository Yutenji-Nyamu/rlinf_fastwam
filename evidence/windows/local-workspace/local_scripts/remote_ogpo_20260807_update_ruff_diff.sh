#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/RLinf_ogpo_pi0_robotwin
/root/autodl-tmp/RLinf/.venv/bin/python -m ruff check --no-cache --fix --diff \
  tests/embodiment/ogpo_real_fsdp_update_probe.py
