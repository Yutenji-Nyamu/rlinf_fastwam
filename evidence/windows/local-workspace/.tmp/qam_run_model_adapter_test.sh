set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv

cd "$repo"
test "$(git rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git branch --show-current)" = codex/qam-pi0-robotwin
export PYTHONPATH="$repo:/root/autodl-tmp/RoboTwin:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=
"$venv/bin/python" -m pytest -q \
  tests/embodiment/test_qam_openpi_adapter.py
