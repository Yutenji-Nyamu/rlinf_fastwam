set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
patch=/root/autodl-tmp/qam_prefix_diagnostic_nonblocking_20260801.patch

test "$(git -C "$repo" branch --show-current)" = codex/qam-pi0-robotwin
test "$(git -C "$repo" rev-parse HEAD)" = dc3711950a2cad8222ba72bfe0c6de2f7a5babdb
test -z "$(git -C "$repo" status --short)"
git -C "$repo" apply --check "$patch"
git -C "$repo" apply "$patch"
git -C "$repo" diff --check
cd "$repo"
PYTHONPATH="$repo" /root/autodl-tmp/RLinf/.venv/bin/python -m pytest -q \
  tests/embodiment/test_robotwin_qam_contract.py \
  tests/workers/test_qam_worker_helpers.py \
  tests/embodiment/test_qam_openpi_adapter.py
git status --short
git diff --stat
