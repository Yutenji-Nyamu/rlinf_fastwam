set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
runtime=/root/autodl-tmp/RLinf/.venv/bin/python

cd "$target"
test "$(git diff --name-only --diff-filter=D | wc -l)" -eq 0
git diff --check
PYTHONDONTWRITEBYTECODE=1 "$runtime" -m ruff check \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/utils/dvac_telemetry.py \
  rlinf/workers/env/env_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  tests/unit_tests/test_dvac_telemetry.py
printf 'TARGETED_LINT_PASS=1\n'
