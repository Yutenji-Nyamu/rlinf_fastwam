set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
base=6d0db56bf26f972cd27fa29535f5eb939e80e5bf

cd "$target"
test "$(git rev-parse HEAD)" = "$base"
test "$(git branch --show-current)" = codex/idea2-dvac-pi0-robotwin
test "$(git diff --name-only --diff-filter=D | wc -l)" -eq 0
test "$(git diff --cached --name-only | wc -l)" -eq 0

git add -- \
  evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/utils/dvac_telemetry.py \
  rlinf/workers/env/env_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  tests/unit_tests/test_dvac_telemetry.py

expected_paths="$(printf '%s\n' \
  evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/utils/dvac_telemetry.py \
  rlinf/workers/env/env_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  tests/unit_tests/test_dvac_telemetry.py | sort | sha256sum | awk '{print $1}')"
actual_paths="$(git diff --cached --name-only | sort | sha256sum | awk '{print $1}')"
test "$actual_paths" = "$expected_paths"
git diff --cached --check
git diff --cached --stat
git commit -m "feat: add opt-in pi0 DVAC evaluation telemetry"
printf 'COMMIT=%s\n' "$(git rev-parse HEAD)"
printf 'BRANCH=%s\n' "$(git branch --show-current)"
git status --short
