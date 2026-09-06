set -euo pipefail

target=/root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin
old_head=73da63f01d52290c3537f12fb4bfb55cfd82f94c
branch=codex/idea2-dvac-pi0-robotwin

cd "$target"
test "$(git rev-parse HEAD)" = "$old_head"
test "$(git branch --show-current)" = "$branch"
test "$(git diff --name-only --diff-filter=D | wc -l)" -eq 0

expected_paths=$(printf '%s\n' \
  evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  tests/unit_tests/test_dvac_telemetry.py | sort)
actual_paths=$(git status --short | sed 's/^...//' | sort)
test "$actual_paths" = "$expected_paths"

git diff --check
git add -- \
  evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  tests/unit_tests/test_dvac_telemetry.py
test "$(git diff --cached --name-only | sort)" = "$expected_paths"
git diff --cached --check
git commit -m "fix: complete DVAC telemetry run provenance"

new_head=$(git rev-parse HEAD)
test "$new_head" != "$old_head"
test -z "$(git status --porcelain)"
printf 'OLD_HEAD=%s\nNEW_HEAD=%s\nBRANCH=%s\nPOST_COMMIT_CLEAN=1\n' \
  "$old_head" "$new_head" "$branch"

