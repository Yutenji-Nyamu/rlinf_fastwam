#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
EXPECTED_HEAD=4ac48d54c63b3a83d99f551fb54f738297525acf
expected=(
  HANDOFF.md
  docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md
  docs/rlinf-robotwin-pi0-rltoken/02_STAGE1_SMOKE_AND_METHOD_ALIGNMENT_20260729.md
  docs/rlinf-robotwin-pi0-rltoken/03_STAGE1_FORMAL_TRAINING_20260729.md
  docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
  docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_formal_20260729/README.md
  docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_formal_20260729/dataset_manifest.json
  docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_formal_20260729/source_config.yaml
  docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_formal_20260729/formal_resolved.yaml
  docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_formal_20260729/exact_command.txt
  docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_formal_20260729/prelaunch_provenance.tsv
  docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_formal_20260729/run_provenance.tsv
  docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_formal_20260729/early_health.json
  docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_formal_20260729/early_health.json.sha256
)

test "$(git -C "$ROOT" branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git -C "$ROOT" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$ROOT" rev-list --left-right --count HEAD...@{upstream})" = $'0\t0'
expected_status_list=$(printf '%s\n' "${expected[@]}" | grep -v '/exact_command\.txt$' | sort)
expected_staged_list=$(printf '%s\n' "${expected[@]}" | sort)
actual_list=$(git -C "$ROOT" status --porcelain --untracked-files=all | cut -c4- | sort)
test "$actual_list" = "$expected_status_list"
git -C "$ROOT" add -f -- "${expected[@]}"
test "$(git -C "$ROOT" diff --cached --name-only | sort)" = "$expected_staged_list"
git -C "$ROOT" diff --cached --check
git -C "$ROOT" commit -m "docs(rlt): record formal Stage 1 launch"
git -C "$ROOT" push personal codex/rlt-pi0-robotwin
test -z "$(git -C "$ROOT" status --porcelain)"
test "$(git -C "$ROOT" rev-list --left-right --count HEAD...@{upstream})" = $'0\t0'
printf 'HEAD\t%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
git -C "$ROOT" ls-remote personal refs/heads/codex/rlt-pi0-robotwin
