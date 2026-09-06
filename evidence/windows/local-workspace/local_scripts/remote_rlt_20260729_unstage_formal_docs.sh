#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
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

test "$(git -C "$ROOT" rev-parse HEAD)" = 4ac48d54c63b3a83d99f551fb54f738297525acf
git -C "$ROOT" restore --staged -- "${expected[@]}"
test -z "$(git -C "$ROOT" diff --cached --name-only)"
printf 'UNSTAGED_WITHOUT_WORKTREE_CHANGES\n'
