#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
cd "$WT"
test "$(git rev-parse HEAD)" = 7faa71108368fbb3b6885649f112af607427a2d4
test "$(git branch --show-current)" = codex/sz-fastwam-dvac-observe
test -z "$(git status --porcelain --untracked-files=all)"
echo '154c339efd33fae119fe902d6efb91cde7475e3cfd0e7b50e606278c1c3eac09  src/fastwam/models/wan22/fastwam.py' | sha256sum -c -
echo '57a34b527b6bc446a6f9e2e4a1b7098a449b7fc09275b20ee7e446e6ef0656ac  experiments/robotwin/fastwam_policy/deploy_policy.py' | sha256sum -c -
echo 'b96b579a8f243c935f1e17f8f3d6e9b7cb6bf2f2a6823272f626ecc33d68056f  configs/sim_robotwin.yaml' | sha256sum -c -
test "$(git hash-object experiments/robotwin/eval_robotwin_single.py)" = "$(git rev-parse HEAD:experiments/robotwin/eval_robotwin_single.py)"
test "$(git hash-object experiments/robotwin/fastwam_policy/deploy_policy.yml)" = "$(git rev-parse HEAD:experiments/robotwin/fastwam_policy/deploy_policy.yml)"
test ! -e experiments/robotwin/fastwam_policy/dvac_telemetry.py
test ! -e tests/test_fastwam_dvac_telemetry.py
mkdir -p tests
echo FASTWAM_DVAC_UPLOAD_GUARD_OK
