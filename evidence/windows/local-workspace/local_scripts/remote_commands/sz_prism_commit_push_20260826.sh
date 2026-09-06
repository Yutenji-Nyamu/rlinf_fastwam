set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/prism-dvac-rank-rloo
BRANCH=codex/sz-prism-dvac-rank-rloo
EXPECTED=0e28ac6f09f821ea12e7d54eba7118ce0000ca86

test "$(git -C "$ROOT" rev-parse HEAD)" = "$EXPECTED"
test "$(git -C "$ROOT" branch --show-current)" = "$BRANCH"
cd "$ROOT"
git diff --check
git add -- \
  examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml \
  rlinf/algorithms/advantages.py \
  rlinf/algorithms/dvac_rank_reward.py \
  rlinf/config.py \
  rlinf/workers/actor/embodied_fsdp_actor_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  tests/unit_tests/test_prism_dvac_rank_rloo.py
git diff --cached --check
git commit -m "feat: add Prism-style DVAC rank RLOO"
git push personal "$BRANCH"
git rev-parse HEAD
git status --short
echo SZ_PRISM_COMMIT_PUSH_OK
