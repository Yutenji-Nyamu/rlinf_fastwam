set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
FILE="$WT/rlinf/utils/runner_utils.py"
RUNNER="$WT/rlinf/runners/embodied_runner.py"
echo "HEAD=$(git -C "$WT" rev-parse HEAD)"
grep -n -A30 -B2 '^def check_progress' "$FILE"
grep -n -A24 -B4 '^    def _maybe_eval_and_checkpoint' "$RUNNER"
