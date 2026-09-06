set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
git -C "$WT" reset --quiet
sed -i '${/^$/d;}' "$WT/examples/embodiment/config/env/robotwin_move_stapler_pad.yaml"
sed -i '${/^$/d;}' "$WT/examples/embodiment/config/model/fastwam_robotwin.yaml"
git -C "$WT" diff --check
