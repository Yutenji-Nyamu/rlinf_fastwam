set -euo pipefail
cd /data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
git diff --check
GIT_PAGER=cat git diff -- examples/embodiment/config/robotwin_move_stapler_pad_grpo_openpi_pi05_sidney.yaml
git add examples/embodiment/config/robotwin_move_stapler_pad_grpo_openpi_pi05_sidney.yaml
git commit -m "config: share Sidney model contract with rollout"
git push personal codex/sz-sidney-pi05-current-rlinf
git rev-parse HEAD
git status --short
