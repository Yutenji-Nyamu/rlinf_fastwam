set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
cd "$WT"
sed -n '1,240p' examples/embodiment/config/robotwin_move_stapler_pad_grpo_openpi_pi05_sidney.yaml
grep -R -n "get_model(cfg.actor.model\|get_model(self.cfg.actor.model\|models.embodiment.openpi import get_model" rlinf tests | head -40 || true
