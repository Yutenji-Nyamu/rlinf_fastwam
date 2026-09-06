set -eu
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
for f in \
  "$WT/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_pi05.yaml" \
  "$WT/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml" \
  "$WT/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml"; do
  echo "=== $f ==="
  nl -ba "$f" | sed -n '1,38p'
  nl -ba "$f" | sed -n '115,185p'
done
