set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
cd "$WT"
sed -n '150,240p' rlinf/models/embodiment/openpi/policies/aloha_policy.py
grep -n "pi05_sidney_robotwin" -A55 -B10 rlinf/models/embodiment/openpi/dataconfig/__init__.py
sed -n '330,385p' rlinf/models/embodiment/openpi/openpi_action_model.py
