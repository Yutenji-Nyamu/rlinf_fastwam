set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
LR=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
MODEL=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
cd "$WT"
echo '=== wrapper setup ==='
sed -n '280,350p' rlinf/models/embodiment/openpi/openpi_action_model.py
sed -n '800,930p' rlinf/models/embodiment/openpi/openpi_action_model.py
echo '=== obs processor ==='
grep -R -n "class .*Obs\|def obs_processor" rlinf/models/embodiment/openpi rlinf/models/embodiment/base_policy.py | head -60 || true
grep -R -n "main_images\|wrist_images\|extra_view_images" rlinf/models/embodiment/openpi | head -100 || true
echo '=== source config ==='
cat "$MODEL/config.json"
echo '=== native model sample API ==='
cd "$LR"
grep -n "def sample_actions" src/lerobot/policies/pi05/modeling_pi05.py | head
sed -n '760,860p' src/lerobot/policies/pi05/modeling_pi05.py
