set -euo pipefail
cd /data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
grep -R -n -E "rollout\.model\.model_type|model_type.*rollout|def validate_cfg" rlinf examples | head -120 || true
grep -R -n -F 'model: ${actor.model}' examples | head -80 || true
grep -R -n -F 'model_type: ${actor.model.model_type}' examples | head -80 || true
find examples -path '*model*pi0_5*.yaml' -o -path '*model*pi05*.yaml'
find . -path '*config/model/pi0_5.yaml' -print
sed -n '1,220p' examples/embodiment/config/model/pi0_5.yaml
sed -n '1455,1540p' rlinf/config.py
sed -n '840,1040p' rlinf/config.py
