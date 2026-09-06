set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
LR=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
cd "$WT"
echo '=== openpi get_model ==='
sed -n '1,180p' rlinf/models/embodiment/openpi/__init__.py
echo '=== aloha wrappers ==='
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import inspect
import openpi.transforms as transforms
import openpi.models.model as model
from openpi.policies import aloha_policy
print(inspect.getsource(aloha_policy.AlohaInputs))
print(inspect.getsource(aloha_policy.AlohaOutputs))
print(inspect.getsource(model.Observation))
PY
echo '=== model wrapper setup ==='
grep -n "def setup_wrappers\|def predict_action_batch\|def sample_actions" rlinf/models/embodiment/openpi/openpi_action_model.py
sed -n '420,610p' rlinf/models/embodiment/openpi/openpi_action_model.py
sed -n '940,1110p' rlinf/models/embodiment/openpi/openpi_action_model.py
echo '=== LeRobot processor factory/use ==='
cd "$LR"
find . -maxdepth 2 -type d | head -40
grep -R -n -E "def make_pre_post_processors|postprocessor\(|predict_action_chunk|class PI05Policy" src/lerobot tests | head -100 || true
sed -n '350,450p' src/lerobot/policies/factory.py
sed -n '1,240p' tests/policies/pi0_pi05/test_pi05_original_vs_lerobot.py
