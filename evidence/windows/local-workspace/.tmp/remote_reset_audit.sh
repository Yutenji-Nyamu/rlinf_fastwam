set -eu
cd /root/autodl-tmp/RLinf_fastwam_rlinf
find . -type f \( -name '*.py' -o -name '*.yaml' \) -print0 | xargs -0 grep -nH -E 'use_fixed_reset_state_ids|_init_reset_state_ids|reset_state_ids' || true
find /root/autodl-tmp/RoboTwin_RLinf -type f -name '*.py' -print0 | xargs -0 grep -nH -E 'use_fixed_reset_state_ids|_init_reset_state_ids|reset_state_ids' || true
/root/autodl-tmp/conda/envs/FastWAM-RLinf/bin/python - <<'PY'
import inspect
from rlinf.envs import create_env
print(inspect.getsourcefile(create_env))
print(inspect.getsource(create_env))
PY
