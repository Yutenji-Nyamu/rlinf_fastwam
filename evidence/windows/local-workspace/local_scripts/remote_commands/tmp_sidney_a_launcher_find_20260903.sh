set -euo pipefail
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes
find "$ROOT" -maxdepth 4 -type f \( -name 'launch.sh' -o -name 'start.sh' -o -name 'resolved*.yaml' \) -path '*v12*' -print
find "$ROOT" -maxdepth 4 -type f -path '*v12*' -name '*.sh' -print
find "$ROOT" -maxdepth 2 -mindepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -30
find "$ROOT/b1-adjust-badseed-retry-m10-phys4-v12" -maxdepth 3 -type f -printf '%p\n' | sort
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
from pathlib import Path
import yaml
p=Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/b1-adjust-badseed-retry-m10-phys4-v12/tensorboard/config.yaml')
c=yaml.safe_load(p.read_text())
print('cluster', c['cluster']['component_placement'])
print('runner', {k:c['runner'].get(k) for k in ('task_type','only_eval','max_steps','val_check_interval')})
print('rollout_model_keys', sorted(c['rollout']['model']))
PY
sed -n '1,120p' "$ROOT/b1-adjust-badseed-retry-m10-phys4-v12/runtime/driver.log"
