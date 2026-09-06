set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
LR=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
MODEL=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
RLPY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
LRPY=/home/chenyiteng/venvs/lerobot-v060-sidney-py310/bin/python
cd "$WT"
$RLPY -m ruff format toolkits/lerobot/sidney_pi05_parity.py
$RLPY -m ruff check toolkits/lerobot/sidney_pi05_parity.py
$RLPY -m py_compile toolkits/lerobot/sidney_pi05_parity.py
cd "$LR"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$LR/src" "$LRPY" - "$MODEL" <<'PY'
import sys
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

config = PreTrainedConfig.from_pretrained(sys.argv[1], local_files_only=True)
assert isinstance(config, PI05Config), type(config)
assert config.chunk_size == 50 and config.num_inference_steps == 10
print('native_config_parse_ok', type(config).__name__, config.chunk_size, config.num_inference_steps)
PY
cd "$WT"
git diff --check
git diff --stat
