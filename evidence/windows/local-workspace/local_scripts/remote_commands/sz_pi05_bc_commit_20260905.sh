set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc
sidney=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
cd "$root"
test "$(git rev-parse HEAD)" = 2467d997831166b70444b0c99d5198a2d3dfc8f6
test "$(git branch --show-current)" = codex/sz-pi05-online-bc
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
v=json.loads(Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/implementation-pi05-20260905/validation.json').read_text())
assert v['passed'] and v['identical_fixed32'] and v['stored_tokens_preserved']
PY
cmp rlinf/models/embodiment/openpi/dataconfig/__init__.py "$sidney/rlinf/models/embodiment/openpi/dataconfig/__init__.py"
cmp rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py "$sidney/rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py"
git add -f -- rlinf/models/embodiment/openpi/dataconfig/__init__.py rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py examples/embodiment/config/online_bc_model/pi05_sidney.yaml tests/unit_tests/test_pi05_online_bc.py rlinf/envs/robotwin/seeds/eval_sidney_fixed32.json
git diff --cached --check
git diff --cached --stat
git commit -m 'Add source-locked Sidney pi05 adapter to online success BC'
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy timeout 90s git -c http.proxy= -c https.proxy= push personal HEAD:refs/heads/codex/sz-pi05-online-bc
git rev-parse HEAD
git status --porcelain
