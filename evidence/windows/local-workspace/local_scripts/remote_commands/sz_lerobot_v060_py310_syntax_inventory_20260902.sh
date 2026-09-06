set -euo pipefail
src=/data/chenyiteng/projects/lerobot-sidney/lerobot-30da8e687a6d
echo '=== PEP695 class/def/type ==='
grep -R -nE '^(class|def|type) [A-Za-z_][A-Za-z0-9_]*\[' "$src/src/lerobot" || true
grep -R -nE '^type [A-Za-z_][A-Za-z0-9_]* *=' "$src/src/lerobot" || true
echo '=== Self imports ==='
grep -R -nE 'from typing import .*Self|typing\.Self' "$src/src/lerobot" || true
echo '=== pipeline imports ==='
sed -n '1,75p' "$src/src/lerobot/processor/pipeline.py"
sed -n '235,275p' "$src/src/lerobot/processor/pipeline.py"
echo '=== eval imports ==='
sed -n '1,90p' "$src/src/lerobot/scripts/lerobot_eval.py"
echo '=== compileall count original ==='
set +e
/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python -m compileall -q "$src/src/lerobot" 2>&1 | head -200
echo "COMPILE_RC=${PIPESTATUS[0]}"
set -e
