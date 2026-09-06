set -euo pipefail
src=/data/chenyiteng/projects/lerobot-sidney/lerobot-30da8e687a6d
mkdir -p /data/chenyiteng/projects/lerobot-sidney
if [ ! -d "$src/.git" ]; then
  git clone -q https://github.com/huggingface/lerobot.git "$src"
fi
git -C "$src" fetch -q --tags origin
git -C "$src" checkout -q --detach 30da8e687a6dfc617fcd94afc367ac7071c376ce
test -z "$(git -C "$src" status --porcelain)"
echo "SRC=$(git -C "$src" rev-parse HEAD)"
py=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
echo '=== compile pi05/processors/eval/env only under py310 ==='
set +e
"$py" -m compileall -q \
  "$src/src/lerobot/policies/pi05" \
  "$src/src/lerobot/processor" \
  "$src/src/lerobot/envs/robotwin.py" \
  "$src/src/lerobot/scripts/lerobot_eval.py"
rc=$?
set -e
echo "COMPILE_RC=$rc"
echo '=== import with PYTHONPATH, existing deps ==='
set +e
PYTHONPATH="$src/src" "$py" - <<'PY'
import sys
mods=['lerobot','lerobot.policies.pi05','lerobot.envs.robotwin','lerobot.scripts.lerobot_eval']
for m in mods:
 try:
  __import__(m); print('OK',m)
 except Exception as e:
  print('ERR',m,type(e).__name__,str(e))
PY
set -e
echo '=== local dep versions ==='
"$py" - <<'PY'
import importlib.metadata as m
for x in ['torch','torchvision','numpy','opencv-python-headless','Pillow','einops','draccus','huggingface-hub','requests','gymnasium','safetensors','transformers','scipy','av']:
 try: print(x,m.version(x))
 except Exception: print(x,'MISSING')
PY
