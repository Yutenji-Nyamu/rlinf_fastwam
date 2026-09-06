set -euo pipefail
src=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
py=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
echo '=== source deps ==='
sed -n '35,135p' "$src/pyproject.toml"
echo '=== base versions ==='
"$py" - <<'PY'
import importlib, importlib.metadata as m
for name in ['torch','torchvision','numpy','transformers','huggingface-hub','draccus','gymnasium','safetensors','diffusers','datasets','opencv-python-headless','scipy','termcolor','typing_extensions']:
 try: print(name, m.version(name))
 except Exception: print(name, 'MISSING')
PY
