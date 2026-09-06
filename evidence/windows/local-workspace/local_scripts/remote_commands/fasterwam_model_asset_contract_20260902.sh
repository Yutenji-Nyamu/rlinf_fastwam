set -euo pipefail

REPO=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
cd "$REPO"

printf 'MODEL_IMPORTS\n'
grep -RInE "DIFFSYNTH|model_id|tokenizer_model_id|Wan2.2|UMT5|VAE|safetensors|download" src/fasterwam | sed -n '1,320p'
printf 'ROBOTWIN_LOCK_RELEVANT\n'
grep -nEi "modelscope|diffsynth|huggingface|transformers|torch|tokenizer" environments/robotwin/uv.lock | sed -n '1,260p'
printf 'OLD_MODEL_TREE\n'
find /data/chenyiteng/models/fastwam -maxdepth 5 -type f -printf '%s %p\n' | sort -n | tail -80
printf 'OLD_VENV_RELEVANT\n'
/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python - <<'PY'
import importlib.metadata as m
for name in ['torch','torchvision','modelscope','huggingface-hub','transformers','diffsynth']:
    try:
        print(name, m.version(name))
    except Exception as exc:
        print(name, 'MISSING', type(exc).__name__)
PY
