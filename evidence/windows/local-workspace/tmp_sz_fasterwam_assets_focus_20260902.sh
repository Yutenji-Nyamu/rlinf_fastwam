set -u
date --iso-8601=seconds

echo '=== FASTWAM_SOURCE ==='
repo='/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711'
if [ -d "$repo/.git" ]; then
  git -C "$repo" rev-parse HEAD
  git -C "$repo" remote -v
  git -C "$repo" status --short --branch | sed -n '1,80p'
fi

echo '=== FASTWAM_ENV ==='
envdir='/home/chenyiteng/venvs/fastwam-7faa-py310-cu128'
ls -la "$envdir" | sed -n '1,100p'
ls -l "$envdir/bin"/{python,python3,python3.10,pip,pip3} 2>/dev/null || true
for exe in "$envdir/bin/python" "$envdir/bin/python3" "$envdir/bin/python3.10"; do
  if [ -x "$exe" ]; then
    printf '%s\t' "$exe"; "$exe" -V 2>&1
    "$exe" -c 'import importlib.util as u; print("torch",bool(u.find_spec("torch")),"diffsynth",bool(u.find_spec("diffsynth")),"huggingface_hub",bool(u.find_spec("huggingface_hub")),"modelscope",bool(u.find_spec("modelscope")),"fastwam",bool(u.find_spec("fastwam")))' 2>/dev/null || true
    "$exe" -c 'import torch; print("torch_version",torch.__version__,"cuda",torch.version.cuda)' 2>/dev/null || true
    break
  fi
done

echo '=== CONDA ==='
/home/chenyiteng/miniforge3/bin/conda env list 2>/dev/null || true
du -sh /home/chenyiteng/miniforge3 /home/chenyiteng/miniforge3/envs/* 2>/dev/null | sort -h

echo '=== MODEL_TREE ==='
modelroot='/data/chenyiteng/models/fastwam'
find "$modelroot" -maxdepth 6 -type f -printf '%s\t%p\n' 2>/dev/null | sort -nr | sed -n '1,220p'
echo '--- model dirs sizes ---'
du -sh "$modelroot"/* "$modelroot"/diffsynth/* "$modelroot"/diffsynth/DiffSynth-Studio/* 2>/dev/null | sort -h

echo '=== MODELSCOPE_CACHE ==='
ms='/home/chenyiteng/cache/fastwam-7faa/modelscope'
du -sh "$ms" 2>/dev/null || true
find "$ms" -maxdepth 6 -type f -printf '%s\t%p\n' 2>/dev/null | sort -nr | sed -n '1,180p'

echo '=== HF_CACHE ==='
du -sh /home/chenyiteng/.cache/huggingface 2>/dev/null || true
find /home/chenyiteng/.cache/huggingface -maxdepth 5 -type d -name 'models--*' -printf '%p\n' 2>/dev/null | sort | sed -n '1,160p'

echo '=== GPU0_3_FREE ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed -n '1,4p'

