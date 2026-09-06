set -u
date --iso-8601=seconds

echo '=== KNOWN ENV IMPORT VERSIONS ==='
for py in \
  /home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python \
  /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python \
  /data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official/.venvs/robotwin/bin/python; do
  if [ -x "$py" ]; then
    printf '%s\t' "$py"
    "$py" - <<'PY' 2>/dev/null || true
import importlib.metadata as m
for name in ('torch', 'nvidia-curobo', 'curobo'):
    try:
        print(f'{name}={m.version(name)}', end=' ')
    except m.PackageNotFoundError:
        pass
try:
    import torch
    print(f'torch_runtime={torch.__version__} cuda={torch.version.cuda}', end=' ')
except Exception as exc:
    print(f'torch_import_error={type(exc).__name__}', end=' ')
print()
PY
  else
    printf 'ABSENT_OR_NOT_EXECUTABLE %s\n' "$py"
  fi
done

echo '=== UV CACHE EXACT METADATA ==='
for cache in /home/chenyiteng/.cache/uv /data/chenyiteng/cache/uv-fasterwam; do
  [ -d "$cache" ] || continue
  echo "CACHE $cache"
  find "$cache" -xdev -maxdepth 5 -type d \
    \( -name 'torch-2.4.1+cu121.dist-info' -o -name 'torch-2.4.1.dist-info' \
       -o -name 'nvidia_curobo-0.7.8.dist-info' -o -name 'curobo-0.7.8.dist-info' \) \
    -print 2>/dev/null | sed -n '1,120p'
done

echo '=== CURRENT DOWNLOAD INSTALL ==='
ps -p 46226,46238,46247,79697 -o user=,pid=,ppid=,etime=,%cpu=,%mem=,rss=,stat=,cmd= 2>/dev/null || true
find /data/chenyiteng/models/fasterwam/release-6bf9471 -maxdepth 5 -type f \
  \( -name '*.incomplete' -o -name 'step_029355.pt' -o -name 'dataset_stats.json' \) \
  -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' 2>/dev/null | sort -k3
du -sh /data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official/.venvs/robotwin \
  /data/chenyiteng/cache/uv-fasterwam 2>/dev/null || true
nvidia-smi -i 3 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
