set -u
date --iso-8601=seconds

repo='/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official'

echo '=== UV CACHE ROOTS ==='
for p in \
  /home/chenyiteng/.cache/uv \
  /data/chenyiteng/cache/uv-fasterwam \
  /data/chenyiteng/cache/uv \
  /data/chenyiteng/cache; do
  if [ -e "$p" ]; then
    du -sh "$p" 2>/dev/null || true
    stat -c '%F %y %n' "$p" 2>/dev/null || true
  else
    printf 'ABSENT %s\n' "$p"
  fi
done

echo '=== LOCK EXPECTATIONS ==='
if [ -f "$repo/environments/robotwin/uv.lock" ]; then
  grep -n -A8 -B2 -E '^name = "(torch|nvidia-cuda-runtime-cu12|curobo|nvidia-cuda-nvrtc-cu12)"' \
    "$repo/environments/robotwin/uv.lock" | sed -n '1,180p'
fi
grep -RIn --include='pyproject.toml' --include='uv.lock' --include='requirements*.txt' \
  -E 'curobo|torch==2\.4\.1|torch =.*2\.4\.1' \
  "$repo/environments/robotwin" 2>/dev/null | sed -n '1,180p'

echo '=== CANDIDATE ENV METADATA ==='
for base in \
  /home/chenyiteng/venvs \
  /home/chenyiteng/.venvs \
  /data/chenyiteng/venvs \
  /data/chenyiteng/projects/fasterwam-standalone \
  /data/chenyiteng/projects; do
  [ -d "$base" ] || continue
  find "$base" -xdev -maxdepth 6 -type d \
    \( -name 'torch-2.4.1.dist-info' -o -name 'torch-2.4.1+cu121.dist-info' \
       -o -name 'curobo-0.7.8.dist-info' -o -name 'nvidia_curobo-0.7.8.dist-info' \) \
    -print 2>/dev/null
done | sort -u

echo '=== PYTHON ENVS WITH TORCH/CUROBO METADATA ==='
for base in /home/chenyiteng/venvs /home/chenyiteng/.venvs /data/chenyiteng/venvs /data/chenyiteng/projects/fasterwam-standalone; do
  [ -d "$base" ] || continue
  find "$base" -xdev -maxdepth 5 -type f -path '*/bin/python*' -perm -u+x -print 2>/dev/null
done | sort -u | while IFS= read -r py; do
  envroot=${py%/bin/python*}
  [ -d "$envroot" ] || continue
  meta=$(find "$envroot" -xdev -maxdepth 5 -type d \
    \( -name 'torch-*.dist-info' -o -name 'curobo-*.dist-info' -o -name 'nvidia_curobo-*.dist-info' \) \
    -printf '%f ' 2>/dev/null | head -c 500)
  [ -n "$meta" ] && printf '%s\t%s\n' "$envroot" "$meta"
done

echo '=== CURRENT OPS ==='
ps -eo user,pid,ppid,etime,%cpu,%mem,rss,stat,cmd \
  | grep -Ei '[s]napshot_download|release-6bf9471|[u]v sync.*environments/robotwin|[i]nstall_robotwin' \
  | sed -n '1,80p'
find /data/chenyiteng/models/fasterwam/release-6bf9471 -maxdepth 5 -type f \
  \( -name '*.incomplete' -o -name 'step_029355.pt' -o -name 'dataset_stats.json' \) \
  -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' 2>/dev/null | sort -k3
du -sh "$repo/.venvs/robotwin" /data/chenyiteng/cache/uv-fasterwam 2>/dev/null || true
nvidia-smi -i 3 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
