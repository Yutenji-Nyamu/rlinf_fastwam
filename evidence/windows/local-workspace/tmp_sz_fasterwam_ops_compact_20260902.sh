set -u
date --iso-8601=seconds
echo '--- proc ---'
ps -p 46220,46226,46237,46238,46247 -o user=,pid=,ppid=,etime=,%cpu=,%mem=,rss=,stat=,cmd= 2>/dev/null || true
ps -eo user,pid,ppid,etime,%cpu,%mem,rss,stat,cmd \
  | awk '$3==46226 || $3==46238 || $3==46237 || $3==46247 {print}' | sed -n '1,80p'

echo '--- download ---'
target='/data/chenyiteng/models/fasterwam/release-6bf9471'
du -sh "$target" 2>/dev/null || true
find "$target" -maxdepth 5 -type f \( -name '*.incomplete' -o -name '*.lock' -o -name 'step_029355.pt' -o -name 'dataset_stats.json' \) -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' 2>/dev/null | sort -k3

echo '--- install ---'
repo='/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official'
du -sh "$repo/.venvs/robotwin" /data/chenyiteng/cache/uv-fasterwam 2>/dev/null || true
if [ -x "$repo/.venvs/robotwin/bin/python" ]; then
  "$repo/.venvs/robotwin/bin/python" -V 2>&1 || true
  "$repo/.venvs/robotwin/bin/python" -c 'import importlib.util as u; print("torch",bool(u.find_spec("torch")),"hydra",bool(u.find_spec("hydra")),"sapien",bool(u.find_spec("sapien")),"robotwin",bool(u.find_spec("robotwin")))' 2>/dev/null || true
fi
find /data/chenyiteng/results/fasterwam-standalone -maxdepth 4 -type f -printf '%T@\t%s\t%p\n' 2>/dev/null | sort -nr | head -n 30

echo '--- gpu3 ---'
nvidia-smi -i 3 --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
nvidia-smi -i 3 --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true

