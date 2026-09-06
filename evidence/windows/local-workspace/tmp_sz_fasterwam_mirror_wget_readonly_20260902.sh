set -u
date --iso-8601=seconds
echo '=== WGET/MIRROR ==='
ps -eo user,pid,ppid,etime,%cpu,%mem,rss,stat,cmd \
  | grep -Ei '[w]get|[h]f-mirror|release-6bf9471|step_029355\.pt' \
  | sed -n '1,120p'
echo '=== EXACT FILES ==='
find /data/chenyiteng/models/fasterwam/release-6bf9471 -maxdepth 6 -type f \
  \( -name '*.incomplete' -o -name 'step_029355.pt' -o -name 'dataset_stats.json' \) \
  -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' 2>/dev/null | sort -k3
echo '=== INSTALL ==='
ps -p 46226,46238,46247 -o user=,pid=,ppid=,etime=,%cpu=,%mem=,rss=,stat=,cmd= 2>/dev/null || true
du -sh /data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official/.venvs/robotwin \
  /data/chenyiteng/cache/uv-fasterwam 2>/dev/null || true
nvidia-smi -i 3 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
