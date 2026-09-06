set -u

date --iso-8601=seconds
echo '=== PROCESSES ==='
ps -eo user,pid,ppid,lstart,etime,%cpu,%mem,rss,stat,cmd --sort=start_time \
  | grep -Ei '[F]asterWAM|[f]asterwam|[i]nstall_robotwin|[h]f download|[h]uggingface-cli|[s]napshot_download|[m]odelscope|[u]v (sync|run|pip)|[p]ip(3)? install' \
  | sed -n '1,180p' || true

echo '=== FASTER_PATHS ==='
for root in /data/chenyiteng/projects /data/chenyiteng/models /data/chenyiteng/results /home/chenyiteng; do
  [ -d "$root" ] || continue
  find "$root" -maxdepth 5 \( -iname '*faster*wam*' -o -iname '*FasterWAM*' \) -printf '%y\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' 2>/dev/null
done | sort -k4 | sed -n '1,320p'

echo '=== TARGET_SIZES ==='
for path in \
  /data/chenyiteng/projects/fasterwam-standalone \
  /data/chenyiteng/projects/FasterWAM \
  /data/chenyiteng/models/fasterwam \
  /data/chenyiteng/models/FasterWAM \
  /home/chenyiteng/.cache/huggingface \
  /home/chenyiteng/.cache/uv; do
  [ -e "$path" ] && du -sh "$path" 2>/dev/null || true
done

echo '=== RECENT_LARGE_FILES ==='
for root in /data/chenyiteng/projects/fasterwam-standalone /data/chenyiteng/projects/FasterWAM /data/chenyiteng/models/fasterwam /data/chenyiteng/models/FasterWAM; do
  [ -d "$root" ] || continue
  find "$root" -maxdepth 7 -type f -printf '%T@\t%s\t%p\n' 2>/dev/null | sort -nr | head -n 40
done

echo '=== VENV_AND_INSTALL_MARKERS ==='
for root in /data/chenyiteng/projects/fasterwam-standalone /data/chenyiteng/projects/FasterWAM; do
  [ -d "$root" ] || continue
  find "$root" -maxdepth 4 \( -type d -name '.venv' -o -type f -name 'pyvenv.cfg' -o -type f -name 'uv.lock' -o -type f -name '*install*.log' -o -type f -name '*download*.log' \) -printf '%y\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' 2>/dev/null | sort
done

echo '=== GPU3 ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits | sed -n '4p'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | while IFS= read -r line; do
  pid=$(printf '%s\n' "$line" | awk -F',' '{gsub(/ /,"",$2); print $2}')
  [ -n "$pid" ] && ps -p "$pid" -o user=,pid=,ppid=,etime=,%cpu=,%mem=,rss=,stat=,cmd= 2>/dev/null || true
done | sed -n '1,100p'

