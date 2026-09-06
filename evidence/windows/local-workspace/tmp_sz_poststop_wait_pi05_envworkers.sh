set -u
echo "wait_started=$(TZ=Asia/Shanghai date --iso-8601=seconds)"
for _ in $(seq 1 60); do
  alive=0
  for pid in 1478462 1478464; do
    [ ! -d "/proc/$pid" ] || alive=$((alive + 1))
  done
  [ "$alive" -eq 0 ] && break
  sleep 2
done
echo '===TARGET_PIDS===' 
for pid in 1478462 1478464; do
  if [ -d "/proc/$pid" ]; then
    ps -o user=,pid=,ppid=,pgid=,sid=,etime=,stat=,wchan=,args= -p "$pid"
  else
    echo "pid=$pid gone"
  fi
done
echo '===GPU===' 
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo '===COMPUTE===' 
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true
echo '===WRAPPERS===' 
for pid in 826980 1477572; do kill -0 "$pid" 2>/dev/null && echo "$pid alive" || echo "$pid gone"; done
echo "wait_finished=$(TZ=Asia/Shanghai date --iso-8601=seconds)"
