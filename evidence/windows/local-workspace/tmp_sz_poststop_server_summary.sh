set -u
echo '===TIME==='; TZ=Asia/Shanghai date --iso-8601=seconds
echo '===GPU==='; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
echo '===GPU_COMPUTE==='; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true
echo '===RAM==='; free -h; cat /proc/pressure/memory
echo '===DISK==='; df -hT / /home /data
echo '===NETWORK_SERVICE==='; systemctl is-active mihomo 2>&1 || true; systemctl show mihomo -p ActiveState -p SubState -p NRestarts 2>&1 || true
for u in https://github.com https://huggingface.co; do
  printf '%s proxy7890=' "$u"
  curl -x http://127.0.0.1:7890 -LIsS --max-time 12 -o /dev/null -w '%{http_code} %{time_total}\n' "$u" || true
done
echo '===OTHER_USERS_GPU_AND_TOP===' 
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
for pid in $gpu_pids; do ps -o user:20=,pid=,etime=,%cpu=,%mem=,rss=,args= -p "$pid"; done
ps -eo user:20=,pid=,etime=,%cpu=,%mem=,rss=,args= --sort=-rss | awk '$1 != "root" && $1 != "chenyiteng" {print}' | head -20
echo '===RAY==='; ps -o user=,pid=,etime=,stat=,args= -p 321933; /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/ray status --address=172.17.0.1:6389 | sed -n '1,24p'
