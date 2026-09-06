set -u

PI_RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-grpo/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2
FW_RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2

echo '===IDENTITY_TIME===' 
date -Is
hostname
id

echo '===TARGET_PROCESSES===' 
ps -eo user=,pid=,ppid=,pgid=,sid=,lstart=,etime=,stat=,%cpu=,%mem=,rss=,args= --sort=pid | \
  grep -E 'pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2|fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2|start_pi05|start_fastwam|monitor_pi05|monitor_fastwam' | \
  grep -v grep || true

echo '===GPU===' 
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
echo '===GPU_COMPUTE===' 
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true

echo '===RAM_PSI===' 
free -h
awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo
cat /proc/pressure/memory

echo '===DISK===' 
df -hT / /home /data

echo '===NETWORK===' 
ip -brief address show | head -20
ss -s
for u in https://github.com https://huggingface.co; do
  printf '%s direct=' "$u"
  curl --noproxy '*' -LIsS --max-time 8 -o /dev/null -w '%{http_code} %{time_total}\n' "$u" || true
  printf '%s default=' "$u"
  curl -LIsS --max-time 8 -o /dev/null -w '%{http_code} %{time_total}\n' "$u" || true
done

echo '===USER_ACTIVITY===' 
ps -eo user=,pid=,etime=,%cpu=,%mem=,rss=,args= --sort=-rss | awk '$1 != "root" && $1 != "chenyiteng" {print}' | head -30

echo '===PI_RUN_FILES===' 
if [ -d "$PI_RUN" ]; then
  find "$PI_RUN" -maxdepth 3 -type f -printf '%T@\t%s\t%p\n' 2>/dev/null | sort -nr | head -40
  echo '---pi checkpoints---'
  find "$PI_RUN" -maxdepth 4 -type d -name 'global_step_*' -printf '%T@\t%p\n' 2>/dev/null | sort -n | tail -20
  echo '---pi latest metric lines---'
  grep -RhaE 'Step [0-9]+|global_step|success|fixed|eval' "$PI_RUN" --include='*.log' --include='*.out' --include='*.txt' 2>/dev/null | tail -80
else
  echo "MISSING $PI_RUN"
fi

echo '===FW_RUN_FILES===' 
if [ -d "$FW_RUN" ]; then
  find "$FW_RUN" -maxdepth 3 -type f -printf '%T@\t%s\t%p\n' 2>/dev/null | sort -nr | head -40
  echo '---fw checkpoints---'
  find "$FW_RUN" -maxdepth 5 -type d -name 'global_step_*' -printf '%T@\t%p\n' 2>/dev/null | sort -n | tail -20
  echo '---fw latest metric lines---'
  grep -RhaE 'Step [0-9]+|global_step|success|fixed|eval' "$FW_RUN" --include='*.log' --include='*.out' --include='*.txt' 2>/dev/null | tail -80
else
  echo "MISSING $FW_RUN"
fi

echo '===RAY_NAMESPACES===' 
ray list actors --address=172.17.0.1:6389 --filter "state=ALIVE" --format=json 2>/dev/null | python -c 'import json,sys,collections; a=json.load(sys.stdin); c=collections.Counter(x.get("rayNamespace") for x in a); print(dict(c))' || true
