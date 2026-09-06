set -u

date '+TIME=%F %T %Z'
hostname
id
printf 'MEMORY\n'
free -h
printf 'FILESYSTEMS\n'
df -h / /home /data
printf 'GPUS\n'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
printf 'GPU_PROCESSES\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
printf 'TARGETS\n'
for p in \
  /data/chenyiteng/projects/fasterwam-standalone \
  /data/chenyiteng/models/fasterwam \
  /data/chenyiteng/results/fasterwam-standalone \
  /home/chenyiteng/venvs \
  /home/chenyiteng/.cache/uv \
  /data/chenyiteng/projects/fastwam-standalone \
  /data/chenyiteng/models/fastwam
do
  if [ -e "$p" ]; then
    du -sh "$p" 2>/dev/null || true
  else
    printf 'MISSING %s\n' "$p"
  fi
done
printf 'WAN_CANDIDATES\n'
find /data/chenyiteng/models/fastwam -maxdepth 3 -type f \
  \( -iname '*wan*' -o -iname '*vae*' -o -iname '*t5*' -o -iname '*token*' \) \
  -printf '%s %p\n' 2>/dev/null | sort -n | tail -40
printf 'TOOLS\n'
command -v git || true
command -v uv || true
command -v curl || true
python3 --version || true
printf 'OFFICIAL_HEAD\n'
timeout 30 git ls-remote https://github.com/hustvl/FasterWAM.git refs/heads/main || true
printf 'NETWORK_DIRECT\n'
timeout 20 curl -LIsS -o /dev/null -w 'github=%{http_code} %{time_total}\n' https://github.com/hustvl/FasterWAM || true
timeout 20 curl -LIsS -o /dev/null -w 'hf=%{http_code} %{time_total}\n' https://huggingface.co/hustvl/FasterWAM/resolve/main/robotwin/dataset_stats.json || true
printf 'PROXY_SERVICE\n'
systemctl is-active mihomo 2>/dev/null || true
printf 'RELEVANT_PROCESSES\n'
ps -eo user,pid,ppid,etimes,rss,cmd --sort=-rss | grep -E 'chenyiteng|FasterWAM|FastWAM|ray::|gcs_server|raylet' | head -80 || true
