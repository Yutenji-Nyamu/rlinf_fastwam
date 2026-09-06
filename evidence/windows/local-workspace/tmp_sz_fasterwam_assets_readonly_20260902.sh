set -u

echo '=== TIME_HOST ==='
date --iso-8601=seconds
hostname
id

echo '=== HEALTH ==='
uptime
printf 'nproc='; nproc
free -h
awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo
cat /proc/pressure/memory 2>/dev/null || true
df -hT / /home /data

echo '=== GPU ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
echo '--- gpu process owners ---'
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -nu); do
  ps -p "$pid" -o user=,pid=,ppid=,etime=,%cpu=,%mem=,rss=,stat=,cmd= || true
done

echo '=== USER_AGGREGATES ==='
ps -eo user=,rss=,%cpu= | awk '{rss[$1]+=$2; cpu[$1]+=$3; n[$1]++} END {for (u in n) printf "%s\t%d\t%.1f\t%.3fGiB\n",u,n[u],cpu[u],rss[u]/1048576}' | sort -k4,4nr | head -n 20

echo '=== CHENYITENG_ROOTS ==='
ls -la /data/chenyiteng | sed -n '1,160p'
for path in /data/chenyiteng/projects /data/chenyiteng/models /data/chenyiteng/cache /data/chenyiteng/assets /data/chenyiteng/results /home/chenyiteng/venvs /home/chenyiteng/.cache; do
  [ -e "$path" ] && du -sh "$path" 2>/dev/null || true
done

echo '=== PROJECT_CANDIDATES ==='
for root in /data/chenyiteng/projects /data/chenyiteng/models /data/chenyiteng/cache /data/chenyiteng/assets /home/chenyiteng; do
  [ -d "$root" ] || continue
  find "$root" -maxdepth 4 -type d \( -iname '*faster*wam*' -o -iname '*fastwam*' -o -iname '*fast-wam*' -o -iname '*robotwin*' -o -iname '*wan2.2*' -o -iname '*wan-series*' \) -printf '%p\n' 2>/dev/null
done | sort -u | sed -n '1,300p'

echo '=== PROJECT_TOP_LEVEL ==='
find /data/chenyiteng/projects -mindepth 1 -maxdepth 2 -type d -printf '%p\n' 2>/dev/null | sort | sed -n '1,240p'

echo '=== MODEL_TOP_LEVEL ==='
find /data/chenyiteng/models -mindepth 1 -maxdepth 4 -type d -printf '%p\n' 2>/dev/null | sort | sed -n '1,320p'

echo '=== LARGE_MODEL_FILES_TARGETED ==='
for root in /data/chenyiteng/models /data/chenyiteng/cache /data/chenyiteng/assets /data/chenyiteng/projects/FastWAM /data/chenyiteng/projects/fastwam /data/chenyiteng/projects/fastwam-official; do
  [ -d "$root" ] || continue
  find "$root" -maxdepth 7 -type f \( -name '*.safetensors' -o -name '*.pt' -o -name '*.pth' -o -name '*.bin' \) -size +100M -printf '%s\t%p\n' 2>/dev/null
done | sort -nr | sed -n '1,160p'

echo '=== TARGET_DU ==='
for path in \
  /data/chenyiteng/projects/FastWAM \
  /data/chenyiteng/projects/fastwam \
  /data/chenyiteng/projects/fastwam-official \
  /data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support \
  /data/chenyiteng/projects/RoboTwin \
  /data/chenyiteng/models/*fastwam* \
  /data/chenyiteng/models/*FastWAM* \
  /data/chenyiteng/models/*Wan* \
  /data/chenyiteng/cache/*huggingface* \
  /data/chenyiteng/cache/*modelscope*; do
  [ -e "$path" ] && du -sh "$path" 2>/dev/null || true
done

echo '=== FASTWAM_RESOLVED_PATHS ==='
fwrun='/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1'
grep -Ein 'fastwam|wan|model_path|checkpoint|cache|tokenizer|vae|t5|robotwin|assets_path|norm' "$fwrun/runtime/resolved.yaml" 2>/dev/null | sed -n '1,260p' || true

echo '=== FASTWAM_PROCESS_ENV_PATHS ==='
pid=3589674
if [ -r "/proc/$pid/environ" ]; then
  tr '\0' '\n' < "/proc/$pid/environ" | grep -Ei '^(FASTWAM|ROBOTWIN|HF_HOME|HF_HUB_CACHE|HF_ENDPOINT|MODELSCOPE_CACHE|DIFFSYNTH|TRANSFORMERS_CACHE|TORCH_HOME|UV_CACHE_DIR|CONDA_PREFIX|VIRTUAL_ENV|PATH)=' | sed -E 's#(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=.*#\1=REDACTED#'
fi

echo '=== VENV_UV_CONDA ==='
command -v uv 2>/dev/null || true
uv --version 2>/dev/null || true
command -v conda 2>/dev/null || true
ls -ld /home/chenyiteng/{venvs,miniforge3,miniconda3,.conda,.local/bin} 2>/dev/null || true
find /home/chenyiteng/venvs -mindepth 1 -maxdepth 2 \( -type f -name pyvenv.cfg -o -type f -name python \) -printf '%p\n' 2>/dev/null | sort | sed -n '1,220p'
du -sh /home/chenyiteng/venvs/* 2>/dev/null | sort -h | tail -n 40
for exe in \
  /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python \
  /home/chenyiteng/venvs/fastwam/bin/python \
  /home/chenyiteng/venvs/fastwam-official/bin/python; do
  if [ -x "$exe" ]; then
    printf '%s\t' "$exe"; "$exe" -V 2>&1
    "$exe" -c 'import importlib.util as u; print("torch",bool(u.find_spec("torch")),"diffsynth",bool(u.find_spec("diffsynth")),"huggingface_hub",bool(u.find_spec("huggingface_hub")),"modelscope",bool(u.find_spec("modelscope")))' 2>/dev/null || true
  fi
done

echo '=== CACHE_LAYOUT ==='
for root in /data/chenyiteng/cache /home/chenyiteng/.cache/huggingface /home/chenyiteng/.cache/modelscope; do
  [ -d "$root" ] || continue
  echo "--- $root ---"
  find "$root" -mindepth 1 -maxdepth 4 -type d \( -name 'models--*' -o -name 'hub' -o -name 'snapshots' -o -name 'models' \) -printf '%p\n' 2>/dev/null | sort | sed -n '1,220p'
done

echo '=== NETWORK_DOWNLOAD_ROUTES ==='
printf 'mihomo='; systemctl is-active mihomo 2>/dev/null || true
ss -lnt 2>/dev/null | awk '$4 ~ /127\.0\.0\.1:(7890|9090)$/ {print}'
probe() {
  label="$1"; shift
  printf '%s\t' "$label"
  "$@" -L -sS -o /dev/null --connect-timeout 5 --max-time 12 -w 'http=%{http_code} connect=%{time_connect}s total=%{time_total}s remote=%{remote_ip}\n' 2>/dev/null || echo 'curl_failed'
}
probe github_direct curl --noproxy '*' https://github.com/
probe hf_direct curl --noproxy '*' 'https://huggingface.co/api/models?limit=1'
probe hf_proxy curl --proxy http://127.0.0.1:7890 'https://huggingface.co/api/models?limit=1'
probe hfmirror_direct curl --noproxy '*' 'https://hf-mirror.com/api/models?limit=1'
probe hfmirror_proxy curl --proxy http://127.0.0.1:7890 'https://hf-mirror.com/api/models?limit=1'
probe modelscope_direct curl --noproxy '*' 'https://www.modelscope.cn/api/v1/models'
probe modelscope_proxy curl --proxy http://127.0.0.1:7890 'https://www.modelscope.cn/api/v1/models'

