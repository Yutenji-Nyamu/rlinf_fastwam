set -euo pipefail
model=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
bytes() { find "$model" -type f -printf '%s\n' | awk '{s+=$1} END{printf "%.0f\n",s+0}'; }
b0=$(bytes); t0=$(date +%s)
echo "START bytes=$b0"
find "$model/.cache" -type f -printf '%s %p\n' 2>/dev/null | sort -nr | head -5 || true
sleep 30
b1=$(bytes); t1=$(date +%s)
rate=$(( (b1-b0)/(t1-t0) ))
echo "END bytes=$b1 delta=$((b1-b0)) seconds=$((t1-t0)) bytes_per_sec=$rate"
awk -v r="$rate" 'BEGIN{printf "MiB_per_sec=%.2f\n",r/1048576}'
