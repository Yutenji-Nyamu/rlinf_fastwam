set -eu
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-core224-m10-compare-existing-v8
printf 'exit='; cat "$RUN/runtime/exit_code.txt"
stat -c 'report bytes=%s mtime=%y' "$RUN/report.json"
cat "$RUN/report.json"
nvidia-smi -i 4,5 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
