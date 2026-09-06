set -eu
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-core224-m10-phys4-v7
date --iso-8601=seconds
printf 'exit='; cat "$RUN/runtime/exit_code.txt" 2>/dev/null || echo running
for f in native.log rlinf.log runtime/wrapper.log report.json; do
  if test -e "$RUN/$f"; then
    stat -c "$f bytes=%s mtime=%y" "$RUN/$f"
    tail -n 8 "$RUN/$f"
  else
    echo "$f missing"
  fi
done
pgrep -af 'parity-core224-m10-phys4-v7|sidney_pi05_parity.py' || true
nvidia-smi -i 4,5 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
