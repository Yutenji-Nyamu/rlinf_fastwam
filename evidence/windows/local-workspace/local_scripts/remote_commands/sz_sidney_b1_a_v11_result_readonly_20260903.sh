set -eu
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/b1-adjust-badseed-retry-m10-phys4-v11
date --iso-8601=seconds
printf 'exit='; cat "$RUN/runtime/exit_code.txt" 2>/dev/null || echo running
for f in runtime/driver.log runtime/resources.log; do
  echo "=== $f ==="
  if test -f "$RUN/$f"; then stat -c 'bytes=%s mtime=%y' "$RUN/$f"; tail -n 160 "$RUN/$f"; else echo missing; fi
done
pgrep -af 'pi05_sidney_b1_adjust_badseed_retry_m10_phys4_v11|b1-adjust-badseed-retry-m10-phys4-v11' || true
nvidia-smi -i 4,5 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
