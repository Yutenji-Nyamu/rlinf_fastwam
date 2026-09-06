set -eu
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-m10-phys4-v6-chw
printf 'exit='; cat "$RUN/runtime/exit_code.txt" 2>/dev/null || echo missing
for f in native.log rlinf.log runtime/wrapper.log report.json; do
  echo "=== $f ==="
  if test -f "$RUN/$f"; then
    stat -c 'bytes=%s mtime=%y' "$RUN/$f"
    tail -n 120 "$RUN/$f"
  else
    echo missing
  fi
done
echo '=== resource ==='
cat "$RUN/runtime/resources.log" 2>/dev/null || true
nvidia-smi -i 4,5 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
