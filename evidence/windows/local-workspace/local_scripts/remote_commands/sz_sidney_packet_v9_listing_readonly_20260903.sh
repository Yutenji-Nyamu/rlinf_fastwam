set -eu
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets
for d in \
  "$ROOT/b1-adjust-badseed-retry-m10-phys4-v9" \
  "$ROOT/move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v9"; do
  echo "=== $d ==="
  find "$d" -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort
  echo '-- command --'
  sed -n '1,220p' "$d/command.txt"
done
