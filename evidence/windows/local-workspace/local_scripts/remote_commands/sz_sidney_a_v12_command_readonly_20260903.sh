set -euo pipefail
P=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/b1-adjust-badseed-retry-m10-phys4-v12
printf '%s\n' '--- command.txt ---'
cat "$P/command.txt"
printf '%s\n' '--- seed file ---'
cat "$P/eval-seeds-adjust-bad1001.json"
