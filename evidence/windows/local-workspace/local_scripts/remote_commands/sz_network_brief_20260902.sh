set -eu

echo 'network'
for url in https://github.com https://huggingface.co; do
  printf '%s direct ' "$url"
  curl --noproxy '*' -L -sS -o /dev/null --connect-timeout 5 --max-time 10 \
    -w 'http=%{http_code} connect=%{time_connect}s total=%{time_total}s\n' "$url" || echo 'failed'
done

echo 'mihomo'
systemctl is-active mihomo 2>/dev/null || true
ss -ltn 2>/dev/null | awk '$4 ~ /:(7890|7891|7892|7893|7897)$/ {print}' || true

for url in https://github.com https://huggingface.co; do
  printf '%s proxy ' "$url"
  curl -x http://127.0.0.1:7890 -L -sS -o /dev/null --connect-timeout 5 --max-time 10 \
    -w 'http=%{http_code} connect=%{time_connect}s total=%{time_total}s\n' "$url" || echo 'failed'
done
