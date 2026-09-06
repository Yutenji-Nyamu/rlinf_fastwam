set -euo pipefail

echo "SYSTEM_RESOLUTION"
getent ahostsv4 github.com || true
echo "PUBLIC_DNS_1_1_1_1"
if command -v dig >/dev/null 2>&1; then
  dig +time=3 +tries=1 +short github.com @1.1.1.1 || true
else
  echo "dig unavailable"
fi
echo "PUBLIC_DNS_8_8_8_8"
if command -v dig >/dev/null 2>&1; then
  dig +time=3 +tries=1 +short github.com @8.8.8.8 || true
fi
echo "TLS_DEFAULT"
curl -sS -I --connect-timeout 8 --max-time 12 https://github.com/ | sed -n '1,5p' || true
