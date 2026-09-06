set -euo pipefail

URL_PATH=hustvl/FasterWAM/resolve/6bf9471ced6919a15ab8fded89f7772f5060c44b/robotwin/step_029355.pt

printf 'MIRROR_8MIB\n'
curl -L --fail --silent --show-error --max-time 30 \
  --range 0-8388607 \
  --output /dev/null \
  --write-out 'status=%{http_code} bytes=%{size_download} speed=%{speed_download} time=%{time_total}\n' \
  "https://hf-mirror.com/$URL_PATH"

printf 'PROXY_OFFICIAL_8MIB\n'
HTTPS_PROXY=http://127.0.0.1:7890 \
curl -L --fail --silent --show-error --max-time 30 \
  --range 0-8388607 \
  --output /dev/null \
  --write-out 'status=%{http_code} bytes=%{size_download} speed=%{speed_download} time=%{time_total}\n' \
  "https://huggingface.co/$URL_PATH"
