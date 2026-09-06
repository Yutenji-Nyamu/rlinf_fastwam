set -euo pipefail

root=/data/chenyiteng/projects/robotwin-native/RoboTwin

date -Is
printf '%s\n' '=== active downloader processes ==='
pgrep -a -u "$(id -u)" -f '(_download_assets|snapshot_download|huggingface)' || true

printf '%s\n' '=== assets tree size and largest current files ==='
du -sh "$root/assets"
find "$root/assets" -type f -printf '%s\t%p\n' | sort -nr | head -n 20

printf '%s\n' '=== HF cache incomplete files ==='
find /home/chenyiteng/.cache/huggingface -type f -name '*.incomplete' -printf '%s\t%p\n' 2>/dev/null | sort -nr | head -n 20 || true

printf '%s\n' '=== target space ==='
df -h /data/chenyiteng
