set -euo pipefail
source /etc/profile.d/mihomo-proxy.sh
base=/data/chenyiteng/projects/lerobot-sidney
sha=dcddb970176382c0fcf4521b0c0e6fc15894dfe0
tsrc=$base/transformers-$sha
if [ ! -d "$tsrc" ]; then
  tmp=$base/transformers-$sha.tar.gz.partial
  curl -fL --retry 4 --retry-delay 2 --max-time 600 \
    "https://codeload.github.com/huggingface/transformers/tar.gz/$sha" -o "$tmp"
  mkdir -p "$tsrc"
  tar -xzf "$tmp" -C "$tsrc" --strip-components=1
  rm -f "$tmp"
fi
echo "TRANSFORMERS_SOURCE sha=$sha size=$(du -sh "$tsrc" | cut -f1)"
venv=/home/chenyiteng/venvs/lerobot-v043-sidney-py310
src=$base/lerobot-0b067df57d21
log=$base/install-lerobot-v043-sidney-py310.retry.log
pidfile=$base/install-lerobot-v043-sidney-py310.retry.pid
nohup env \
  http_proxy="$http_proxy" https_proxy="$https_proxy" all_proxy="$all_proxy" \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  "$venv/bin/python" -m pip install \
    -e "$tsrc" \
    -e "$src" \
    'scipy>=1.10.1,<1.15' \
    -c "$base/constraints-v043-py310.txt" \
  >"$log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$pidfile"
echo "INSTALL_RETRY_STARTED pid=$pid log=$log"
sleep 3
tail -40 "$log" || true
