set -euo pipefail
echo '=== direct ==='
timeout 30 git ls-remote https://github.com/huggingface/transformers.git refs/heads/fix/lerobot_openpi || true
echo '=== mihomo ==='
source /etc/profile.d/mihomo-proxy.sh
curl -IsS --max-time 20 https://github.com | head -3 || true
timeout 45 git -c http.version=HTTP/1.1 ls-remote https://github.com/huggingface/transformers.git refs/heads/fix/lerobot_openpi || true
