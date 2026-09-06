set -euo pipefail
tmp=$(mktemp -d /data/chenyiteng/tmp/lerobot-contract-XXXXXX)
trap 'rm -rf "$tmp"' EXIT
git clone -q --depth 1 --branch v0.6.0 https://github.com/huggingface/lerobot.git "$tmp/lerobot"
echo '=== pyproject package contract ==='
sed -n '1,180p' "$tmp/lerobot/pyproject.toml"
echo '=== robotwin official install/eval ==='
sed -n '70,180p' "$tmp/lerobot/docs/source/robotwin.mdx"
echo '=== env path loading ==='
sed -n '220,340p' "$tmp/lerobot/src/lerobot/envs/robotwin.py"
echo '=== gpu4-5 owners ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
