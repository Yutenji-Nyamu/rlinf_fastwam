set -euo pipefail
echo '=== mihomo profile ==='
sed -n '1,200p' /etc/profile.d/mihomo-proxy.sh
echo '=== hf model identity ==='
meta=$(mktemp /tmp/sidney-meta-api-XXXXXX.json)
curl -fsSL -A 'curl/8' --max-time 60 https://hf-mirror.com/api/models/SidneyXie/pi05_robotwin -o "$meta"
/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python - "$meta" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f: d=json.load(f)
print('sha', d.get('sha'))
print('lastModified', d.get('lastModified'))
print('downloads', d.get('downloads'), 'likes', d.get('likes'))
for s in d.get('siblings',[]): print(s.get('rfilename'), s.get('size'))
PY
rm -f "$meta"
echo '=== small checkpoint metadata ==='
base=https://hf-mirror.com/SidneyXie/pi05_robotwin/resolve/main
for f in README.md config.json train_config.json policy_preprocessor.json policy_postprocessor.json; do
  echo "--- $f ---"
  curl -fsSL --max-time 60 "$base/$f" | head -c 30000
  echo
done
echo '=== v0.6 robotwin docs/package deps ==='
tmp=$(mktemp -d /data/chenyiteng/tmp/sidney-meta-XXXXXX)
trap 'rm -rf "$tmp"' EXIT
git clone -q --depth 1 --branch v0.6.0 https://github.com/huggingface/lerobot.git "$tmp/lerobot"
git -C "$tmp/lerobot" rev-parse HEAD
grep -R "robotwin" -n "$tmp/lerobot/pyproject.toml" "$tmp/lerobot/docs/source/robotwin.mdx" "$tmp/lerobot/src/lerobot" | head -80 || true
sed -n '/robotwin *=/,/]/p' "$tmp/lerobot/pyproject.toml" || true
grep -n "pi05\|pi0" "$tmp/lerobot/pyproject.toml" | head -40 || true
