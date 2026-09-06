set -u
R=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
date '+TIME %Y-%m-%d %H:%M:%S %Z'
grep -RIn --include='*.py' -E '^    def close_env|sapien\.render\.clear_cache|self\.renderer = None|self\.scene = None' "$R" | head -400 || true
for f in $(grep -Rl --include='*.py' -E '^    def close_env' "$R" | sort -u | head -20); do
  echo "FILE $f"
  n=$(grep -n '^    def close_env' "$f" | head -1 | cut -d: -f1)
  s=$((n-8)); e=$((n+85)); nl -ba "$f" | sed -n "${s},${e}p"
done
