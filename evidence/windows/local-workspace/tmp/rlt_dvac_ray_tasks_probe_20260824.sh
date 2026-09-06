set -u
/root/autodl-tmp/RLinf/.venv/bin/ray list tasks --filter "state=RUNNING" --format=table --limit=200 | head -160 || true
