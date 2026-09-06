set -euo pipefail
base=/data/chenyiteng/projects/lerobot-sidney
repo=$base/lerobot-30da8e687a6d
src=$base/lerobot-0b067df57d21
for pid in 1340419 1340435; do
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    [ -z "$cmd" ] && continue
    case "$cmd" in *lerobot-0b067df57d21*) echo "STOP_OWNED $pid"; kill -TERM "$pid" || true;; *) echo "REFUSE $pid $cmd"; exit 2;; esac
  fi
done
sleep 2
if [ -d "$src" ]; then
  partial=/data/chenyiteng/tmp/lerobot-0b067df57d21-partial-$(date +%Y%m%dT%H%M%S)
  mv "$src" "$partial"
  echo "MOVED_PARTIAL $partial"
fi
git -C "$repo" cat-file -e 0b067df57d21d3a02d6c511f1609172fa39ac29b^{commit}
GIT_LFS_SKIP_SMUDGE=1 git -C "$repo" worktree add -q --detach "$src" 0b067df57d21d3a02d6c511f1609172fa39ac29b
test -z "$(git -C "$src" status --porcelain)"
echo "WORKTREE $(git -C "$src" rev-parse HEAD) size=$(du -sh "$src" | cut -f1)"
venv=/home/chenyiteng/venvs/lerobot-v043-sidney-py310
if [ ! -x "$venv/bin/python" ]; then
  /home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python -m venv --system-site-packages "$venv"
fi
printf '%s\n' 'numpy<2' 'torch==2.7.1' 'torchvision==0.22.1' > "$base/constraints-v043-py310.txt"
log=$base/install-lerobot-v043-sidney-py310.log
pidfile=$base/install-lerobot-v043-sidney-py310.pid
nohup env PIP_DISABLE_PIP_VERSION_CHECK=1 \
  "$venv/bin/python" -m pip install -e "$src[pi]" -c "$base/constraints-v043-py310.txt" \
  >"$log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$pidfile"
echo "INSTALL_STARTED pid=$pid venv=$venv log=$log"
sleep 3
tail -40 "$log" || true
