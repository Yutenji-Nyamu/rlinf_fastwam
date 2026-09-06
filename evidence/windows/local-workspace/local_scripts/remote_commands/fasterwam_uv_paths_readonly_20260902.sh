set -u

printf 'KNOWN_ACTUAL_CMDLINES\n'
ps -eo pid,args | grep -E '/home/chenyiteng/.*/uv (sync|pip)' | grep -v grep || true
printf 'UV_CANDIDATES\n'
for path in \
  /home/chenyiteng/.local/bin/uv \
  /home/chenyiteng/.cargo/bin/uv \
  /home/chenyiteng/miniforge3/bin/uv \
  /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/uv; do
  if test -e "$path"; then
    stat -c '%A %U:%G %s %y %n' "$path"
    printf 'RESOLVED %s -> %s\n' "$path" "$(readlink -f "$path")"
  else
    printf 'ABSENT %s\n' "$path"
  fi
done
printf 'INSTALLER_PATH_RESOLUTION\n'
PATH=/home/chenyiteng/miniforge3/bin:$PATH command -v uv
