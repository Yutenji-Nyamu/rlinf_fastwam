set -u
pid="$(pgrep -f '/home/chenyiteng/miniforge3/bin/uv sync .*FasterWAM-hustvl-official/environments/robotwin' | head -n 1)"
printf 'PID=%s\n' "${pid:-}"
if test -n "$pid"; then
  printf 'CMDLINE_NUL_AS_SPACES='
  tr '\000' ' ' < "/proc/$pid/cmdline"
  printf '\n'
fi
