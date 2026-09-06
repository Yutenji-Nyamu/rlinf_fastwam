set -u

REPO=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
printf 'VENV_PYTHON\n'
readlink -f "$REPO/.venvs/robotwin/bin/python" 2>/dev/null || true
"$REPO/.venvs/robotwin/bin/python" - <<'PY' 2>/dev/null || true
import sys, sysconfig
print(sys.executable)
print(sys.version)
print('include=', sysconfig.get_path('include'))
print('platinclude=', sysconfig.get_path('platinclude'))
PY

printf 'SYSTEM_INCLUDE_CANDIDATES\n'
find /usr/include /usr/local/include -type f -name Python.h -print 2>/dev/null | sort -u

printf 'USER_INCLUDE_CANDIDATES\n'
for root in \
  /home/chenyiteng/miniforge3/include \
  /home/chenyiteng/miniforge3/envs \
  /home/chenyiteng/.local/share/uv/python \
  /home/chenyiteng/.cache/uv/python \
  /data/chenyiteng/cache/uv-fasterwam \
  /home/chenyiteng/venvs; do
  test -d "$root" || continue
  find "$root" -type f -name Python.h -print 2>/dev/null
done | sort -u

printf 'DPKG_PYTHON_DEV\n'
dpkg-query -W -f='${Package}\t${Status}\t${Version}\n' 'python3.10-dev' 'libpython3.10-dev' 'python3-dev' 'libpython3-dev' 2>&1 || true
