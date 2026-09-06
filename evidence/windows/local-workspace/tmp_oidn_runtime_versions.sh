set -u
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python

echo '=== python packages ==='
"$PY" - <<'PY'
import importlib.metadata as m
import pathlib
import sapien
print('sapien.__file__=', sapien.__file__)
for name in ('sapien','numpy','torch'):
    try: print(name, m.version(name))
    except Exception as e: print(name, type(e).__name__)
p=pathlib.Path(sapien.__file__).resolve().parent
print('sapien_root=', p)
for x in sorted(p.rglob('*')):
    n=x.name.lower()
    if x.is_file() and ('oidn' in n or 'openimagedenoise' in n or 'svulkan' in n):
        print(x)
PY

echo '=== candidate shared libraries ==='
SITE=$($PY -c 'import pathlib,sapien; print(pathlib.Path(sapien.__file__).resolve().parent)')
find "$SITE" -type f \( -iname '*oidn*' -o -iname '*openimagedenoise*' -o -iname '*svulkan*' \) -printf '%p\n' | sort

echo '=== linked versions/sonames ==='
while IFS= read -r lib; do
  echo "-- $lib"
  readelf -d "$lib" 2>/dev/null | grep -E 'SONAME|NEEDED.*(OpenImageDenoise|oidn|vulkan|tbb)' || true
  strings "$lib" 2>/dev/null | grep -E -m 8 'Open Image Denoise|OIDN_VERSION|2\.[0-9]+\.[0-9]+' || true
done < <(find "$SITE" -type f \( -iname '*oidn*.so*' -o -iname '*openimagedenoise*.so*' -o -iname '*svulkan*.so*' \) | sort)
