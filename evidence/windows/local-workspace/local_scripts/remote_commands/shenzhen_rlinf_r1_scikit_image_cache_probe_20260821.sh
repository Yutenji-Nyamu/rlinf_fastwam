#!/usr/bin/env bash
set -euo pipefail

find /home/chenyiteng/.cache/uv -type f -path '*/scikit_image-*.dist-info/METADATA' -print \
  -exec grep -E '^(Name|Version|Requires-Dist):' {} \; | head -n 100 || true
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import requests
metadata = requests.get("https://pypi.org/pypi/scikit-image/json", timeout=30).json()
info = metadata["info"]
print({"pypi_latest": info["version"], "requires_dist": info["requires_dist"]})
PY
