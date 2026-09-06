set -euo pipefail

CURRENT=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin

find /home/chenyiteng/venvs -maxdepth 2 -type f -path '*/bin/python' -print | sort
"$CURRENT/bin/python" - <<'PY'
for name in ("boto3", "botocore", "s3transfer", "jmespath"):
    try:
        module = __import__(name)
        print(f"CURRENT {name} {getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"CURRENT_MISSING {name} {type(exc).__name__}")
PY

for python in /home/chenyiteng/venvs/*/bin/python; do
  case "$python" in
    *rlinf-7d07-openpi-robotwin*) continue ;;
  esac
  "$python" - <<'PY' 2>/dev/null || true
import sys
try:
    import boto3, botocore, s3transfer, jmespath
except Exception:
    raise SystemExit(1)
print(sys.executable, boto3.__version__, botocore.__version__, s3transfer.__version__, jmespath.__version__)
PY
done
