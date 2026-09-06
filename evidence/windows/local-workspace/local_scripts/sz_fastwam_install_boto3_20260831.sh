set -euo pipefail

VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
"$VENV/bin/python" -m pip install \
  boto3==1.35.99 \
  botocore==1.35.99 \
  s3transfer==0.10.4 \
  jmespath==1.0.1
"$VENV/bin/python" - <<'PY'
import boto3, botocore, jmespath, s3transfer
print("BOTO3_RUNTIME_OK", boto3.__version__, botocore.__version__, s3transfer.__version__, jmespath.__version__)
PY
