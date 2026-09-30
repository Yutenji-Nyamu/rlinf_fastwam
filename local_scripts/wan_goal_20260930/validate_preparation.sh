#!/bin/bash
# Run after the corresponding detached installer/download has completed.
set -euo pipefail
ROOT=/data/chenyiteng/projects/wan-goal-sz3
kind=${1:?assets or oft or pi05}
stamp=${2:?unique receipt label}
case "$kind" in assets|oft|pi05) ;; *) exit 2;; esac
[[ "$stamp" =~ ^[a-zA-Z0-9_-]+$ ]]
OUT="$ROOT/logs/validation-$stamp-$kind"
role="install-$kind"
[ "$kind" != assets ] || role=download
python3 - "$ROOT/logs/$role-current.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['phase']=='COMPLETE' and d['exit_code']==0,d
print('COMPLETION_RECEIPT_VERIFIED',d['role'],d['finished'])
PY
test ! -e "$OUT"
mkdir -m 700 "$OUT"
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/cache/huggingface" XDG_CACHE_HOME="$ROOT/cache/xdg"
export OPENPI_DATA_HOME="$ROOT/models/openpi-assets" TORCH_HOME="$ROOT/cache/torch"
export TRITON_CACHE_DIR="$ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/torchinductor"
export WAN_PATH="$ROOT/src/diffsynth-studio" ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PATH=/data/chenyiteng/venvs/rlinf-sz1-parity-py311-20260917/bin:$PATH
if [ "$kind" = assets ]; then
  python3 "$ROOT/scripts/verify_assets.py" --root "$ROOT" --output "$OUT/assets.json"
  python3 "$ROOT/scripts/tokenizer_fix.py" > "$OUT/tokenizer.log" 2>&1
  cat "$OUT/tokenizer.log"
else
  PY="$ROOT/envs/$kind-wan/bin/python"
  REPO="$ROOT/RLinf"
  export LIBERO_CONFIG_PATH="$ROOT/config/libero"
  if [ "$kind" = pi05 ]; then
    REPO="$ROOT/RLinf-pi05"
    export LIBERO_CONFIG_PATH="$ROOT/config/libero-pi05"
  fi
  cd "$REPO"
  set +e
  uv --no-config pip check --python "$PY" > "$OUT/pip-check.log" 2>&1
  pip_code=$?
  "$PY" "$ROOT/scripts/check_env_cpu.py" --repo "$REPO" --kind "$kind" --receipt "$OUT/imports.json" > "$OUT/imports.log" 2>&1
  import_code=$?
  set -e
  cat "$OUT/pip-check.log"
  cat "$OUT/imports.log"
  printf 'PIP_CHECK_EXIT=%s IMPORTS_EXIT=%s\n' "$pip_code" "$import_code"
  if [ "$kind" = pi05 ] && [ "$import_code" -eq 0 ]; then
    "$PY" "$ROOT/scripts/pi05/check_contract_cpu.py" --repo "$REPO" --model-path "$ROOT/models/pi05-libero" --wm-path "$ROOT/models/wan-goal" --receipt "$OUT/pi05-contract.json"
  fi
  test "$pip_code" -eq 0
  test "$import_code" -eq 0
fi
printf 'PREPARATION_CPU_VALIDATED %s %s\n' "$kind" "$OUT"
