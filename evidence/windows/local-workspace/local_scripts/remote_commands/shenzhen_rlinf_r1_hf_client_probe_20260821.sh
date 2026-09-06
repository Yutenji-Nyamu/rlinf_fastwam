#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' '=== HF EXECUTABLES ==='
for candidate in \
  /home/chenyiteng/miniforge3/bin/hf \
  /home/chenyiteng/miniforge3/envs/*/bin/hf \
  /home/chenyiteng/.local/bin/hf; do
  test -x "$candidate" && "$candidate" --version || true
done

printf '%s\n' '=== PYTHON HF MODULES ==='
for python_bin in \
  /home/chenyiteng/miniforge3/bin/python \
  /home/chenyiteng/miniforge3/envs/*/bin/python; do
  test -x "$python_bin" || continue
  if "$python_bin" -c 'import huggingface_hub; print(huggingface_hub.__version__)' >/tmp/rlinf-hf-version.$$ 2>/dev/null; then
    printf '%s ' "$python_bin"
    cat /tmp/rlinf-hf-version.$$
  fi
  rm -f /tmp/rlinf-hf-version.$$
done

printf '%s\n' '=== PARTIAL STATE ==='
find /home/chenyiteng/.cache/.partial-openpi-tokenizer-befaa248-20260821 \
  -maxdepth 3 -printf '%y %s %p\n' 2>/dev/null | sort || true
