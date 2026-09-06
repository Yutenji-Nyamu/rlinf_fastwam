#!/usr/bin/env bash
set -u
FILE=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages/numpydantic/ndarray.pyi
printf '=== FILE_CONTENT ===\n'
nl -ba "$FILE" 2>/dev/null || true
printf '=== PIP_METADATA ===\n'
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -m pip show numpydantic 2>/dev/null || true
printf 'SZ_NUMPYDANTIC_STUB_READONLY_OK\n'
