#!/usr/bin/env bash
set -u
ROOT=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages/numpydantic
grep -RInE 'ndarray\.pyi|stub|write_text|open\(' "$ROOT" --include='*.py' 2>/dev/null | head -n 240 || true
printf 'SZ_NUMPYDANTIC_GENERATION_SOURCE_OK\n'
