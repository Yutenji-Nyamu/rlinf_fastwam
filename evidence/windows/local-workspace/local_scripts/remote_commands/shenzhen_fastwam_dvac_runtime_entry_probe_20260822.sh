#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
PREFIX=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
cd "$WT"
echo "TIME_UTC=$(date -u +%FT%TZ)"
echo "PREFIX_BEGIN"
ls -ld "$PREFIX" || true
ls -la "$PREFIX" || true
ls -la "$PREFIX/bin" || true
readlink "$PREFIX/bin" || true
readlink -f "$PREFIX/bin/python" || true
find "$PREFIX" -maxdepth 3 -type f \( -name python -o -name activate -o -name pyvenv.cfg \) -print 2>/dev/null | sort | head -80
echo "PREFIX_END"
echo "CONDA_BEGIN"
command -v conda || true
command -v micromamba || true
find /home/chenyiteng -maxdepth 4 -type f -path '*/bin/python' 2>/dev/null | grep -E 'fastwam|conda|miniforge|venv' | sort | head -120 || true
echo "CONDA_END"
echo "IGNORE_BEGIN"
git check-ignore -v tests/test_fastwam_dvac_telemetry.py || true
git status --short --untracked-files=all
echo "IGNORE_END"
echo FASTWAM_DVAC_RUNTIME_ENTRY_PROBE_OK
