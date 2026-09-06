#!/usr/bin/env bash
set -euo pipefail

PARENT_PID=1373923
PIP_PID=1374024
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128

test "$(stat -c %U "/proc/$PARENT_PID")" = chenyiteng
test "$(stat -c %U "/proc/$PIP_PID")" = chenyiteng
test "$(awk '{print $4}' "/proc/$PIP_PID/stat")" = "$PARENT_PID"
test "$(readlink -f "/proc/$PIP_PID/exe")" = "$ENV/bin/python3.10"
tr '\0' ' ' < "/proc/$PIP_PID/cmdline" | grep -Fq 'torch==2.7.1+cu128'
tr '\0' ' ' < "/proc/$PIP_PID/cmdline" | grep -Fq 'torchvision==0.22.1+cu128'
tr '\0' ' ' < "/proc/$PARENT_PID/cmdline" | grep -Fq '/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711'

printf 'sending SIGINT to owned FastWAM pip pid=%s\n' "$PIP_PID"
kill -INT "$PIP_PID"

for _ in $(seq 1 30); do
  if ! kill -0 "$PIP_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done

if kill -0 "$PIP_PID" 2>/dev/null; then
  printf 'pip did not exit after SIGINT; sending SIGTERM to exact pid=%s\n' "$PIP_PID"
  kill -TERM "$PIP_PID"
fi

for _ in $(seq 1 30); do
  if ! kill -0 "$PARENT_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done

test ! -e "/proc/$PIP_PID"
test ! -e "/proc/$PARENT_PID"
printf '%s\n' FASTWAM_OWNED_ENV_INSTALL_STOPPED
