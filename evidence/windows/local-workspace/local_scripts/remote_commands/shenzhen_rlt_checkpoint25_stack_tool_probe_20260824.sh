#!/usr/bin/env bash
set -u

for tool in py-spy pstack gstack gdb perf; do
  printf '%s=' "$tool"; command -v "$tool" || true
done

