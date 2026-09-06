#!/usr/bin/env bash
set -euo pipefail

ps -u "$(id -u)" -o pid=,ppid=,pgid=,stat=,etime=,args= --sort=pid \
  | grep -E 'rlinf|uv pip|curobo|timeout' || true
