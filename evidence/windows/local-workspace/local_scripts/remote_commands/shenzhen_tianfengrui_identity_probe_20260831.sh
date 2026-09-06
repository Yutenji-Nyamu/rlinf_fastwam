#!/usr/bin/env bash
set -euo pipefail

test "$(id -un)" = tianfengrui
test "$HOME" = /home/tianfengrui
test "$(readlink -f "$HOME/data")" = /data/tianfengrui
test "$(readlink -f "$HOME/shared")" = /data/shared
if id -nG | tr ' ' '\n' | grep -Fxq sudo; then
  exit 11
fi
if sudo -n true >/dev/null 2>&1; then
  exit 12
fi
printf 'user=%s\nhome=%s\ngroups=%s\ndata=%s\nshared=%s\nsudo=denied\n' \
  "$(id -un)" "$HOME" "$(id -nG)" "$(readlink -f "$HOME/data")" "$(readlink -f "$HOME/shared")"
