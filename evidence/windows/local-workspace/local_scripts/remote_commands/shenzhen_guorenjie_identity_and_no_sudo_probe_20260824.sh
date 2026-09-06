#!/usr/bin/env bash
set -euo pipefail

printf 'MARKER=SZ_GUORENJIE_IDENTITY_NO_SUDO_PROBE_V1\n'
test "$(id -un)" = guorenjie
id
test "$(readlink -f "$HOME/data")" = /data/guorenjie
test "$(readlink -f "$HOME/shared")" = /data/shared
test -w "$HOME"
test -w "$HOME/data"
if id -nG | tr ' ' '\n' | grep -Fxq sudo; then
  echo 'error=unexpected_sudo_group' >&2
  exit 20
fi
if sudo -n true >/dev/null 2>&1; then
  echo 'error=unexpected_passwordless_sudo' >&2
  exit 21
fi
printf 'home=%s\ndata=%s\nshared=%s\nsudo=denied\n' "$HOME" "$(readlink -f "$HOME/data")" "$(readlink -f "$HOME/shared")"
printf 'MARKER=SZ_GUORENJIE_IDENTITY_NO_SUDO_PROBE_OK\n'
