set -euo pipefail
date -Is
for path in \
  /root/autodl-tmp/oracles/qam-2726d767 \
  /root/autodl-tmp/oracles/qam-2726d767.failed-bundle-20260731 \
  /root/autodl-tmp/venvs/qam-oracle-2726d767; do
  if test -e "$path"; then
    printf 'EXISTS %s\n' "$path"
    du -sx --block-size=1 "$path"
  else
    printf 'MISSING %s\n' "$path"
  fi
done
if test -d /root/autodl-tmp/oracles/qam-2726d767/.git; then
  git -C /root/autodl-tmp/oracles/qam-2726d767 rev-parse HEAD
  git -C /root/autodl-tmp/oracles/qam-2726d767 status --short
fi
