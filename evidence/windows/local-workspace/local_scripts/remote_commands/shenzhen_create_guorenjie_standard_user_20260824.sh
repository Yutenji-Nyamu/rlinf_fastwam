#!/usr/bin/env bash
set -euo pipefail

printf 'MARKER=SZ_CREATE_GUORENJIE_STANDARD_USER_V1\n'

if getent passwd guorenjie >/dev/null; then
  echo 'error=user_already_exists' >&2
  exit 10
fi
getent group labdata >/dev/null
test -d /data/shared

# stdin line 1 is consumed by sudo; the remaining chpasswd record is consumed below.
sudo -S -p '' /usr/sbin/useradd -m -U -s /bin/bash guorenjie
sudo -n /usr/sbin/usermod -aG labdata guorenjie
sudo -n chmod 700 /home/guorenjie
sudo -n install -d -o guorenjie -g guorenjie -m 700 /data/guorenjie
sudo -n -u guorenjie ln -s /data/guorenjie /home/guorenjie/data
sudo -n -u guorenjie ln -s /data/shared /home/guorenjie/shared
sudo -n /usr/sbin/chpasswd

if id -nG guorenjie | tr ' ' '\n' | grep -Fxq sudo; then
  echo 'error=unexpected_sudo_group' >&2
  exit 11
fi

printf '%s\n' '=== ACCOUNT ==='
getent passwd guorenjie
id guorenjie
sudo -n passwd -S guorenjie
stat -c '%U:%G %a %n' /home/guorenjie /data/guorenjie
readlink -f /home/guorenjie/data
readlink -f /home/guorenjie/shared
printf '%s\n' '=== SUDO CONTRACT ==='
sudo -n -l -U guorenjie 2>&1 || true
printf 'MARKER=SZ_CREATE_GUORENJIE_STANDARD_USER_OK\n'
