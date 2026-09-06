#!/usr/bin/env bash
set -euo pipefail

printf 'MARKER=SZ_CREATE_YANCHUHAN_STANDARD_USER_V1\n'

if getent passwd yanchuhan >/dev/null; then
  echo 'error=user_already_exists' >&2
  exit 10
fi
getent group labdata >/dev/null
test -d /data/shared

# stdin line 1 is consumed by sudo; the remaining chpasswd record is consumed below.
sudo -S -p '' /usr/sbin/useradd -m -U -s /bin/bash yanchuhan
sudo -n /usr/sbin/usermod -aG labdata yanchuhan
sudo -n chmod 700 /home/yanchuhan
sudo -n install -d -o yanchuhan -g yanchuhan -m 700 /data/yanchuhan
sudo -n -u yanchuhan ln -s /data/yanchuhan /home/yanchuhan/data
sudo -n -u yanchuhan ln -s /data/shared /home/yanchuhan/shared
sudo -n /usr/sbin/chpasswd

if id -nG yanchuhan | tr ' ' '\n' | grep -Fxq sudo; then
  echo 'error=unexpected_sudo_group' >&2
  exit 11
fi

printf '%s\n' '=== ACCOUNT ==='
getent passwd yanchuhan
id yanchuhan
sudo -n passwd -S yanchuhan
stat -c '%U:%G %a %n' /home/yanchuhan /data/yanchuhan
sudo -n readlink -f /home/yanchuhan/data
sudo -n readlink -f /home/yanchuhan/shared
printf '%s\n' '=== SUDO CONTRACT ==='
sudo -n -l -U yanchuhan 2>&1 || true
printf 'MARKER=SZ_CREATE_YANCHUHAN_STANDARD_USER_OK\n'
