#!/usr/bin/env bash
set -euo pipefail

printf 'MARKER=SZ_CREATE_QIUFUWEN_STANDARD_USER_V1\n'

if getent passwd qiufuwen >/dev/null; then
  echo 'error=user_already_exists' >&2
  exit 10
fi
getent group labdata >/dev/null
test -d /data/shared

# stdin line 1 is consumed by sudo; the remaining chpasswd record is consumed below.
sudo -S -p '' /usr/sbin/useradd -m -U -s /bin/bash qiufuwen
sudo -n /usr/sbin/usermod -aG labdata qiufuwen
sudo -n chmod 700 /home/qiufuwen
sudo -n install -d -o qiufuwen -g qiufuwen -m 700 /data/qiufuwen
sudo -n -u qiufuwen ln -s /data/qiufuwen /home/qiufuwen/data
sudo -n -u qiufuwen ln -s /data/shared /home/qiufuwen/shared
sudo -n /usr/sbin/chpasswd

if id -nG qiufuwen | tr ' ' '\n' | grep -Fxq sudo; then
  echo 'error=unexpected_sudo_group' >&2
  exit 11
fi

printf '%s\n' '=== ACCOUNT ==='
getent passwd qiufuwen
id qiufuwen
sudo -n passwd -S qiufuwen
stat -c '%U:%G %a %n' /home/qiufuwen /data/qiufuwen
sudo -n readlink -f /home/qiufuwen/data
sudo -n readlink -f /home/qiufuwen/shared
printf '%s\n' '=== SUDO CONTRACT ==='
sudo -n -l -U qiufuwen 2>&1 || true
printf 'MARKER=SZ_CREATE_QIUFUWEN_STANDARD_USER_OK\n'
