#!/usr/bin/env bash
set -euo pipefail

printf 'MARKER=SZ_CREATE_TIANFENGRUI_STANDARD_USER_V1\n'

if getent passwd tianfengrui >/dev/null; then
  echo 'error=user_already_exists' >&2
  exit 10
fi
getent group labdata >/dev/null
test -d /data/shared

# stdin line 1 is consumed by sudo; the remaining chpasswd record is consumed below.
sudo -S -p '' /usr/sbin/useradd -m -U -s /bin/bash tianfengrui
sudo -n /usr/sbin/usermod -aG labdata tianfengrui
sudo -n chmod 700 /home/tianfengrui
sudo -n install -d -o tianfengrui -g tianfengrui -m 700 /data/tianfengrui
sudo -n -u tianfengrui ln -s /data/tianfengrui /home/tianfengrui/data
sudo -n -u tianfengrui ln -s /data/shared /home/tianfengrui/shared
sudo -n /usr/sbin/chpasswd

if id -nG tianfengrui | tr ' ' '\n' | grep -Fxq sudo; then
  echo 'error=unexpected_sudo_group' >&2
  exit 11
fi

printf '%s\n' '=== ACCOUNT ==='
getent passwd tianfengrui
id tianfengrui
sudo -n passwd -S tianfengrui
stat -c '%U:%G %a %n' /home/tianfengrui /data/tianfengrui
sudo -n readlink -f /home/tianfengrui/data
sudo -n readlink -f /home/tianfengrui/shared
printf '%s\n' '=== SUDO CONTRACT ==='
sudo -n -l -U tianfengrui 2>&1 || true
printf 'MARKER=SZ_CREATE_TIANFENGRUI_STANDARD_USER_OK\n'
