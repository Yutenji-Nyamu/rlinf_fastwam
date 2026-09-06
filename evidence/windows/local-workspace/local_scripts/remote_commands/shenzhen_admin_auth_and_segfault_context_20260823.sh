#!/usr/bin/env bash
set -u

printf '=== AUTH LOG INVENTORY ===\n'
ls -l /var/log/auth.log* 2>/dev/null || true
printf '%s\n' '--- successful SSH by user/source across retained auth logs ---'
zgrep -hE 'sshd.*Accepted ' /var/log/auth.log* 2>/dev/null \
  | sed -nE 's/.* for ([^ ]+) from ([^ ]+).*/\1 \2/p' \
  | sort | uniq -c | sort -nr || true
printf '%s\n' '--- top failed/invalid sources across retained auth logs ---'
zgrep -hE 'sshd.*(Failed password|Invalid user)' /var/log/auth.log* 2>/dev/null \
  | sed -nE 's/.* from ([0-9a-fA-F:.]+) port.*/\1/p' \
  | sort | uniq -c | sort -nr | head -n 15 || true

printf '=== SEGFAULT CONTEXT ===\n'
journalctl --since '2026-08-22 04:28:30' --until '2026-08-22 04:31:00' \
  --no-pager -o short-iso 2>/dev/null | tail -n 120 || true
printf '%s\n' '--- exact PID fields if retained ---'
journalctl _PID=3368681 --since '2026-08-22 04:00:00' --until '2026-08-22 05:00:00' \
  --no-pager -o verbose 2>/dev/null | head -n 160 || true

printf 'SZ_ADMIN_AUTH_SEGFAULT_CONTEXT_OK\n'
