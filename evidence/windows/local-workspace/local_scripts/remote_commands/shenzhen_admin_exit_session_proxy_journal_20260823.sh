#!/usr/bin/env bash
set -u

ROOT_SCRIPT=$(cat <<'ROOT'
set -u

printf '=== SESSION_SCOPES_1150_1210_UTC ===\n'
journalctl --since '2026-08-23 11:50:00 UTC' --until '2026-08-23 12:10:00 UTC' --no-pager -o short-iso 2>/dev/null \
  | grep -Ei 'session-[0-9]+\.scope|Started Session|Removed session|Consumed .*memory|peak memory|sshd.*session (opened|closed)|Accepted (password|publickey)' \
  | tail -n 600 || true

printf '=== USER1003_SYSTEMD_1150_1210_UTC ===\n'
journalctl _UID=0 --since '2026-08-23 11:50:00 UTC' --until '2026-08-23 12:10:00 UTC' --no-pager -o short-iso 2>/dev/null \
  | grep -Ei 'user-1003|session-[0-9]+|memory|oom|kill|sshd' | tail -n 600 || true

printf '=== MIHOMO_1130_NOW_UTC ===\n'
journalctl -u mihomo --since '2026-08-23 11:30:00 UTC' --no-pager -o short-iso 2>/dev/null \
  | sed -E 's#https?://[^ ]+#URL_REDACTED#g; s#([?&](token|key|auth)=)[^& ]+#\1REDACTED#g' \
  | grep -Ei 'error|failed|timeout|TLS|SSL|EOF|handshake|provider|subscription|health|update' \
  | tail -n 300 || true

printf 'SZ_ADMIN_EXIT_SESSION_PROXY_JOURNAL_OK\n'
ROOT
)

sudo -S -k -p '' /bin/bash -c "$ROOT_SCRIPT"
rc=$?
sudo -k 2>/dev/null || true
exit "$rc"
