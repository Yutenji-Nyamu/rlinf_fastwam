#!/usr/bin/env bash

set +e

printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
sudo -S -p '' true
printf 'sudo_auth_rc=%s\n' "$?"

printf '%s\n' '=== kernel severe events since PPO launch day ==='
sudo -n journalctl -k --since '2026-08-22 00:00:00' --no-pager -o short-iso 2>&1 \
  | grep -Ei 'NVRM|Xid|I/O error|EXT4-fs error|XFS.*error|nvme.*(reset|timeout|fatal)|out of memory|oom-kill|killed process' \
  | tail -n 120
printf 'kernel_severe_match_count='
sudo -n journalctl -k --since '2026-08-22 00:00:00' --no-pager -o cat 2>&1 \
  | grep -Eic 'NVRM|Xid|I/O error|EXT4-fs error|XFS.*error|nvme.*(reset|timeout|fatal)|out of memory|oom-kill|killed process'

printf '%s\n' '=== ssh auth since midnight ==='
sudo -n journalctl -u ssh.service --since '2026-08-22 00:00:00' --no-pager -o cat 2>&1 | awk '
  /Accepted (password|publickey)/ {
    accepted++
    for (i=1; i<=NF; i++) {
      if ($i=="for" && (i+1)<=NF) user[$(i+1)]++
      if ($i=="from" && (i+1)<=NF) accepted_source[$(i+1)]++
    }
  }
  /Failed password/ {
    failed++
    for (i=1; i<=NF; i++) if ($i=="from" && (i+1)<=NF) failed_source[$(i+1)]++
  }
  /Invalid user/ {invalid++}
  END {
    printf "accepted=%d failed_password=%d invalid_user=%d\n",accepted,failed,invalid
    for (u in user) printf "accepted_user=%s count=%d\n",u,user[u]
    n=0; for (ip in accepted_source) n++; printf "accepted_unique_sources=%d\n",n
    n=0; for (ip in failed_source) n++; printf "failed_unique_sources=%d\n",n
  }'

printf '%s\n' '=== failed units ==='
sudo -n systemctl --failed --no-pager
sudo -k
exit 0
