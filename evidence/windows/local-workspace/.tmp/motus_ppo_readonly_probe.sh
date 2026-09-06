set -eu
for root in /root/autodl-tmp/RLinf /root/autodl-tmp/wamppo /root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40; do
  echo "[ROOT $root]"
  if [ -d "$root" ]; then
    find "$root/rlinf/models/embodiment" -maxdepth 3 -type f \( -iname '*motus*.py' -o -iname '*lawam*.py' \) -print 2>/dev/null | sort
    find "$root/examples/embodiment/config" -maxdepth 2 -type f \( -iname '*motus*ppo*.yaml' -o -iname '*lawam*ppo*.yaml' -o -iname '*openpi*a800*ppo*.yaml' \) -print 2>/dev/null | sort
  fi
done
