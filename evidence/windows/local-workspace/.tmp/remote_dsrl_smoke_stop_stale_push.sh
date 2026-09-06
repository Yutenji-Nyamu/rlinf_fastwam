set -euo pipefail

for pid in 66754 66761 66763; do
  test -r "/proc/$pid/cmdline"
done
tr '\0' ' ' < /proc/66754/cmdline | grep -F 'BRANCH=codex/dsrl-pi0-robotwin HEAD=ff0d8d22'
tr '\0' ' ' < /proc/66761/cmdline | grep -F 'git -C /root/autodl-tmp/RLinf_fastwam_rlinf push personal codex/dsrl-pi0-robotwin'
tr '\0' ' ' < /proc/66763/cmdline | grep -F 'git-remote-https personal https://github.com/Yutenji-Nyamu/rlinf_fastwam.git'

kill -TERM 66763 66761 66754
sleep 2
for pid in 66754 66761 66763; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "STALE_PUSH_PID_REMAINS=$pid"
    exit 73
  fi
done
echo "STALE_PUSH_STOPPED=1"
