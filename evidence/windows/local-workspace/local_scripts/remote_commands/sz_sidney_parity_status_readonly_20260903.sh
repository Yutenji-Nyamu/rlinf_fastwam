set -eu
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PARITY="$WT/toolkits/lerobot/sidney_pi05_parity.py"
printf 'HEAD='; git -C "$WT" rev-parse HEAD
printf 'STATUS='; if test -z "$(git -C "$WT" status --porcelain)"; then echo clean; else echo dirty; fi
if test -f "$PARITY"; then
  stat -c 'PARITY=%n bytes=%s mtime=%y' "$PARITY"
else
  echo 'PARITY=missing'
fi
ps -eo user,pid,etimes,args --sort=-etimes | grep -E 'sidney_pi05_parity|export-native|export-rlinf|pi05-current-rlinf' | grep -v grep || true
find /data/chenyiteng/results/rlinf-shenzhen/pi05-sidney -maxdepth 4 -type f \
  \( -name 'input.npz' -o -name 'native.pt' -o -name 'rlinf.pt' -o -name 'report.json' \) \
  -printf 'ARTIFACT=%p bytes=%s mtime=%TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort | tail -20
