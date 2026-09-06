set -u
for pgid in 820430 820500; do
  if ps -eo pgid= | awk -v g="$pgid" '$1==g {found=1} END {exit !found}'; then
    kill -TERM -- "-$pgid"
    echo "terminated owned git probe PGID $pgid"
  fi
done
ps -eo pid,ppid,pgid,stat,etime,args | grep -E 'git (push|ls-remote)|git-remote-https' | grep -v grep || true
