#!/usr/bin/env bash
set -u
date -Is
id
ps -o user:16,pid,ppid,pgid,sid,lstart,etime,pcpu,pmem,rss,nlwp,stat,wchan:24,args -p 201345,200301,916861
ps -eo user:16,pid,ppid,etime,pcpu,rss,args | awk '$1=="liwenbo" && ($3==201345 || $2==201345 || $2==200301 || $2==916861)'
for pid in 201345 200301; do
  printf 'PROC_VISIBLE_METADATA %s\n' "$pid"
  readlink "/proc/$pid/cwd" "/proc/$pid/exe" || true
  sed -n '/^Name:/p;/^State:/p;/^PPid:/p;/^Uid:/p;/^VmRSS:/p;/^Threads:/p;/^voluntary_ctxt_switches:/p;/^nonvoluntary_ctxt_switches:/p' "/proc/$pid/status" 2>/dev/null || true
done
nvidia-smi pmon -c 3 -s um -d 2
ps -o pid,ppid,etime,pcpu,rss,stat,args -p 201345,200301
