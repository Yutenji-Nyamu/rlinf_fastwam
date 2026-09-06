set -eu
pid=2071299
test -r "/proc/$pid/cmdline"
cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
case "$cmd" in *task=move_stapler_pad*) ;; *) printf 'refuse_task pid=%s\n' "$pid"; exit 42 ;; esac
case "$cmd" in *gpu=4*) ;; *) printf 'refuse_gpu pid=%s\n' "$pid"; exit 42 ;; esac
case "$cmd" in *"seeds='1000 1002 1004'"*) ;; *) printf 'refuse_seeds pid=%s\n' "$pid"; exit 42 ;; esac
kill -STOP "$pid"
ps -o pid,ppid,stat,args -p "$pid"
