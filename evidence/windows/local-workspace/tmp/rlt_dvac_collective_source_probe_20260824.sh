set -u
cd /root/autodl-tmp/RLinf_rlt_teacher_dvac
printf 'PROCESS_METRICS_DEF\n'
files=$(grep -RIl 'def process_train_metrics' rlinf --include='*.py' || true)
for f in $files; do echo FILE=$f; n=$(grep -n 'def process_train_metrics' "$f" | head -1 | cut -d: -f1); s=$((n-10)); [ $s -lt 1 ] && s=1; e=$((n+90)); sed -n "${s},${e}p" "$f"; done
printf 'ALL_REDUCE_DICT\n'
files=$(grep -RIl 'def all_reduce_dict' rlinf --include='*.py' || true)
for f in $files; do echo FILE=$f; n=$(grep -n 'def all_reduce_dict' "$f" | head -1 | cut -d: -f1); s=$((n-10)); [ $s -lt 1 ] && s=1; e=$((n+90)); sed -n "${s},${e}p" "$f"; done
