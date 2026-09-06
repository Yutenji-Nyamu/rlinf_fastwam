set -euo pipefail
old_run="/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1"
echo "=== directories ==="
find "$old_run" -maxdepth 5 -type d -printf '%p\n' | sort
echo "=== checkpoint-like files and dirs ==="
find /root/autodl-tmp/experiments /root/autodl-tmp/experiment_exports -maxdepth 7 \
  \( -path '*rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1*' -a \
     \( -iname '*checkpoint*' -o -iname '*trainer_state*' -o -iname '*global_step*' \) \) \
  -printf '%y %s %p\n' 2>/dev/null | sort | tail -n 300
echo "=== stage state ==="
find /root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/high_info_archive_20260825 -maxdepth 4 -printf '%y %s %p\n' 2>/dev/null | sort | head -n 100
