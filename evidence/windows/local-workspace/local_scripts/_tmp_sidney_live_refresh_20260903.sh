set -eu
echo '=== time ==='
date '+%F %T %Z'
echo '=== gpu45 ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed -n '5,6p'
echo '=== owned sidney processes ==='
ps -u chenyiteng -o pid,ppid,etimes,stat,args --sort=pid | grep -E 'sidney|pi05_robotwin|lerobot.*eval|robotwin.*eval' | grep -v grep || true
echo '=== sidney project/output ==='
find /data/chenyiteng/projects/lerobot-sidney -maxdepth 3 -type f \( -name '*.log' -o -name '*.json' -o -name '*.mp4' -o -name '*.txt' -o -name '*.py' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -80
echo '=== output tree ==='
find /data/chenyiteng/results/lerobot-sidney -maxdepth 8 -printf '%y %TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort -k2,3 | tail -160
echo '=== task scripts ==='
find /data/chenyiteng/projects/lerobot-sidney -maxdepth 2 -type f -newermt '2026-09-02 14:00:00' -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -100
echo '=== run identities ==='
for f in /data/chenyiteng/results/lerobot-sidney/pi05_robotwin-e49e2ab/*/run_identity.txt; do echo "--- $f"; cat "$f"; done
echo '=== exit codes ==='
for f in /data/chenyiteng/results/lerobot-sidney/pi05_robotwin-e49e2ab/*/exit_code.txt; do echo "--- $f"; cat "$f"; done
echo '=== log signals ==='
for f in /data/chenyiteng/results/lerobot-sidney/pi05_robotwin-e49e2ab/*/eval.log; do echo "--- $f"; grep -E 'SUCCESS|FAIL|UnStable|success|episode|seed|Traceback|Error|Exception|strict|missing|unexpected|PARITY' "$f" | tail -80 || true; done
echo '=== likely runner scripts ==='
find /data/chenyiteng -xdev -type f \( -iname '*sidney*.py' -o -iname '*sidney*.sh' -o -iname '*robotwin*eval*.py' -o -iname '*lerobot*robotwin*.py' \) -newermt '2026-09-02 13:30:00' -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -100
