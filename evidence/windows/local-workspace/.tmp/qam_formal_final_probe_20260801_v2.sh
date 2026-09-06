set -u
date --iso-8601=seconds
printf 'TRAIN='; pgrep -fc '[t]rain_embodied_agent.py' || true
printf 'RAY='; pgrep -xc raylet || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'EXIT_FILE='; test -f /root/autodl-tmp/experiment_exports/qam_formal_20260801_v2/runtime/exit_code.txt && cat /root/autodl-tmp/experiment_exports/qam_formal_20260801_v2/runtime/exit_code.txt || printf 'absent-running\n'
