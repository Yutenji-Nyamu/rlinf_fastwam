hostname
date
cd /root/autodl-tmp/RLinf_fastwam_rlinf
git rev-parse HEAD
git status --short --branch
RUN=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260719_124315-robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu
tail -n 220 "$RUN/run_embodiment.log"
cat "$RUN/resource_monitor/peak.txt"
