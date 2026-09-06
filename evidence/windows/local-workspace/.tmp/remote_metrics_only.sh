set -u
RUN=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260718_100910-robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu
grep -aE 'Global Step:|success_once=|actor/approx_kl=|actor/grad_norm=' "$RUN/run_embodiment.log" 2>/dev/null || true
