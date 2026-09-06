# Pure03 / Pure04 user-stopped closeout

- Pure03: complete Step448; cumulative train success 55.11%; raw/MA5/MA10/MA20=`75/90/86.25/86.88%`; Step425 fixed20=`16/20`.
- Pure04: complete Step446; cumulative train success 59.98%; raw/MA5/MA10/MA20=`100/95/90/92.5%`; Step425 fixed20=`18/20`.
- Both experiment process groups and the pair-dedicated Ray head exited; GPU memory returned to 0 MiB. CUDA OOM and cgroup OOM/OOM-kill were zero.
- Server checkpoint tensors remain in place; latest complete checkpoint for both is Step425.

`RLT_DVAC_PURE03_PURE04_SUCCESS_RAW_MA5_MA10_MA20.png` is the stopped pair plot. The lightweight ZIP also contains logs, configs, metrics, latest DVAC traces, resource history, checkpoint inventories and the final six-run plot.
