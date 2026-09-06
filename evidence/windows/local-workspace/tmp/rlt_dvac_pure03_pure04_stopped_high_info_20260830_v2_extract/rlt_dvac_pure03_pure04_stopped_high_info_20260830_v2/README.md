# RLT-DVAC Pure03 / Pure04 stopped high-information package

User-requested stop: 2026-08-30 19:55:41 CST.

- Pure03 stopped after complete Step448/480; cumulative train success 55.11%; MA5/MA10/MA20 90.0/86.25/86.88%; latest fixed20 at Step425 was 16/20.
- Pure04 stopped after complete Step446/480; cumulative train success 59.98%; MA5/MA10/MA20 95.0/90.0/92.5%; latest fixed20 at Step425 was 18/20.
- Both owned process groups and their dedicated Ray head exited. GPU memory returned to 0 MiB; CUDA OOM and cgroup OOM/OOM-kill were zero.
- Latest retained checkpoint for both runs is Step425; checkpoint tensors are not duplicated into this package.
- Pure03/Pure04 latest weight ESS is 0.763/0.655; top-20% weight mass is 0.354/0.411.

Contents: complete metrics logs, resolved configs and commands, concise foreground tails, latest lightweight DVAC traces, checkpoint inventories, pair resource trace, final state, pair/six-run plots, curve CSV and JSON summaries.

Excluded to keep the package small: checkpoint tensors, replay buffers, videos, full Ray session logs and full foreground logs.
