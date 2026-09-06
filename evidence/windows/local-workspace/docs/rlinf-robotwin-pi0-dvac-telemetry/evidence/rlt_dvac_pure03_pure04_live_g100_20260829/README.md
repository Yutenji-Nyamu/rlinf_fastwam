# Pure03 / Pure04 live snapshot

Snapshot: 2026-08-29 18:30:58 CST.

- Pure03: complete Step 100/480; cumulative training-rollout success 16.38%; latest MA5/MA10/MA20 12.50/11.25/13.75%; fixed20 at Step 100 is 0/20.
- Pure04: complete Step 99/480 and running the next rollout; cumulative 15.66%; latest MA5/MA10/MA20 12.50/16.25/15.63%; latest completed fixed20 is Step 75, 0/20.
- Both drivers are alive. Narrow fatal signatures, CUDA OOM and OOM-kill are all zero.
- Pair resource trace: cgroup RAM latest/peak 182.25/182.29 GiB; GPU memory peak 25.12/25.09 GiB.
- These cycles are still before the online actor-weighting phase: the logs do not yet contain Pure per-action BC weight statistics. Current curve differences therefore do not measure the DVAC weighting effect.

Artifacts:

- `RLT_DVAC_PURE03_PURE04_SUCCESS_RAW_MA5_MA10_MA20.png`: raw and trailing 5/10/20-cycle training-rollout success.
- `success_curves.csv`: plotted values.
- `summary.json`: numeric snapshot and resource summary.
