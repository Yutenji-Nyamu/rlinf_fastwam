# Pure03 / Pure04 live snapshot

Snapshot: 2026-08-30 10:37:57 CST.

- Pure03: complete Step316/480; cumulative training-rollout success 42.52%; MA5/MA10/MA20 75.00/82.50/83.75%; Step300 fixed20 15/20.
- Pure04: complete Step313/480; cumulative 47.24%; MA5/MA10/MA20 95.00/92.50/88.75%; Step300 fixed20 18/20.
- Both drivers remain alive; narrow fatal signatures, CUDA OOM and OOM-kill are zero.
- DVAC is active. Latest Pure03 p05/mean/p95, ESS and top20 mass are 0.001/1.000/1.966, 0.761 and 0.354. Pure04 values are 0/1.000/2.301, 0.654 and 0.411.
- Pair resource trace: cgroup RAM latest/peak 238.55/240.00 GiB; GPU-memory peak 25.17/25.14 GiB.

Artifacts:

- `RLT_DVAC_PURE03_PURE04_SUCCESS_RAW_MA5_MA10_MA20.png`: raw and trailing 5/10/20-cycle training-rollout success.
- `success_curves.csv`: plotted values.
- `summary.json`: numeric snapshot, fixed evaluation, method weights and resources.
