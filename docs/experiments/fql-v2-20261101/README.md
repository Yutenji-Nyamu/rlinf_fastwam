# FQL v2 · 2026-11-01

Single-task native RLinf pi05 adjust_bottle. Fresh student/Q from the original base; FQL-fix alpha30 control with minimum target twin-Q and globally normalized actor Q loss. Source: `582bdb8f7` on `codex/fql-v2-20261101`.

- CPU: 15 tests, including two-process Gloo normalization, original behavior, Bellman terminal handling, replay contract and Q calibration.
- GPU: SZ1 physical2/3, 224/rank, global448. Twelve updates -> full resume to14 -> native N1 export ->64 closed-loop episodes, 44 successful. No video or signal sampling. Q recording leaves actions/RNG unchanged.
- Peak allocated/reserved:72.92/77.38GiB per GPU; median8.80s/update. Same normalization scale across ranks; no denominator-floor hits.
- Formal: queued behind the live SZ1 GPUs4/5 GAD MFD owner, then existing RLT.200 budget epochs =2200 updates,17 evaluations of128 episodes, each GPU32 environments with two waves. Fresh formal training had not started at this receipt.
- Data output, latest complete large checkpoint only. Temporary smoke checkpoint/policy deleted after validation; small Q exports and all light records remain. No other-user process was stopped.

Files: [compact smoke result](smoke-result.json), [queue receipt](queue-receipt.json), [formal config](formal/fql_fixed_teacher.json), `training/` metrics and N1 export checks, `evaluation/g2,g3/` episode/Q records and initialization timings, `smoke/` logs and CPU checks. [Compressed copy](light-evidence.tar.gz) is a small convenience copy of the same smoke materials, not model weights.

Short-smoke success is an execution check, not evidence that the Q-overestimation problem is solved. Raw Q scores are retained alongside actual discounted returns. Scientific background, unchanged settings and exact formulas: [implementation note](../../small_flow_distillation/42_FQL_V2_IMPLEMENTATION_20261101.md).
