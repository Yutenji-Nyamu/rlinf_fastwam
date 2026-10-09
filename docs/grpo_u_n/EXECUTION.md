# GRPO U / Norm execution

Updated2026-10-09 21:32 China. Implementation e04e9d4fbc041519019f7e642765f157ffb78553 is pushed to codex/grpo-u-n-20261009. U and Norm passed their complete one-round smokes. Original GPU6/7 RLT was restored and verified at21:32. No formal GRPO training was launched.

## Fixed configuration

Control: original pi0.5 turn_switch Clean256. Each smoke keeps64 environments x4 rollout epochs =256 trajectories, group8, update_epoch2, globalB512/micro32, H50/D14, 200 physical actions, seed42 and native Flow-SDE noise0.5. Enable local/chunk tau2.5, chunk dropout0.2 and the existing alpha1-to0 schedule (R1-toR200). Stop after one complete round; evaluation/checkpoint intervals are disabled for smoke. Source/config/output and physical GPU6/7 placement are the only other deployment changes.

Stop conditions: one complete round, nonfinite losses or weights, scope escape, or failed process. There is no performance threshold. Successful release requires the exact driver and Ray namespace to be gone and both compute and graphics contexts to clear.

## Validation and current results

[Server CPU tests](evidence/cpu-tests.txt): 196 passed in114.94s, CUDA disabled. Suites cover signal/sampler action and RNG parity, hooks, actual actor grouping/gradients, action-advantage losses, controls/annealing, exponential mapping and resume. Ruff0.14.3 preview lint/format and git diff checks passed. Core AST after formatting equals the server-tested source.

Both [U](evidence/u-smoke.json) and [Norm](evidence/norm-smoke.json) completed256 trajectories and2 actor-update epochs. Per-rank artifacts[4,128,50] are finite and nonconstant, with nonuniform weights. Both dropped141/715 eligible chunks (19.72%); all dropped chunk weights are exactly1. Both exited0 with cleanup_error=null; exact namespaces and GPU6/7 compute and graphics contexts were clear before return.

| Metric | U | Norm |
|---|---:|---:|
| Weight min-max | 0.61510-1.64441 | 0.59534-1.62188 |
| Global weight mean | 0.99754 | 1.00020 |
| Gradient norm | 24.64157 | 24.38032 |
| Total loss | 0.00015504 | 0.00027288 |
| Actor training seconds | 25.93 | 25.42 |
| Initial-policy rollout successes | 111/256 | 111/256 |

These success counts precede parameter updates and are smoke telemetry, not post-training evaluations. U elapsed time includes the storage stall; these runs do not establish method speed or learning quality.

Norm attempt1 exited before creating training workers because the storage switch initially missed the original Ray socket inode. Its failed-before-workers.json preserves the failure. The socket subtree was retained and new Ray connections verified; attempt2 uses identical source and budget with a new namespace/output. The final receipt passed at21:22, including renderer UUID/CUDA checks restricted to attempt2.

## Artifacts and return

Control: /data/chenyiteng/deployment-20261009/grpo-un-smoke-g67-v1. It stores plan.json, lane requests, resolved runtime configs, prelaunch scope proof, completion/release and smoke validation receipts.

Run root: /data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/.

- U: grpo-u-turn256-t25-tricks-smoke1-g67-20261009-v1.
- Norm: grpo-norm-turn256-t25-tricks-smoke1-g67-20261009-v2.

[Return receipt](evidence/rlt-return.json): original GPU6 Clean/GPU7 Combo both loaded complete DCP step325 and advanced to326; both began the next rollout. Max3000, original configs and requests remain unchanged. Both C/G bindings match their individual GPU; there were no scope escapes. Return owner PID1324434/start435623387 has a live heartbeat; original4/5 owner and its plan are unchanged.

Authoritative6/7 control is /data/chenyiteng/deployment-20261009/grpo-un-smoke-g67-v1/return-rlt/plan.json and status.json. The parent control has rlt-return-dispatched.json and rlt-return-verified.json. Original requests are /data/chenyiteng/deployment-20261008/bc-signal-tau-v1/requests/rlt-g6.json and rlt-g7.json. Do not dispatch them again or add6/7 to a competing owner.

[Storage incident and retained mounts](STORAGE_INCIDENT.md) records the authorized online2.33.5 recovery and Ray socket correction. Old inode domains remain intentionally retained until their users release them.
