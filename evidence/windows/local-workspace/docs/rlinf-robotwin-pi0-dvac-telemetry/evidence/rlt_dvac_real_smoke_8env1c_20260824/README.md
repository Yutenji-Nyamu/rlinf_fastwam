# RLT teacher-DVAC real smoke evidence (2026-08-24)

## Outcome

- Contract: successful RLT Stage-2 topology (2 A800, 8 train env, C10 student) with
  teacher-DVAC L2/L3/L4, selected L3 and global-z weights in `[0,2]`.
- Lifecycle: `2026-08-24T17:18:19+08:00` to `17:52:06+08:00`, exit `255`.
- The first 8-env rollout completed in `119.15 s`. Both actor ranks then produced
  DVAC traces for update steps `0,2,4,6`, so collection, baseline freeze and the
  actual weighted actor-Q path were entered.
- The run did **not** reach evaluation or `global_step_1` checkpoint. At 17:52 the
  30-minute NCCL watchdog terminated a mismatched collective:
  rank 1 was waiting at sequence 44, `ALLREDUCE Numel=105`, while rank 0 had
  advanced to sequence 45, `ALLREDUCE Numel=1` (the final barrier).

## Narrow root-cause evidence

`process_train_metrics()` sorts each rank's local metric dictionary, converts its
values into one tensor, then calls `all_reduce`. The new
`_rlt_dvac_context_metrics()` conditionally adds metric keys according to whether
that rank-local replay batch contains positive/non-positive rewards or
student/reference routes. Consequently, two ranks can produce different tensor
lengths for the same collective. The observed `105`-element reduction followed by
the other rank's one-element barrier is the exact failure pattern expected from
that schema divergence.

The scientific weighting formula and RLT objective do not need to change for this
failure. The narrow repair is to make the distributed telemetry schema fixed on
all ranks (prefer fixed sum/count fields for conditional groups), then repeat this
same smoke.

## Method trace summary

Across the eight compressed traces (320 `query × h` weights):

- all numeric arrays are finite;
- `teacher_dvac_v`: `[4,3,50]` per trace;
- selected L3 variances / weights: `[4,10]`;
- student action: `[4,10,14]`;
- weight min / p05 / median / mean / p95 / max:
  `0 / 0.2584 / 0.9810 / 1.0247 / 1.8670 / 2`;
- per-query weight ESS mean: `0.8891`; top-20% weight mass mean: `0.2904`;
- 0-weight / 2-weight fractions: `0.94% / 3.44%`;
- all 32 sampled `actor_switch` values are false because this one-cycle smoke
  collected its transitions on the reference route.

## Resource summary

- 899 two-second samples; monitored interval `2029 s`.
- max GPU memory: `17,111 / 17,193 MiB`; max utilization `100% / 100%`.
- max cgroup memory: `54.68 GiB`; minimum host available memory: `910.64 GiB`.
- `memory.events` deltas: high/max/OOM/OOM-kill all zero.
- After exit, driver and monitor were gone, no matching worker remained, and both
  GPUs returned to `0 MiB`.

## Files

- `driver.log`, `ray_actor_rank*.err`: full failure evidence.
- `resolved.yaml`, `exact_command.txt`, provenance and preflight/unit-test logs:
  reproducible launch contract.
- `resources.csv`: full two-second resource trace.
- `actor_rank*_update_*.npz`: the eight method traces.

The directory is about 0.30 MiB and intentionally excludes checkpoints, videos and
bulk Ray logs. No checkpoint exists for this failed smoke.
