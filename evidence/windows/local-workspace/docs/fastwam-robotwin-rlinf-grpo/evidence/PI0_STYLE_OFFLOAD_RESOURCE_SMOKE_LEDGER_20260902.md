# Fast-WAM pi0-style offload resource smoke ledger

## Objective and authorization

On 2026-09-02, the user authorized a resource-only smoke on the original Fast-WAM
GPU pair 6/7. The training estimator and per-step data budget remain unchanged.

## First configuration

The source packet is the stopped Fast-WAM256 formal run:

- 32 train envs, `rollout_epoch=8`;
- 256 trajectories, 32 G8 groups, 2,048 query records per outer step;
- `global_batch_size=2048`, `micro_batch_size=2`, `update_epoch=2`;
- H32/C24/M10, fixed32/eval5, DCP;
- physical GPUs 6/7.

Only the phase offload flags change:

| field | formal value | smoke value |
|---|---:|---:|
| `env.train.enable_offload` | true | false |
| `env.eval.enable_offload` | true | false |
| `actor.enable_offload` | false | true |
| `rollout.enable_offload` | true | true |

The smoke is one real outer step in a new run-scoped directory. It tests whether
keeping the RoboTwin renderers alive removes the repeated SAPIEN/OIDN lifecycle
while actor phase-offload leaves enough GPU memory for the actor update.

## Authorized fallback

Only if the first configuration OOMs:

- reduce train envs from 32 to 16;
- increase `rollout_epoch` from 8 to 16;
- retain 256 trajectories, 32 G8 groups, 2,048 query records and the complete
  optimizer budget.

This changes concurrency and wall time, but not the amount of rollout data or
training presentations per outer step.

## Operations and result

### 32-env first attempt

- run: `/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-pi0style-offload-smoke1-2gpu32x8-g8-b2048-u2-m10-phys67-v1`
- the resolved leaf diff had no unexpected fields;
- all 8/8 rollout waves completed without a numerical or renderer fatal error;
- rollout-stage GPU peak was about 46,106 MiB per card;
- when the actor returned to GPU for the real update, GPU6/7 peaked at
  81,017/81,011 MiB and PyTorch raised `OutOfMemoryError`, with only about
  62.8 MiB free;
- host `MemAvailable` remained about 1.75 TiB or more, so this was a device-memory
  overlap peak rather than host-memory pressure;
- wrapper exited 255 and both target GPUs returned to zero allocation.

This confirms that actor phase-offload reduces the rollout-phase footprint but
does not make a 32-env resident renderer set fit together with the Fast-WAM actor
during update.

### 16-env fallback

Started after verifying zero old-namespace actors and empty GPU6/7:

- run: `/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-pi0style-offload-smoke1-2gpu16x16-g8-b2048-u2-m10-phys67-v1`
- packet: same run name under the `packets/` directory;
- wrapper: `1358520`;
- resolved unexpected diff: zero.

The authorized fallback changes only physical concurrency to 16 and
`rollout_epoch` to 16, preserving 256 trajectories and every training budget.

All 16/16 rollout waves completed. During the real actor update, GPU6/7
ultimately peaked at 75,356 MiB each, leaving about 6.1 GiB per card; no OOM or
fatal error was observed. The per-card process decomposition near that boundary
was approximately:

- resident EnvWorker: 11,142 MiB;
- actor: 63,386 MiB;
- rollout model: 790 MiB.

The complete one-step result was:

- wrapper `1358520` ran from 22:31:26 to 23:22:06 CST (50m40s) and exited 0;
- the final-step fixed-32 evaluation completed. Train `success_once` was
  `0.29296875` (75/256); eval `success_once=success_at_end=0.4375` (14/32);
- actor metrics were finite: approximate KL 0.0026, clip fraction 0.020 and
  gradient norm 6.308;
- during eval, each card used about 66,488 MiB: the same `EnvWorker` rose to
  about 27,266 MiB while rollout and actor used about 25,194 and 13,988 MiB.
  This directly shows train and eval renderer state coexisting in that worker;
- minimum host `MemAvailable` was 1,865,074,340 KiB and memory PSI stayed zero;
- DCP save produced two real shards (14,455,154,049 and 14,454,317,078 bytes)
  plus a 2,919,657-byte `.metadata` file under `global_step_1`;
- after natural exit, GPU6/7 were empty, the run namespace had zero named
  actors and the shared Ray head remained alive.

Conclusion: 32 resident train envs do not fit with the actor update; 16 train
envs do fit for one complete `rollout -> update -> fixed32 eval -> save` step
without changing the 256-trajectory/2,048-record budget. This one-step result
does **not** yet prove the next update after an eval will fit: eval increased the
resident `EnvWorker` allocation by about 16.1 GiB/card, and `max_steps=1` exits
before another actor update. No two-step run was started.
