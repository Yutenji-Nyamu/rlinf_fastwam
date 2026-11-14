# Cumulative 400-round continuation and short RT2 RLT queue

The current four experiments keep their learning, rollout and evaluation settings. Their active processes finish the existing segment (80 rounds on slots 4, 5 and 7; 64 on slot 6), then resume from a full checkpoint to cumulative round 400.

| Slot | Following RT2 RLT task | Episode limit | Rounds | Train / eval parallelism |
|---|---|---:|---:|---:|
| 4 | `scan_object` | 200 | 400 | 48 / 48 |
| 5 | `place_bread_skillet` | 200 | 400 | 48 / 48 |
| 6 | `move_stapler_pad` | 200 | 400 | 48 / 48 |
| 7 | `place_dual_shoes` | 200 | 400 | 48 / 48 |

The new RLT tasks reuse the successful RT2 settings: chunk 10, batch 512, 20,000 warmup transitions and 30,000 post-collection critic updates. Each task uses its own 50 Clean demonstrations for 2,000 token reconstruction updates, followed by a frozen-reference evaluation and online training. Evaluation uses one 48-episode batch. Train and eval environments are unloaded sequentially through `process_offload`.

`wait_owner.py` is a CPU-only prerequisite waiter around the existing experiment owners. Operator-local `plan.json`, `wait.json` and environment files specify the source tree, process identities, checkpoints, queue and output directories. The operator prepares and validates all deployments, starts unregistered waiters, atomically reserves the slots, then registers them. A waiter requires both predecessor exit and cleanup before starting. No running trainer is interrupted during preparation.

Checkpoint handling distinguishes DSRL local shards from RLT distributed checkpoints. A fixed entropy coefficient has no entropy optimizer checkpoint; trainable coefficients require their saved optimizer state. An incomplete save marker does not qualify as a full checkpoint. If a segment exits early, continuation selects its latest complete save. Existing emergency RLT recovery remains available after owned work releases the card.

`next_rlt.patch` contains the narrow owner/evaluation adaptations on top of the successful RT2 source. The renderer fix is documented in `../robotwin_process_offload`. The dataset, model assets, deployment-specific files and checkpoints stay on the server.

Validation on 2026-11-14: all eight production configurations passed the normal CPU validation path; each new task's first RGB/state sample decoded correctly; all four current full-state checkpoint layouts were recognized. The prerequisite-ordering and checkpoint-format tests passed. GPU execution of the new tasks occurs when their reserved slots become available.

Run CPU tests with `python -m unittest discover -s tools/experiment_queue_20261114 -p 'test_*.py'`.
