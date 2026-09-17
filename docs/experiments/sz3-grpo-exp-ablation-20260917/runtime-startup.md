# sz3 verified formal startup

The two formal GRPO128 runs were launched from `e03c9075986179873fc5d3fc61f61b9817cdb456`. Both completed a full first round, including finite positive optimizer gradients. The resolved training config matches the frozen config; each run has 15 actors on its assigned physical GPU pair and rollout seeds 42/43.

- `t15-drop02-anneal200`: physical GPU 4,5, both tau=1.5; dropout 0.2 and both alpha R1=1 to R200=0.
- `t3-notricks`: physical GPU 6,7, both tau=3; dropout and annealing disabled.

The budget remains 128 trajectories/round, G8, B512/micro32, U2, lr5e-6, fresh initial weights, 200 rounds, fixed32 evaluation every5 and checkpoints every10. Direct personal Ray is used; no Slurm configuration changes were made.

Ray 2.57 returns ActorState dataclasses. The next-launch tooling now uses `dataclasses.asdict` for actor queries and cleanup. This publication used a detached worktree: the active source checkout, its local branch, its frozen HEAD and all 1181 baseline source files remain unchanged. Any future fresh launch from the published code must regenerate its contract for that HEAD.

Current drivers retain the original cleanup implementation. A companion exit guard is already installed and verified for each exact driver/job/namespace. It waits for the driver and wrapper to exit, then cleans only verified remaining actors of that job; it never stops Ray. Raw driver/wrapper exit records are preserved even if the original cleanup returns an error. Guard installation is startup evidence, not evidence of a completed future cleanup.

`first-round-evidence.json` contains selected metrics, method differences, changed config field names and guard receipts. Private environments, full actor command lines and large source hash inventories are retained only on the server.
