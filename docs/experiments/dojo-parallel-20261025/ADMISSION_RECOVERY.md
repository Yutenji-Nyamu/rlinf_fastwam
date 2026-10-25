# Recover a task admission failure without aborting sibling work

At 20:36 Asia/Shanghai on 2026-10-25, the private six-lane queue exited after an initial GPU ownership assertion in `fill_pen_holder-c50-n1`. That assertion ran before the task's guarded lifecycle, so no result/clearance receipt was produced, and the queue reported `Incomplete private cleanup`. The task had not launched its simulator. The original assertion did not log the offending PID; an exited snapshot PID is a plausible race, not a proven diagnosis.

Move admission into the guarded lifecycle, re-read process identities, ignore exited snapshot PIDs, and keep live unowned processes as blockers. Admission failures now produce the normal task result and exact private cleanup receipt. The bounded queue recovery also recognizes missing-clearance failures after verifying that its designated GPU is clear; it archives orphan configurations and interrupted jobs before retrying them. Completed and ordinarily failed results stay intact.

Validation on the server: three focused tests passed, including stale PID/resident/live foreign-owner admission cases. Recovery retained 47 finished condition records, 140 valid episodes and 21 successes. At 21:45, all six lanes were producing actions with 24 environments on physical GPU3; memory was 79,352 MiB and the utilization sample was 100%. The two distillation runs on GPUs4–7 kept advancing. GPU3 remains scheduled for release when the scan finishes; RLT fallbacks remain only on GPUs4–7.

No task budgets, seeds, policy numerics, success criteria, or other users' processes were changed. A live unowned process still prevents admission, and recovery will not proceed while the designated GPU has remaining processes.
