# Experiment update · 2026-10-27

Direct distillation C10/H50: checkpoint 22260 achieved 130/368 successes (35.33%), compared with the previous best of 31.52%. Original N10: 153/368 (41.58%); original N1: 75/368 (20.38%). Each complete point includes 46 valid tasks × 8 episodes; four tasks are missing.

Fixed-teacher FQL: checkpoint 565 achieved 6/368 (1.63%); checkpoint 1130 evaluation was incomplete at this update.

Five single-task methods completed functional save/resume/export/rollout smoke checks. Their short-run weights and previously approved obsolete checkpoints were removed after user authorization; small logs and curves remain. This does not remove current training recovery points or protected evaluation weights.

Detailed infrastructure and storage ownership records are retained locally, outside this public report.
