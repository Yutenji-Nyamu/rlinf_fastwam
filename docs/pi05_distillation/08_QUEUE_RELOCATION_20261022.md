# Checkpoint-preserving queue relocation

The selected per-rank batch remains 64 (global 128). The queue now takes its physical GPU pair from the deployment record and can continue an existing complete training checkpoint: it evaluates an unevaluated saved step before training the next segment. Smoke evidence remains associated with the original measured pair.

CP uses 19,000 optimizer updates, evaluating every 1,900 updates; OneDP-S uses 760 updates, evaluating every 190. Each evaluation uses 128 episodes, 32 environments per GPU and two batches per GPU, N1/H50/C50 with the official 400-action limit. Both methods continue from full optimizer and RNG state after evaluation. The completed original-policy baselines are N1 64/128 and N10 74/128.

Validation: CPU queue test covers continuation from CP step 1900, evaluation before further updates, no repeated first training segment, all ten CP and four OneDP evaluations, and the relocated physical pair 6/7. Existing batch, resume and native N1 inference smoke results are reused. The relocation is scheduled at the running segment's next complete checkpoint; this document does not claim the target training has already started.
