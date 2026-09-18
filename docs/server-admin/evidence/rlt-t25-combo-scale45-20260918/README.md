# SZ1 GPU4/5 RLT switch

2026-09-18 19:26 CST: stopped GPU4/5 GRPO tau2, dropout0.2, alpha anneal200.
Last collected round158, retained checkpoint150. Archive includes all recorded
scalars, raw events, configs and lightweight logs; weights remain on server.

GPU5: RLT tau2.5, success_scale1, dropout0.2 plus both alpha R1=1 to R500=0.
GPU4: RLT tau2.5, success_scale2, no dropout or annealing.
Both fresh800, 4env per round, otherwise Clean4 budget and Stage1 CP2000.
Existing tested production source c42cbd50 reused; no algorithm changes.
GPU0-3 and GPU6-7 tasks preserved. Shared Ray was not restarted.
Free-to-submit: GPU5 0.143s, GPU4 0.289s; model loading follows.
