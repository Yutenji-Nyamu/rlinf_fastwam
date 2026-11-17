# Complete-round K cost

Online success FM-BC, fixed update workload; turn_switch; 32 environments; 10 denoising steps; 4 updates of 128 samples. Three measured rounds per K after one warmup.

K4 versus K1: -99.70 s (-17.22%). Extra candidate sampling occupies 1.82% of the K4 round.

Synchronous measured wall time; module totals and full-device memory readings retained. Initialization and optional evaluation/checkpointing are outside training rounds.
