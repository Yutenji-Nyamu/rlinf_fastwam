# RLT DVAC controls

Optional controls on the full replay batch before microbatch splitting. They apply
only to extra successful-chunk BC weighting. Q loss and failure BC are unchanged.
Disabled controls preserve existing outputs and metrics.

`algorithm.rlt_dvac.chunk_dropout: {enabled: true, probability: 0.2, seed: 42}`
returns selected successful replay queries (all H positions together) to weight 1.
There is no inverse-probability scaling or renormalization. A private generator
uses the existing checkpointed update_step, without consuming training RNG.

`algorithm.rlt_dvac.alpha_schedule.enabled: true` enables independent `local` and
`chunk` schedules, each `{enabled: true, start_step: 1, end_step: 500, end_alpha: 0.0}`.
The index is the one-based runner round, including teacher collection, not Adam
updates. R1 starts at configured alpha; R500 reaches exactly zero and stays zero.
Temperature remains fixed. With success_scale=1, both alpha=0 recovers Clean BC.
Existing strict resume configuration guards reject changed controls.

CPU validation: 94 tests covering endpoints, Bernoulli rate, RNG preservation,
failure rows, actual worker/microbatch gradients, resume, and temperature baseline.
GPU smoke skipped by explicit user request on 2026-09-18.
