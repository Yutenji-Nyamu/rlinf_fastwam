# RLT U smoke v2

One additional fixed smoke is allowed to cover naturally successful teacher
episodes. This is a separate attempt; v1's failure and return receipts stay
unchanged. Do not repeat attempts to search for a favorable result.

Call `revised, changes = revise_smoke(v1_actual_smoke_config)`. The helper only
changes:

| Dotted path | v1 | v2 |
| --- | ---: | ---: |
| `runner.max_steps` | 2 | 4 |
| `runner.save_interval` | 2 | 4 |
| `algorithm.rlt_schedule.warmup_post_collect_updates` | 2 | 8 |

N4, batch32/micro16, at most two critic updates per round, U1, 200 actions,
Stage1 CP2000, seeds, evaluator and weighting method remain unchanged. The
four training collections begin below learner update 8, so each uses the
teacher. Evaluation still measures the fresh student and does not populate
successful training replay. No seed is selected or relabeled.

The deployment caller separately assigns a new output identity, namespace,
owner and GPU5 lease. Accept only real successful query coverage, nontrivial
applied U weights, finite updates, complete save/exit and exact GPU release.
If this single bounded attempt lacks coverage or fails, retain its evidence
and return the card; no unbounded extension or repeated seed search follows.

Formal is unchanged: fresh Stage2 from the same original complete Stage1,
N4/800 rounds/B512/micro256/U5, 10k initial replay and 15k initialization
updates. Formal never resumes a smoke checkpoint or imports its replay.
