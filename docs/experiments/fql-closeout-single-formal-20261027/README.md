# FQL closeout and native single-task formal queue

The fixed-teacher multitask FQL adaptation deteriorated to 2/368 successful held-out episodes at update 1130. The last logged update before user-authorized closeout was 1290; the latest retained complete checkpoint is 1130. This is a failed result, not a successful FQL replication.

`fql-metrics.csv` records full rank-0 training dynamics; `summary.json` records complete evaluation points and transition auditing. Infrastructure paths and other users' information are excluded.

The native single-task successor uses the five previously smoke-tested adapters. CPU tests passed; the next real workload registered 4.95 seconds after the stop signal, with model loading measured separately. Full settings and method-specific caveats are in [the formal plan](../../small_flow_distillation/12_SINGLE_FORMAL_QUEUE_20261027.md), and diagnosis is in [the FQL analysis](../../pi05_distillation/70_FQL_FAILURE_ANALYSIS_20261027.md).
