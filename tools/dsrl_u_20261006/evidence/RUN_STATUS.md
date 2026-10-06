# DSRL π0.5 Clean / U: formal launch

Snapshot: 2026-10-06, approximately 13:28 China Standard Time. Both formal runs
have completed round 2/200 and continue collecting warmup data. Clean has 142
resident chunks; U has 160. The inherited warmup threshold is 500, so this snapshot
does not claim formal optimizer updates or a learning improvement. The displayed
warmup ETA excludes later optimizer/evaluation costs and is not a completion forecast.

| Setting | Both runs |
|---|---|
| Task/model | adjust_bottle / Sidney π0.5, matching mean/std normalization and transforms |
| Resources | GPU6 Clean, GPU7 U, one GPU per experiment |
| Environments | 4 training and 4 evaluation environments per experiment |
| Actions | predict50, execute10, physical14 coordinates, latent32 |
| Solver | main ODE10; U adds a complete ODE5 with the same full cast latent/noise |
| Replay/training | capacity25000, warmup500, GB256, MB256, UTD20 |
| Formal budget | 200 collection/update rounds, eval every13, save every65 plus final |

U is computed only during collection, saved with replay, and used to weight the
whole chunk actor loss `alpha*log_pi-Q`. The critic, alpha objective and uniform
replay sampling remain unchanged. No VLA solve is added during replay SGD.

Validation completed: 30 algorithm CPU tests, 7 operations CPU tests, real-model
Gaussian/learned-latent probe, both fresh environment smokes, and both new-process
checkpoint resumes. Each resume advanced updates320→480 and replay16→24, saved
CP3, exited zero, and released its namespace/process/GPU resources. The initial
checkpoint audit failure was an FSDP name-prefix mismatch in the auditor; all156
target-shadow values passed exact round-trip checks after its correction.

Observed fresh/resume GPU occupancy peaks were40040/40132MiB on GPU6/7 (10-second
sampling, not allocator high-water marks). Formal source/config hashes and real
UID/PID/start, Ray namespaces and compute/graphics bindings were rechecked. No
fatal training error appeared in this snapshot; about443GiB of data-disk space remained.

Training source stays frozen at `8bcd99a6df38a1700a988d8982bde9368bf18025`.
The publication commit only adds lightweight records; it does not change the
active checkout. Configs, model/norm/seed hashes and smoke evidence are in this
directory. Raw replay, model checkpoints and full resource logs stay on SZ1.

RLT on GPU6/7 was saved at CP25/CP300 and released before DSRL. The unique owner
prevents RLT from competing for these cards. After a formal run exits, it checks
the exact process identity, namespace and GPU cleanup, then restores that card's
original RLT configuration/checkpoint under a new run and namespace. Identity or
cleanup ambiguity requires attention instead of taking the GPU. No other user's
jobs or GPU4/5 were stopped by this task.

Evidence: [formal launch](formal-launch-v1.json), [progress and process audit](formal-progress-v1.json),
[sampled resources](resources-formal-v1.json), [Clean resolved config](clean-formal-resolved.yaml),
[U resolved config](u-formal-resolved.yaml).
