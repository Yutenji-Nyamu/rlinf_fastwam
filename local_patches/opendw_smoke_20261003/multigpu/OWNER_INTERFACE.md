# Four-card smoke owner interface

This is the isolated four-card implementation. Server CPU checks passed on
2026-10-03; the GPU suite started at 23:31 and remains subject to live receipts.
It does not overwrite or import the old single-card owner.
Root owns the separate lifecycle wrapper and deployment; this file creates no
borrow/return CLI, private Ray instance, or scheduler replacement.

## Frozen run contract

- `mode: multigpu_smoke`; physical GPUs `[4,5,6,7]`.
- Actor and rollout placement `4,5`, each world size 2, visible devices `[['4'],['5']]`.
- Environment placement `6,7`, world size 2, visible devices `[['6'],['7']]`.
- Both trials use N64/G8/R8, C32/H50, microbatch 8, update epochs 2.
- First trial: L32, global batch 512, 512 collected chunks, 2 scheduled optimizer steps.
- Second trial: L384, global batch 2048, 6144 collected chunk slots, 6 scheduled optimizer steps.
- One runner iteration, save 1, validation disabled, no resume checkpoint.
- GRPO, relative reward, filter `[0.1,0.9]`, and threshold 0.9 remain unchanged.
- Both WM services load on CPU before borrowing. Service 0 is physical GPU6;
  service 1 is physical GPU7. The train `service_urls` list uses that same order.
- Per-trial cleanup and dual service offload happen between trials. A single
  full return follows the entire suite, or exception/timeout cleanup.
- Exit 0 does not prove a nonzero policy update or native task improvement.

## Plan fields

The existing owner fields remain: `owner_dir`, `repo`, `repo_head`, `python`,
`environment_file`, `source_sha256`, `physical_gpus`, `trials`, and optional
`restore_wait_seconds` (defaults to 60 and at most 60).

Replace the old `cycle_dir` and singular `service` fields with:

```json
{
  "mode": "multigpu_smoke",
  "lifecycle_module": "/data/chenyiteng/.../four_card_lifecycle.py",
  "lifecycle_path": "/data/chenyiteng/.../four_card_transaction",
  "services": [
    {
      "key": "wm6",
      "physical_gpu": 6,
      "url": "http://127.0.0.1:18946",
      "argv": ["/data/chenyiteng/.../bin/python", "-u", "-B", "/data/chenyiteng/.../opendw_service.py", "--physical-gpu", "6", "--port", "18946", "--output-dir", "/data/chenyiteng/.../owner/services/wm6/artifacts"],
      "cwd": "/data/chenyiteng/.../service",
      "startup_seconds": 1200
    },
    {
      "key": "wm7",
      "physical_gpu": 7,
      "url": "http://127.0.0.1:18947",
      "argv": ["/data/chenyiteng/.../bin/python", "-u", "-B", "/data/chenyiteng/.../opendw_service.py", "--physical-gpu", "7", "--port", "18947", "--output-dir", "/data/chenyiteng/.../owner/services/wm7/artifacts"],
      "cwd": "/data/chenyiteng/.../service",
      "startup_seconds": 1200
    }
  ]
}
```

The example paths and service argv are placeholders, not a launchable plan.
The actual argv must include existing reviewed model/reward/T5 flags. Every
service output directory must be under its own `owner_dir/services/<key>/`.
Freeze all service code/dependencies and the lifecycle module in `source_sha256`.
The lifecycle module hash is verified before importing it. The base environment
and per-service additions must not contain GPU visibility masks; the owner adds
each service's numeric single-card CUDA mask. Drivers retain no CUDA mask so
RLinf can discover its requested physical placements.

Each trial specifies a unique `key` and `opendw_` namespace, a config path,
`num_envs: 64`, `episode_steps: 32` or `384`, and positive `timeout_seconds` no greater than 10800. Its config
logger path must be beneath `owner_dir/<key>/`. At most two explicitly listed
trials are accepted in the exact order L32 then L384; both are required. L384
starts only after L32 exits 0, has placement/completion proofs, and both services
acknowledge offload. There is no automatic retry or repeated training loop.

## Lifecycle module contract

Loading `C.install_helper(path)` provides `C.H`. The module supplies:

- `C.load_plan(path)`: immutable transaction plan with `python`, `ray_address`,
  `ray_dashboard_url`. Its `plan.json` hash must remain unchanged after stopping.
- `C.stop(path)`: exact four-card stop; on successful return, a complete
  top-level `rlt-stopped.json` must exist. The owner holds `operation.lock`.
- `H.resume(path, release_receipt)`: dispatch recovery of all four original
  runs. Called only after all owner processes are stopped and **all compute and
  graphics contexts on GPUs4–7 are absent**. The owner never kills an unknown
  process to make that condition pass.
- `H.status(path)`: `runs` must contain exactly `gpu4`, `gpu5`, `gpu6`, `gpu7`;
  `all_first_rounds_verified` comes from each original run's verified recovery.
  The owner waits at most 60 seconds. A healthy dispatch with a slower first round
  writes `rlt-first-round-pending.json` and retains an explicit pending outcome;
  this does not turn a completed WM test into a failure or a verified RLT round.
  A returned process that already exited still produces a recovery error.
- `C.recover_partial(path, release_receipt)`: invoked only when a top-level
  `clean-old-stop-attempt.json` exists without complete `rlt-stopped.json`.
  Its receipt proves only that all owned smoke processes stopped, and includes
  `cycle_id`, requested `gpus`, original `terminal_status`, and process identities.
  It does **not** claim those GPUs are empty. The wrapper returns only child
  runs with complete stopped proofs, keeps untouched children unchanged, and
  rejects ambiguous state. A child may finalize an existing exact stopped
  proof without new signals. The owner records the result but keeps full
  `rlt_borrowed=false` and `rlt_first_round_verified=false`.

The helper also retains existing `proc`, `same`, `now`, `save`, `atomic`, `config`,
`actors`, `active`, `validate_actor_rows`, `kill_actors`, and `gpu_processes` APIs.
Signals use pinned PID/start-time/UID identities and pidfds. Ray actor cleanup
requires this exact namespace and driver job receipt; shared Ray stays alive.

## Checks and evidence

`test_opendw_multigpu_owner.py` contains CPU-only fixtures for scale/route drift,
outside-card or wrong-service contexts, import hash rejection, PID reuse, both
services ready before borrowing, return once after the complete suite, failed
readiness without borrowing, partial recovery, and full-return empty-card gates.
Run it with the server's existing Python. Local work only parsed syntax.

Evidence stays under the owner directory: per-service health/log/artifacts,
per-trial job/placement/driver/cleanup/results, resources, exact process catalog,
complete or partial release, RLT recovery status, and final outcome. Partial
return never produces the complete `smoke-release.json` or a full-return claim.
