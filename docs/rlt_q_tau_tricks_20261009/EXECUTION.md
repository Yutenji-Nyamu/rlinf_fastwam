# Execution: RLT-Q U / Norm controls

- Existing U/Norm implementation: b1d2d8d55; current branch starts at 348005d5f.
- Added configurable chunk dropout and Q-alpha annealing; temperature remains 2.5 by default. Controls are independent and opt-in.
- CPU on SZ3, all GPUs hidden: 54 tests passed; Ruff check passed (Oct9). Covers the real actor loss and gradient, full/global vs microbatch equality, both signals, all three control modes, unit endpoints, private RNG, unchanged disabled resume contract, and rejection of incompatible resumes.

GPU smoke uses the existing SZ3 `lift_pot` Control and its complete Stage1 checkpoint at R2000. The old Q `adjust_bottle` Stage1 is absent on SZ3; these are integration checks, not task-matched outcome comparisons. Preserve the Control's seeds, task assets, action budget, model, optimizers, and actor Q/BC curriculum. Disable evaluation (runner.val_check_interval=-1), then reduce the existing smoke budgets: 1 round, N4, B32/MB16, pool4/initial2, 2 updates per training call. Both methods use tau2.5, dropout0.2, alpha R1=1 -> R500=0; the endpoint is covered by CPU tests.

- GPU6: U, namespace `rlt-q-ugrow-both-g6-smoke-20261009-v2`.
- GPU7: Norm, namespace `rlt-q-norm-both-g7-smoke-20261009-v2`.
- Control directory: `/data/chenyiteng/deployment-20261009/rlt-q-tau-tricks-v1`.
- Source: `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-q-tau-tricks-20261009`.
- Outputs: `/data/chenyiteng/results/rlinf-rlt/<namespace>/`.
- Command per card: `/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -B /data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops/driver.py --plan /data/chenyiteng/deployment-20261008/bc-signal-tau-v1/plan.json --request <control>/g{6,7}-v2.json`, via the existing owner.launch helper and saved runtime/environment.json.
- Reuse each card's existing checked C/G scope. Stop at one round or error. No formal Q dispatch. Once both namespaces and cards release, restore the original RLT fallback per card using the latest complete checkpoint.

The first dispatch inherited five evaluation batches from the formal Control. It completed its first real update, then was stopped to avoid that unintended evaluation cost. Both exact namespaces and C/G contexts were released. The corrected v2 runs one complete training/update/checkpoint round with evaluation disabled; training source is unchanged; v2 uses the example R500 schedule and retains tau2.5/dropout0.2. The CPU checks already cover the annealing endpoint; the final GPU check does not repeat that simulation budget.


## Final GPU acceptance

Both corrected runs exited 0 with cleanup_error=null. Each collected 80 real teacher-signal chunks, completed its actor Q update, saved global_step_1, and released its exact namespace plus compute/graphics contexts. All recorded losses and gradients are finite. No official evaluation or formal Q training was run by v2.

| Signal / GPU | Q weight std | Dropout fraction | Actor grad norm | Actor loss | Critic loss |
|---|---:|---:|---:|---:|---:|
| U / 6 | 0.0838654 | 0.21875 | 13.5682 | 15.7071 | 0.0312682 |
| Norm / 7 | 0.129282 | 0.21875 | 14.3633 | 17.6706 | 0.0335873 |

The configured dropout probability is 0.2; 7 of the sampled 32 rows were restored to unit Q weight in each run. The private RNG gives the same mask at the same seed/update counter. The first-round alpha is 1; its interpolation and unit-weight endpoint were verified on CPU. Rollout success was 0/4 for both; this is an integration check, not evidence of training benefit.


Each completion manifest records saved_runner_step=1 and update_step=2. The verified training source is bd3f8d45b24a1d319a8a8f6ab33db06ab5a945b5; SHA256 of all three changed Python files matches the server copy. See smoke-passed.json, cpu-checks.json and the two resolved smoke YAML files.


## Card return

At 19:25 SZ3 both fallback slots were RLT_RUNNING under the original per-card queue. GPU6 resumes the original lift_pot Clean and GPU7 the original Combo from complete R150 checkpoints. New CPU owner: 2608312; exact drivers: 2608321 / 2608322. Their compute contexts were observed only on their respective cards while the original checkpoints loaded. This check does not claim a completed new fallback training round. The 4/5 WM plan entries were preserved, and shared Ray was not restarted. The new fallback requests are <control>/fallback-g6.json and fallback-g7.json; the live queue is still /data/chenyiteng/deployment-20261008/bc-signal-tau-v1/plan.json.

Return identities, original checkpoint paths, and exact Q release proof are in returned.json and return-verified.json. Q formal training was never dispatched.
