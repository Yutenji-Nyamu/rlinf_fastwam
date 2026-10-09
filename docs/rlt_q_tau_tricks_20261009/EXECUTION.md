# Execution: RLT-Q U / Norm controls

- Existing U/Norm implementation: b1d2d8d55; current branch starts at 348005d5f.
- Added configurable chunk dropout and Q-alpha annealing; temperature remains 2.5 by default. Controls are independent and opt-in.
- CPU on SZ3, all GPUs hidden: 54 tests passed; Ruff check passed (Oct9). Covers the real actor loss and gradient, full/global vs microbatch equality, both signals, all three control modes, unit endpoints, private RNG, unchanged disabled resume contract, and rejection of incompatible resumes.

GPU smoke uses the already running SZ3 `lift_pot` Control and its complete Stage1 checkpoint at R2000. The old Q `adjust_bottle` Stage1 is absent on SZ3; these are integration checks, not task-matched outcome comparisons. Preserve the Control's seeds, task assets, action budget, model, optimizers, and actor Q/BC curriculum. Reduce only the existing smoke budgets: 2 rounds, N4, B32/MB16, pool4/critic2, 2 updates per training call. Both methods use tau2.5, dropout0.2, smoke-only alpha R1=1 -> R2=0; the reusable example retains R500.

- GPU6: U, namespace `rlt-q-ugrow-both-g6-smoke-20261009-v1`.
- GPU7: Norm, namespace `rlt-q-norm-both-g7-smoke-20261009-v1`.
- Control directory: `/data/chenyiteng/deployment-20261009/rlt-q-tau-tricks-v1`.
- Source: `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-q-tau-tricks-20261009`.
- Outputs: `/data/chenyiteng/results/rlinf-rlt/<namespace>/`.
- Command per card: `/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -B /data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops/driver.py --plan /data/chenyiteng/deployment-20261008/bc-signal-tau-v1/plan.json --request <control>/g{6,7}.json`, via the existing owner.launch helper and saved runtime/environment.json.
- Reuse each card's existing checked C/G scope. Stop at two rounds or error. No formal Q dispatch. Once both namespaces and cards release, restore the original RLT fallback per card using the latest complete checkpoint.

Measured GPU results and return receipt will be recorded after completion.
