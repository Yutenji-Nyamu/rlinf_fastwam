# Execution

CPU validation: 196 tests passed in 114.94 s on SZ2 (CUDA disabled): signal/sampler, actual actor, action-advantage loss, linear controls and exp mapper suites. Ruff 0.14.3 preview lint, format and git diff checks passed. Core AST after formatting equals the tested version. GPU smoke preparation is in progress; no formal run is authorized.

Two sequential one-round smokes on SZ2 physical GPU6/7: U then Norm. Inherit complete256-trajectory training round and loss budget; enable tau2.5/drop0.2/alpha controls. Disable only periodic evaluation/large checkpoint saves for smoke. Independent namespaces/outputs, original SFT.

Before launch record resolved config, command, output, source digest, compute+graphics binding and fallback requests. Stop on one complete round, nonfinite loss/weights, scope escape or failed process. No accuracy threshold. After both jobs exit verify exact namespaces and C/G release, then restore original per-card RLT with existing checkpoints and owner/driver. No formal U/Norm GRPO launch.
