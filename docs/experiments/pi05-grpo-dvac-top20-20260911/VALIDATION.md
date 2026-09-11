# CPU validation receipt — 2026-09-11

Base: `1d015a2aa03ec8132d8207ba47a2be3dbe1d9591`. Branch: `codex/sz-pi05-grpo-dvac-top20-20260911`.

The targeted server test run passed **30 tests in 6.75s** with
`CUDA_VISIBLE_DEVICES=''`. Syntax and `git diff --check` passed. The inherited
resolved configuration was composed in memory with the method-only overlay;
all non-method leaves matched the archived target baseline.

See [test output](evidence/tests.log), [test environment](evidence/test-receipt.json),
[reviewed source hashes](evidence/source-manifest.json), and
[configuration contract](evidence/config-contract.json).

The GRPO suite combines actual native PPO gradient checks and two-process CPU/Gloo
checks with an actor-loop control-flow harness; it does not load the full model or
run real FSDP. SARM uses CPU tensors/Adam and mocks GPU synchronization.
No GPU smoke, model training, CUDA/FSDP restore, Ray service operation or experiment
reconfiguration was performed. Real GPU integration and performance remain untested.
