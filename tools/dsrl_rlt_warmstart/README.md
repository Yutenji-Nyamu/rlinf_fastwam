# DSRL warmstart and clean RLT on RT2 / Dojo

This opt-in package applies source-hash-locked patches to two accepted exported
runtimes. The patch package is used because the deployed runtime and this public
checkout have different integration histories; it does not overwrite unrelated
core files in the checkout.

- `apply.py ROOT`: DSRL Gaussian-policy critic warmstart and compact noise groups.
- `apply_rlt.py ROOT`: Dojo registration and action adapter, token-only checkpoint
  loading, and optional fused Adam / metric batching / target EMA for RLT.
- `check_cpu.py`, `check_rlt_cpu.py BENCH`: targeted server CPU checks.
- `check_rlt_replay.py`: exclude the unexecuted bootstrap query from short
  rollouts, while preserving first-done termination and truncation bootstraps.
- `group_eval.py DEPLOYMENT`: one compact Gaussian-noise grouping screen.
- `stage1.py DEPLOYMENT`: cache the accepted deterministic frozen VLA prefixes,
  train the existing causal RLT token module, and save token + optimizer resume.
- `reference_eval.py DEPLOYMENT`: reference-action decode parity and 8 episodes.
- `rlt_bench_driver.py DEPLOYMENT`: baseline / combined / baseline-repeat timing
  on the same real smoke replay and complete model/optimizer/RNG state.

Formal DSRL: 2000 collected macro transitions, 2000 critic-only updates against
standard Gaussian next noise, then critic:actor 2:1. Warmup phase and completed
updates are included in the strict resume contract. No entropy bonus is used in
the warmup Q target. Gaussian collection is unchanged until critic pretraining
finishes. Grouped replay retains G×32 coordinates; only VLA execution expands to
50×32. Default G=0 and zero pretraining updates preserve legacy behavior.

Formal RLT: frozen pi0.5 and FP32 token/actor/critic. Stage1 retains all 50 task
demonstrations, 2000 optimizer steps, batch32 (4×8), AdamW 2.5e-5, warmup100 and
cosine decay. The existing token-only loss uses `train=False` for VLA extraction;
the cache stores its BF16 output bit-for-bit, without another quantization step.
Dojo uses 10 additional training-dataset episodes for reconstruction validation.
These are distinct from simulator evaluation layouts. Stage2 uses 2000 collected
macro transitions, 5000 warmup updates before student takeover, critic:actor2:1,
batch256, 5 updates per new transition, at most1600 per call. BC/Q weights retain
the accepted 7/0.05 → 2.5/0.45 schedule, with a10000-update ramp. Gamma0.999 is per
physics step. RT2 executes20 actions/query; Dojo50. Both use8 environments and
400 training rounds. Evaluation inherits the current benchmarks: RT2 uses12
episodes and Dojo32, every20 rounds.

`prepare_*` records resolved configs, commands, output paths, and ownership before
cutover. `cutover.py` registers exact UID/PID/start identities and real RLT fallback
before stopping predecessors; it also retargets the waiting FQL join. Phase errors
release the slot to that registered fallback. Never use global Ray/GPU resets.

Sources and transfer limits are recorded in `sources.json`. Measurements, known
issues, current deployment paths and formal readiness belong in the experiment
ledger, not in claims of algorithmic improvement.
