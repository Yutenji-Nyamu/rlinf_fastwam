# DSRL π0.5 Clean / U, Shenzhen 1

This opt-in experiment derives from the completed `adjust_bottle` DSRL recipe.
The inspected server source was clean at
`4ec52bd8d3dad1744b3f61c76636314b9d69c7a6` (historical port base
`7d07a4212ee6858cc333e1d4fab7a37256d1f839`).

## Fixed comparison

- GPU6 Clean and GPU7 U, one actor/env/rollout rank per experiment. Each retains
  four training and four evaluation environments.
- Sidney π0.5 checkpoint and its mean/std normalization and action transforms;
  prediction H50, physical D14, DSRL latent32, solver ODE10, executed prefix C10.
- Original method parameters: global batch256, replay25000, warmup500, UTD20,
  gamma.999, ten mean-aggregated Q heads, unchanged SAC critic/alpha/target and
  uniform replay. Formal budget200, evaluation13, save65 plus final save.
- Microbatch64 initially; 128/256 can be measured on both roles without changing
  global batch, environment totals, UTD or the formal budget.

`make_config.py` only writes fresh run input files, never starts jobs. It copies
`provenance/dsrl_pi0_formal_200.yaml` (SHA256
`a7f13ca6bccf346a562a2c815a5ae37869603df87f1f05600f666d3c56b91582`),
records every changed leaf and output hash, and distinguishes reduced smoke
settings from the formal recipe. Final composed configs and pair differences
must be archived before launch.

## U contract

The rollout reuses its sampled latent, full cast initial noise and prefix cache.
It retains the normal complete ODE10 action and adds one complete ODE5 solve.
Per-coordinate population std divided by RMS plus 1e-8, averaged over fourteen
physical coordinates, produces each submitted action's U. The first ten scores
and validity/policy-version/phase accompany the original transition mask into
replay. No VLA solve is added to replay SGD.

Before microbatch splitting, the entire sampled global batch maps mean(log(U+
1e-12)) through minmax and `exp(scaled / 2.5)`, normalized to mean one. Detached
chunk weights multiply the complete actor `alpha * log_pi - Q` objective. All
valid chunks participate, including failures. Critic TD, alpha fitting, rewards,
discounts and sampling probabilities are unchanged. No action-local weighting,
success filter, dropout or annealing is introduced.

Enable both `actor.model.openpi.dsrl_u_enabled/dsrl_u_spec` and
`algorithm.dsrl_u.enabled/spec`; use replay schema2 for U. Clean defaults remain
legacy schema1. The helper is `rlinf.algorithms.dsrl_ugrow`; U replay and trainer
checkpoints reject incompatible signal definitions or weighting contracts.

## Evidence and launch boundary

At the p009 checkpoint, four CPU test modules passed 30 tests on the server:
`test_dsrl_ugrow`, `test_dsrl_u_actor`, `test_dsrl_transition_replay`, and
`test_dsrl_target_shadow_resume`. Real GPU sampling, single-card memory peaks,
fresh/resume smoke and formal training were not yet validated at that checkpoint.

The final provenance manifest must bind the modified source commit, resolved
config, model/conversion manifest, model weight hash, normalization hash and both
seed-table hashes. The p012 server snapshot records:

| File | SHA256 |
|---|---|
| Sidney model.safetensors | `a875898babcb06bd767f09b10ebbf8aa550e68ad28ee7e99cb103ccfe65393db` |
| conversion_manifest.json | `20fc75f4f44d79dcca3101babc8d5c53d9a94bddcb5a798fc3ece178e013512c` |
| norm_stats.json | `dce9aa230699df9ba9e6a8cc88c2bdb2b838e564000305df06b1845b811320cf` |
| train_seeds.json | `55b25866308d4e7da486282982ead4d5857b53fd977f659b216045f1950b45e2` |
| eval_seeds.json | `194164f7380fd7cad2a8940ca93def01c2be865da265e4af1c463d73b2aa482f` |

The `adjust_bottle.success_seeds` arrays contain 1000 training and 150 evaluation
IDs. A count of two in the early manifest counts task-object keys, not seed IDs.
The server seed files match the old formal source byte for byte; older Windows
hashes differ only because those checkouts use CRLF. Do not replace server seed
tables based on those Windows hashes.

The replay contract does not embed these model-file hashes. Keep actual evaluation
seed IDs: the inherited config has `use_fixed_reset_state_ids=false`; four envs
times three rollout epochs must not be relabeled as a fixed set of twelve IDs.

DSRL has priority on the assigned cards. The owner/checkpoint/namespace handoff
must prevent the previous RLT owner from relaunching over DSRL. Restore RLT only
after the exact DSRL process identity and GPU resources are released. Never infer
availability from 0% utilization alone.
