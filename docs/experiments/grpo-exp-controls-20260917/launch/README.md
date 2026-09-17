# GRPO128 exp tau2 dropout0.2 anneal200 launch

Source `ab4d065488c1dd1181731f107a9ba6721a421e27` was tested and pushed before the GPU4/5 cutover. The full128 smoke completed with exit0 and passed its bounded gate; the separate fresh formal run passed startup checks before this evidence commit.

Formal: 200 rounds, 64 environments x 2 rollout epochs =128 episodes/round, G8, U2, B512/micro32, LR5e-6; fixed32 every5 rounds, save every10. The Clean128 configuration is preserved outside method and identity/path fields. DVAC is exp_mean, positive advantages only, both tau2; both alpha start1 at R1 and linearly reach0 at R200. Chunk dropout probability is0.2, keeps the base GRPO advantage unchanged on selected chunks, with no inverse-probability compensation.

Smoke observed 28 / 129 eligible chunks dropped (21.71%). Kept and dropped tensors, finite losses/gradients and rollout seeds42/43 were checked by the smoke gate. The one-round smoke does not test full learning effectiveness or empirically reach R200; annealing endpoint and restore semantics were covered by prior CPU tests.

Run: `/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-exp-t2-drop02-anneal200-positive-half128-seed42-formal200-phys45-20260917-v1`

Namespace: `RLinf_grpo_exp_t2_drop02_anneal200_positive_half128_seed42_formal45_20260917`

[Startup acceptance](STARTUP_VERIFIED.json), [smoke result](smoke-result.json), [method and budget diff](formal/baseline-diff.json), [actual config](formal/actual-config.yaml), [command](formal/command.txt), [smoke metrics](smoke/metrics.csv).

[Smoke detail](smoke-summary.json) preserves the CPU replay checks, coefficient distributions and dropout displacement measured after formal launch. Coefficient displacement is not a measurement of model-gradient reduction.

Only lightweight, explicit evidence files are copied. No model, checkpoint, replay payload or environment secret is included. Original source/config files and all training processes are preserved. Production SHA and every committed evidence blob are checked before push.
