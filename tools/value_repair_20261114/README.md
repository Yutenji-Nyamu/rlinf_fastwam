# Value-learning repair experiments

These patches target the accepted DSRL and RLT deployments. Apply each method's patch to its matching source snapshot in an isolated export. Defaults preserve the old behavior; recipes explicitly enable the new mechanisms.

## Apply and reproduce

Run commands from the root of the isolated RLinf source checkout containing this folder:

```sh
# Choose exactly one DSRL patch, according to the existing worker:
git apply --check tools/value_repair_20261114/dsrl_existing_cql.patch
git apply tools/value_repair_20261114/dsrl_existing_cql.patch
# For the earlier worker without the CQL block/helper, use dsrl_add_cql.patch instead.
python tools/value_repair_20261114/test_dsrl.py .
```

For the separate RLT source checkout:

```sh
git apply --check tools/value_repair_20261114/rlt.patch
git apply tools/value_repair_20261114/rlt.patch
```

`apply_rlt.py --source-root SOURCE --output-root OUTPUT` is an alternative to applying `rlt.patch`: it checks exact code anchors and produces the same three edited files. Do not apply both mechanisms to one checkout.

The JSON files in `config/` are merge fragments for an existing resolved experiment configuration, not standalone launch configurations. Apply `dsrl_slow_cql.json` to both DSRL variants; additionally apply `dsrl_gaussian_shift.json` to both actor and rollout model configurations for the shift variant. `rt2_eval32.json` changes the evaluation budget and simulator lifetime together. Environment-specific paths, seeds, device placement and model assets stay in the local experiment configuration.

For RLT, generate the objective schedule from the checkpoint's actual saved optimizer-update count:

```sh
python tools/value_repair_20261114/configure_rlt.py --saved-updates SAVED_COUNT --output rlt-overrides.json
```

Merge that fragment with the existing BC configuration and preserve its checkpoint/replay settings. The fixed-replay probe takes the complete resolved JSON configuration matching the checkpoint. Probe output contains the input checkpoint/config paths for local reproducibility; `cpu-rlt-result.json` here retains only aggregate results.

## Experiments

| Variant | Changes | Rationale |
|---|---|---|
| DSRL slow actor + broader CQL candidates | actor lr 5e-6, critic lr 1e-4, CQL weight .5, candidate noise scale 1 | Let the critic track policy changes; evaluate candidates outside the policy's narrow distribution |
| DSRL Gaussian mean shift | Above recipe plus z=epsilon+.5*tanh(mu(s)); independent standard-normal epsilon; zero-initialized mean | Preserve base-policy noise diversity while allowing a bounded learned change |
| RLT value repair | Separate optimizer gradient clipping; output feature normalization; TD target bounds [0,1]; Q warmup then gradual small Q weight | Repair observed value-scale divergence while preserving learned imitation |

DSRL keeps FP32 small networks/Adam, KL per dimension weight .5, alpha0, UTD5 and 2000 critic-only warmup updates. The mean-shift network is initialized fresh together with critic and replay. No prior tanh policy is loaded into it.

RLT resumes an existing BC checkpoint including optimizer and replay. Keep BC weight1, reference dropout0, fixed reference replay20000 and 50/50 reference/online sampling. Set schedule warmup to the saved optimizer update count +2000; ramp Q weight from0 to.05 over10000 subsequent updates. Use the current actor for TD bootstrap. The target bounds are appropriate for this task's single terminal reward1 and otherwise0, not arbitrary rewards.

## How this differs from previous attempts

Previous DSRL KL regularization still allowed conditional variance to shrink. Gaussian mean shift fixes it at1 and retains Gaussian tails. Its correction bound.5 is a local design choice: KL/d cannot exceed.125. It borrows the bounded, zero-initialized correction idea from LP-DS; it does not implement LP-DS's action critic or adaptive constraint.

Previous CQL used narrow current-policy candidates. Broader candidates follow verl-vla's pre-tanh variance addition: std=sqrt(std^2+1). Only the conservative-loss candidates are broadened. Environment collection and TD bootstrap retain their normal sampling. This is a recipe subset: success-balanced replay is not added because current replay lacks episode-success labels.

RLT previously used LayerNorm and pure BC, with Q still diverging below its valid return range. OFN normalizes the final feature before the scalar head; this differs from LayerNorm. Gradient clipping now excludes gradients belonging to the other optimizer, preserving dQ/da for actor learning.

## Verification

- `test_dsrl.py SOURCE`: CPU tests for exact initial Gaussian noise, independence, tails, learned correction, fixed variance, KL/density and candidate variance.
- `probe_rlt.py --source SOURCE --checkpoint CHECKPOINT --config JSON --output OUTPUT --updates 2000`: restores named critic Adam state from DCP, uses fixed mixed replay with held-out observations, updates Q only. The original checkpoint remains unchanged.
- `cpu-rlt-result.json`: original Q mean−11.07; after2000 Q updates mean.012, bounded TD RMSE.085; actor weights unchanged. Some negative predictions remain and immediate-terminal AUC fell .845→.726 (8 positive held-out transitions). This validates numerical recovery, not improved control or full return calibration.
- DSRL CPU contracts passed; measured base-noise std1.00055 and broadened pre-tanh std1.41059 (expected sqrt2).
- Online validation must cover actual FSDP updates, checkpoint resume and real success rates. Record results separately for each recipe.

## Evaluation concurrency

RT2 training/evaluation use32 environments and one evaluation batch, replacing4 environments×3 batches. The existing evaluation table contains32 valid seeds. Both training and evaluation environments are offloaded between phases so two32-environment simulator sets do not remain resident together. Dojo keeps32 environments for bottle disposal and56 for pouring. Expanded evaluation results have denominator32 and are kept separate from the previous12-episode series.

## Sources

- [verl-vla CNN actor](https://github.com/verl-project/verl-vla/blob/c1de826c7c9256e4d9b6900091e4b0e243d16f5e/src/verl_vla/models/dsrl/actor/cnn.py) and [SAC worker](https://github.com/verl-project/verl-vla/blob/c1de826c7c9256e4d9b6900091e4b0e243d16f5e/src/verl_vla/workers/engine/sac/training_worker.py): candidate sampling and conservative loss.
- [verl-vla π0.5 recipe](https://github.com/verl-project/verl-vla/blob/c1de826c7c9256e4d9b6900091e4b0e243d16f5e/docs/reinforcement-learning/dsrl/pi05/libero-spatial.md): learning rates and CQL settings.
- [LP-DS official code](https://github.com/HikmetSimsir/lpds/blob/main/stable_baselines3/lpds/lpds.py): zero-initialized perturbation around base noise.
- [OFN paper](https://arxiv.org/html/2403.05996v2): final-feature L2 normalization for value divergence.
- [GSBR author's behavior-regularized agents](https://github.com/charlesxu0124/gsbr/blob/main/libero/nicerl/agents/cached_behavior_regularized.py): configurable critic TD target bounds.
- [Spinning Up SAC](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py): separate critic/actor optimization while differentiating Q with respect to actions.
