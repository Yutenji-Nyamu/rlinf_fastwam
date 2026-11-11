# Appendix signal collection

Bounded offline measurements and small frozen-policy rollouts for appendix figures. No learner parameters or replay sampling are changed. The optional `pre_native_phases` hook runs under the existing Dojo owner and returns to its prepared formal checkpoint continuation, including when a measurement exits unsuccessfully.

## Measurements and limits

| Collector | Fixed scope | Saved evidence |
|---|---|---|
| `prepare_rt` | Existing 75 gallery candidates; two-task, 64-observation query pool | Original K4 normalized/physical actions, initial noise, exact executed-prefix mapping |
| `rt_sampling` | Two independent K8 banks on the same 64 observations; K1/2/4/8, three timing repeats | Native B16 replay gate, seeds, actions, timing and allocator peaks |
| `rt_appearance` + `mcr_features` | Two tasks, four episodes per task, four states per episode, two lighting conditions, three cameras | Frozen physical-state hashes, original images, raw 2048-D view features, distances and graph gates |
| `dojo_sampling` | Two tasks, 12 episodes each, at most 12 K4 query positions per rollout wave | Main action and RNG equality, actual H/C, executed prefixes, observations and terminal frames |
| `rynn_values` | At most 20 original clips and four instruction counterfactuals; 352 prefix queries | Eight causal frames per query, all prediction slots, last-slot remaining seconds, raw output shape |
| `geowam_sampling` | Private implementation invoked from an authorized checkout | Private source/data remain in that repository; no private implementation is vendored here |

The query pool is fixed before examining scores. Existing videos are source material, not new model success measurements. Three repeated latency samples are a short cost check, not a broad performance benchmark.

## Definitions and source basis

- Entropy uses the existing `tools/pi05_policy_eval/entropy_analysis.py` convention: diagonal Gaussian plug-in estimate on normalized action coordinates, unbiased sample variance, epsilon `1e-6`. It measures sampled action disagreement. Absolute entropy across different K has estimator bias; the renderer reports both raw and bias-centered differences, and compares against an independent K8 bank rather than calling that bank ground truth.
- Main-plus-extra K4 collection reuses `tools/pi05_u_video/noise_samples.py`. The original action and global RNG states must stay unchanged. No K bank combines observations from different times.
- [GR-BC](https://github.com/Gamma-Robot/GR-BC/tree/fbdc03b764276751db7658c907b741d816511ced): MCR ResNet50, Resize256/CenterCrop224/ToTensor without ImageNet normalization; three-camera and gripper gates from its graph code. The diagnostic graph is sparse and is not a full GR-BC training evaluation. Same-state lighting pairs are controlled; other-time frames need semantic review before they are described as different task stages.
- [MCR weights](https://huggingface.co/GqJiang/robots-pretrain-robots/tree/057651ca0b6a27aba390449603f674cd0977f87c) are checksum-locked in the run receipt.
- [RynnValue](https://github.com/alibaba-damo-academy/RynnValue) recommends causal attention at inference; this uses SDPA and the last value slot of each observed prefix. [RoboValue](https://github.com/RoboValue-Benchmark/RoboValue) supplies the prefix/remaining-time convention. Remaining seconds are not a success probability. Instruction-mismatch inputs are labeled counterfactuals.

## Operation

Prepare model/data paths and the complete continuation checkpoint before transferring ownership. The deployment plan lists exact commands, output paths, per-stage caps and an absolute deadline. `pipeline.py` runs stages sequentially; cleans only matching UID/PID/start-time and stage-tagged descendants; checks physical GPU binding; and writes a receipt for each completed, failed or skipped stage. The optional pre-native phase then hands back to the accepted native training lifecycle.

The resource scope, storage paths and deadline belong to the local deployment plan. The collector guard and process monitor enforce the single-device scope; runtime identities and host details are retained with local execution receipts.

CPU contract checks, run in the server runtime with no visible GPU:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tools/appendix_signals/test_contracts.py
python -m ruff check --select E4,E7,E9,F,I tools/appendix_signals
python -m ruff format --check tools/appendix_signals
```

Render saved public and private arrays into separate local galleries without rerunning the models:

```bash
python -m tools.appendix_signals.render --data /path/to/appendix-results
```

Keep simulation arrays, manifests and plots together. Keep real-robot RGB-D/state arrays and their derived figures in the existing private repository. Download credentials, model shards, learner checkpoints and training replay are not source-code artifacts.

## Validation status

Preparation: seven CPU contract tests and the targeted Ruff checks pass. GPU replay parity, real sampling, resource peaks and restoration progress require the execution receipts; preparation alone does not establish their success.
