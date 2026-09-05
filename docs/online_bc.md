# RoboTwin pi0 online success BC

Status: implementation and CPU/config checks completed; GPU end-to-end validation
is still pending. Based on official RLinf `dc9b87cc49334c7516487ead68ebeb060fd7c090`.

The existing embodied runner performs collection, supervised updates, weight
synchronization, evaluation and checkpointing. `online_bc` selects a supervised
actor that reuses the DAgger FSDP update loop, but **not** its teacher or intervention
filter. No critic, advantage objective, additional exploration network or renderer
patch is required by this method.

## Data and loss

Collect complete, chunk-aligned RoboTwin rollout rounds with `auto_reset=false`.
Keep the pre-query observations and the command passed to the environment. Admit
successful episodes into a cumulative replay and sample query records uniformly
with replacement. Stop recording an environment after success or termination,
until its next explicit reset. Failed attempts still belong to the collection
budget, but do not enter the success actor replay.

RoboTwin applies TOPP to the whole waypoint command and may stop its physical
control loop on success. Its current interface does not expose a reversible
waypoint execution prefix. This implementation learns the **submitted macro
command**, not a fabricated sequence of per-physics-step actions or images.
The action mask excludes padding/invalid target dimensions; it does not claim
that every waypoint was physically reached before an early stop.

Native pi0 FM supervises the submitted action after the existing input transforms.
There is no GRPO loss or intermediate SDE exploration injection. Collection uses
the model's normal random-initial-noise ODE inference path (`mode=eval`), not a
claim of deterministic behavior across independently drawn initial noise.

## Parameters

Use `examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml`.
It is a two-round single-GPU capacity smoke, **not** an automatic formal launch:
GPU6; 32 parallel training environments x one serial sampling batch; 32 fixed
evaluation environments; micro-batch 32 / global batch 1024; two optimizer steps
per round. It keeps the intended per-GPU parallelism and training batch sizes.

Model parameters inherit the official pi0 recipe: `num_steps=4` comes from
`model/pi0.yaml`, also inherited by official RoboTwin pi0 DAgger. This is an
inference integration budget, not a claim that the checkpoint is distilled to
exactly four steps. The previous local override to 10 has been removed.
The supervised optimizer LR 2.5e-5, Adam betas 0.9/0.95, epsilon 1e-8, weight
decay 1e-10 and gradient clip 1 come from official RoboTwin **pi0** DAgger.
Its 1000-step warmup / 30000-step cosine schedule is not inherited: this short
budget uses constant LR and zero warmup. Two updates per round is our explicit
budget choice, not DAgger's default one update.

The proposed 100-round run also collects 32 attempts per round, with evaluation
every 5 rounds and saving every 10; only the smoke's round/interval/name fields
and optimizer total differ. Always use a new run directory and the original SFT
initialization, not the smoke's trained weights, for that separate run.

- `actor.model.openpi.train_expert_only=true`: freeze PaliGemma vision/language;
  train the action expert and its action/state/time projections. The actor checks
  the trainable parameter scope at initialization.
- `algorithm.online_bc.demo_weight=0`: pure online success BC. No demo dataset is
  loaded. For a positive weight, also provide an existing local LeRobot dataset
  through `algorithm.online_bc.demo_data_path`.
- With demos: `loss = (online_FM + demo_weight * demo_FM) / (1 + demo_weight)`.
  Each supervised micro-batch gets a demo micro-batch of the same size. This is a
  **loss mixture weight**, not a ratio of collected episodes. The optional loader
  uses RLinf/OpenPI's existing RoboTwin data transforms. Its real dataset path is
  not yet end-to-end verified; missing data raises an error rather than silently
  disabling the mixture or downloading a dataset.
- Collection attempts per round are `env.train.total_num_envs * rollout_epoch`.
  `algorithm.update_epoch` counts optimizer steps, each consuming
  `actor.global_batch_size` replay query records; it does not iterate the full pool.
- `algorithm.online_bc.data_path` stores success batches. Checkpoints also store
  replay contents, replay RNG and learner update count. The normal strategy stores
  model and optimizer state. No claim of bitwise demo-loader resume is made.

Set `REPO_PATH`, `EMBODIED_PATH`, `ASSETS_PATH`, `PI0_MODEL_PATH`, and
`ONLINE_BC_RUN_DIR`; include the selected RoboTwin repository on `PYTHONPATH`.
`ASSETS_PATH` / `env.*.assets_path` must be the **RoboTwin repository root**,
not its `assets/` child: the native environment sets this variable before import,
and the clutter loader appends `assets/objects/objaverse/list.json` itself.
On a shared Ray server, set `RAY_ADDRESS` and `RLINF_CODE_WORKING_DIR`, select an
idle physical GPU in `cluster.component_placement`, and use absolute run paths.
Never restart the shared head to launch this job.

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-name robotwin_adjust_bottle_online_bc_openpi
```

## Tests and remaining validation

```bash
python -m pytest -q tests/unit_tests/test_online_bc.py
```

Nine tests cover pre-query/command alignment, success vs termination, exclusion
of post-terminal queries, cumulative archives, replay RNG restoration, masked FM
gradients, optional demo loss mixture, the teacher-free expert-only config, and
real OpenPI augmentation with its native FP32 inputs. As in official pi0
DAgger, model/FSDP precision is null: OpenPI keeps its native BF16 backbone and
FP32 projections. Globally forcing FSDP BF16 breaks augmentation and native
projection input contracts. No extra image/projection cast patch is required.
This BC configuration explicitly sets `openpi.image_augmentation: false`:
random crop, rotation and color jitter are off, but model resize and normalization
remain. The new flag defaults to true for other configurations; inference always
uses the original non-augmented path. This is a baseline choice, not a claim
that augmentation cannot affect learning or generalization.
Worker import and full `validate_cfg` have also passed on the target server.
The synthetic demo-loss test does not validate a real LeRobot dataset.

Still required: GPU collection/update/redeployment and real model/optimizer
checkpoint validation. A single-GPU smoke is not a learning-gain measurement or a
long-run renderer stability result. Multi-rank readiness is synchronized, but
multi-GPU training has not been tested for this new method.
