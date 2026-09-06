# Shenzhen dual 2-GPU GRPO formal operation ledger (2026-08-26)

## Objective

Replace the running 4-GPU GRPO-DVAC global-z weights `[0,5]` experiment with two concurrent, strict-matched 100-step experiments:

- physical GPUs 4,5: original GRPO control;
- physical GPUs 6,7: GRPO-DVAC global-z weights `[0,2]`.

Both retain the Shenzhen-validated 32 train environments per GPU. The linked two-rank contract is `64 train env x 4 rollout epochs = 256 trajectories/step`, `G=8` (32 groups), at most 1024 chunk records, `global_batch_size=1024`, `micro_batch_size=32`, `update_epoch=2`, and fixed-64 evaluation as `64 eval env x 1 wave` every 5 steps. Checkpoints are every 10 steps.

## Authorization and boundaries

- User explicitly authorized stopping the owned `[0,5]` run and immediately launching both formal 100-step runs.
- Do not touch GPU 0--3 jobs owned by other users.
- Preserve the persistent shared Ray head; cleanup is limited to the exact owned process group and its exact namespace.
- Prepare configs and launchers before stopping the old run to minimize idle time on GPUs 4--7.

## Operation log

### 2026-08-26: preflight and preparation

1. Read workspace rules, handoff, current Shenzhen implementation plan, current `[0,5]` launcher, and prior namespace-scoped stop script.
2. Reconstructed the approved linked-parameter change: world size 4 to 2; train env 128 to 64; global batch 2048 to 1024. Rollout epochs, fixed-64 evaluation (`64 x 1`), G8, MB32, update2, method-independent model/reward/log-prob/optimizer leaves, eval frequency, and save frequency stay unchanged.
3. Corrected two draft assumptions before any launch. `B512` would double optimizer calls from 2 to 4, whereas `B1024` preserves 512 records/rank and 2 calls/outer-step. `32 eval env x 2 waves` cannot implement fixed-64 because RoboTwin leaves fixed reset IDs unchanged between waves; it would repeat the same 32 resets. Therefore evaluation stays `64 x 1`. The two new runs share the same two-rank fixed-reset partition with each other, though that partition is not claimed identical to the old four-rank partition.
4. Live server identity, exact namespace/job ownership, output-path absence, source HEAD/dirty status, and shared-Ray status are checked immediately before preparation and mutation.

Live preflight at 11:05 CST found the old `[0,5]` wrapper alive at complete Step 44 and Step 45 rollout `3/4`, fatal count zero, exact Ray job `24000000`, namespace `RLinf`, and exactly 12 GPU workers on physical GPUs 4--7. The shared Ray head at `172.17.0.1:6389` was healthy; source HEAD was the clean `0e28ac6f...`; both new output paths were absent. GPU 0--3 processes were outside the authorized target and were not touched.

The two server-composed resolved packets passed exact pair parity: unexpected differences zero. The only differences are DVAC `mode`, physical placement, experiment/run names, and run-scoped absolute output paths.

### 2026-08-26: cutover

1. At 11:18:01 CST, the old `[0,5]` run was stopped after complete Step 45. Cleanup targeted only its owned process group and exact `RLinf` named actors; the shared Ray head remained alive. Marker: `runtime/stopped_by_user_for_dual_2gpu.txt`.
2. The control started on physical GPUs 4,5 with wrapper PID `356706`, Ray job `2c000000`, namespace `RLinf`.
3. Once the control managers registered, the DVAC `[0,2]` run started on physical GPUs 6,7 with wrapper PID `360398`, Ray job `32000000`, namespace `RLinf_1`.
4. At 11:19 CST both wrappers were alive, each namespace had 15 named actors, GPU job IDs were disjoint, fatal counts were zero, and host available memory was about 1.9 TiB.
5. At 11:26 CST both runs had completed the first of four rollout waves (`1/4`): control 295.66 s, DVAC 292.54 s. Both wrappers remained alive with fatal counts zero. GPU memory was about 28.2/28.7 GiB on 4/5 and 21.5/20.8 GiB on 6/7; host available memory was about 1.7 TiB. This closes the requested startup observation; neither run is claimed to have completed an optimizer update yet.

### Old `[0,5]` closeout package

The 70 GiB run was not copied. A 136,410-byte high-information ZIP contains the driver log, one-minute resource CSV, resolved configuration, exact command, parity/contract/launch records, stop marker, TensorBoard event/config, and a 2,331-row flat scalar CSV. It excludes checkpoints, video, and RoboTwin bulk data while preserving enough inputs to redraw training/evaluation/resource figures:

`evidence/dvac-global-z-w0to5-step45-high-info-20260826.zip`

### 2026-08-26 15:26 CST: first inline evaluation failure

Read-only refresh used
`local_scripts/remote_commands/shenzhen_dual_grpo_2gpu_startup_readonly_20260826.sh`,
`shenzhen_dual_grpo_latest_metrics_readonly_20260826.sh`, and
`shenzhen_dual_grpo_exit_readonly_20260826.sh` through the fixed-host-key Paramiko route.

- Both runs completed four full training outer steps. Their latest TensorBoard rows are finite: control Step 4 success/KL/clip/grad=`83.20%/.0105/.0863/27.23`; DVAC Step 4=`73.83%/.0177/.0818/33.80`, DVAC ESS=`.8197`.
- At the first scheduled fixed-64 evaluation, both jobs independently failed while resetting the RoboTwin evaluation vector environments with `vk::Device::getSemaphoreFdKHR: ErrorInitializationFailed`; the subsequent `ray.kill` actor death and Gloo peer-closed messages are downstream consequences.
- Wrappers exited `255` at 13:01/13:03 CST. There is no Step 10 checkpoint. Physical GPUs 4--7 are idle, host available RAM is about 1.9 TiB, and the persistent shared Ray head remains alive.
- This is not a numerical-training or host-memory failure. The common new stressor is inline fixed-64 on two ranks (`32 eval env/GPU`) and simultaneous evaluation by both jobs, versus the prior four-rank run's `16 eval env/GPU`; the exact Vulkan limit contribution is not yet uniquely separated. No restart or configuration change was made.

### 2026-08-26 15:53--16:05 CST: fixed-32 fresh v2 restart

By explicit user authorization, two fresh v2 runs were prepared and launched. Relative to v1, the only scientific leaf change is `env.eval.total_num_envs: 64 -> 32`; `rollout_epoch=1`, so each two-rank run evaluates fixed-32 with 16 eval environments/GPU. Training remains `64x4`, G8, B1024/MB32/update2, eval5, save10 and 100 steps. Pair resolved parity again reports unexpected differences zero.

- control: GPUs4,5; namespace `RLinf`; Ray job `3e000000`; wrapper PID `3882371`;
- DVAC `[0,2]`: GPUs6,7; namespace `RLinf_1`; Ray job `43000000`; wrapper PID `3884405`.

At 16:05 both wrappers were alive, each namespace had exactly 15 actors, fatal count was zero, GPU ownership was disjoint, and both had entered the first real `Generating Rollout Epochs 0/4` path. GPU memory was about 18.4--18.7 GiB/card and host available RAM about 1.8 TiB. No further observation or configuration change was made.

At 18:13 both runs had completed Step 5, including their first fixed-32 evaluation, and had entered Step 6 rollout `2/4`; therefore the v1 Vulkan evaluation failure boundary was crossed successfully. Control/DVAC Step 5 train success was `86.33%/85.55%`, fatal count remained zero, GPU memory was about 65.5--68.7 GiB/card, and host available RAM about 1.2 TiB.

### 2026-08-27 09:56 CST: Step 39 matched comparison

Read-only refresh found both wrappers alive, fatal count zero, and both runs complete through Step 39 while finishing the Step 40 evaluation/save boundary. DVAC-Control train-success differences through Step 39 are cumulative `+4.57 pp`, trailing-5 `-0.70 pp`, and trailing-10 `+1.64 pp`; seven fixed-32 evaluations are tied at `204/224` each. Pair parity remains `unexpected_differences=[]`. DVAC is active rather than a no-op: `weight_mean=0.9991`, `ESS=0.8094`, warm-up off. Figures, compact raw evidence, exact implementation audit, and cross-machine interpretation are indexed by `evidence/dual-2gpu-comparison-live-20260827/README.md`. No process or server state was changed.

### 2026-08-27 15:08 CST: Step 51 refresh

Both wrappers remain alive with fatal count zero and have entered Step 52 rollout. Through Step 51, DVAC-Control train-success differences are cumulative `+3.21 pp`, trailing-5 `-1.09 pp`, and trailing-10 `-1.09 pp`; ten fixed-32 evaluations total `296/320` for DVAC and `294/320` for Control. Latest DVAC `weight_mean=0.9909`, `ESS=0.8301`. Step50 checkpoints exist for both runs. GPUs4--7 use about 68--70 GiB/card and host available memory is about 307 GiB. The lightweight snapshot and figures were refreshed in place; no server state was changed.

### 2026-08-27 15:28 CST: DVAC Step52 closeout and Prism replacement

By explicit user authorization, the GPU6/7 DVAC `[0,2]` run was stopped after complete Step52 and replaced immediately by the separate Prism-style DVAC-Rank-RLOO formal. The GPU4/5 Control, shared Ray head, and other users were not changed. Exact cleanup removed only old job `43000000` / namespace `RLinf_1`; the new Prism job reused the namespace with job `50000000`. The cutover gap was about one second.

The final paired Step1--52 DVAC-Control train-success differences are cumulative `+3.13 pp`, trailing-5 `-1.95 pp`, and trailing-10 `-1.33 pp`; fixed-32 remains `296/320 vs 294/320`. A 19-file, 336,197-byte lightweight archive preserves the resolved config, command/contract/parity, driver/resource logs, TensorBoard event/config, stop marker, derived CSVs and two redrawable figures; checkpoints, video and RoboTwin bulk data are excluded:

`exports/shenzhen_grpo_dvac_w0to2_2gpu_stopped_step52_light_evidence_20260827.zip`
