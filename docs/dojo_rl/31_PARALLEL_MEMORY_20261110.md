# Parallel rollout memory and worker file limits

Increasing environment count grows simulator/render state and inference activations, not every resident model or optimizer tensor. Measure concurrent process memory over collection, update, reset and evaluation; do not multiply total VRAM by the parallelism ratio or sum non-concurrent component maxima.

A measured DSRL learner used about 17.7 GiB before updates (model tensor payload about 8 GiB plus a target copy and runtime allocations; optimizer checkpoint tensors about 55 MiB). Its inference process reached about 14.5 GiB at 32 environments. A RoboTwin environment process used about 32.8 GiB in a steady collection snapshot, with larger initialization/reset transients. RLT learner weights were only 8.25 MiB and its learner process about 1.3 GiB. These are measured deployment examples, not universal sizing constants.

RoboTwin 64-env startup failed while exporting a camera Vulkan semaphore. The training driver raised its file limit, but Ray environment workers still had a 1024 soft limit; a measured 32-env worker already used 602 descriptors. This supports descriptor exhaustion as the cause to test, rather than diagnosing the Vulkan error as a proven CUDA OOM.

Set `env.max_open_files: 8192` to raise the soft limit inside each environment worker before simulator initialization. The default leaves limits untouched, existing larger limits are not reduced, and the hard limit is preserved. This does not reset shared Ray or change rewards, replay contents, policy precision or update ratios. Linux subprocess tests call the actual initialization entry point and check that more than 64 descriptors can be opened after raising the limit.

`env.train.enable_init_offload: false` is an existing optional setting which avoids closing freshly initialized environments immediately before the first reset; it is separately configurable from between-round offloading. Runtime validation and steady throughput measurements must accompany memory sizing. A target near 60 GiB with 6–10 GiB peak headroom is a workload preference, not a correctness requirement.

References: existing `rlinf/workers/env/env_worker.py` initialization, `rlinf/workers/rollout/hf/huggingface_worker.py` inference batch calculation, and Linux [getrlimit/setrlimit](https://man7.org/linux/man-pages/man2/getrlimit.2.html). Training-driver limits are not sufficient for workers created by a long-lived Ray daemon.

## Follow-up: the later failure was an actual CUDA OOM

With the worker limit raised to 8192, the 64-env retry subsequently failed to allocate 1.89 GiB: the simulator occupied 62.31 GiB, inference 15.20 GiB and learner 1.33 GiB. Raising descriptor limits does not solve VRAM exhaustion. A 48-env continuation was therefore prepared, estimating 60–66 GiB concurrent memory; runtime verification remains necessary. The original policy recipe and 12-episode evaluation are preserved, while the per-round update cap scales from 12800 to 9600. Resuming round 610 to 616 collects 288 planned episodes, versus 256 originally remaining; the 32-episode rounding difference is explicit.

The three Linux subprocess tests passed after supplying the worker group fields in the fixture. They check descriptor handling only, not GPU capacity or policy performance.

The 48-env continuation completed round 611: collection 967.74 s, whole training round 1073.02 s, 2545 critic and 1272 actor updates. One-second device sampling observed a 69.1 GiB peak without CUDA OOM, leaving about 10.5 GiB on the measured card. Simulator unloading still reduces memory between collection phases. This validates the first collection/update cycle, not every future peak; no new evaluation result was available at acceptance.
