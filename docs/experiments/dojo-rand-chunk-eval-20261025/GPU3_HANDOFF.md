# GPU3 queue handoff · 2026-10-25

User requested releasing SZ1 GPU2 while keeping GPU3 occupied. The existing randomized four-episode sweep now runs on GPU3, followed by the paused four-episode Dojo sweep and then the existing RLT fallback workflow. GPUs4–7 retain their original training and fallback configuration.

The move retained the completed C10/N1 summary and 15 completed C10/N10 task results. Dojo retained 19 completed or explicitly skipped conditions; the manually interrupted condition is resumed separately. Sample selection, seeds, model, chunk sizes and inference steps are unchanged. Original artifacts remain at their previous paths.

The two old dispatchers were stopped by PID/UID/start identity through their existing cleanup handlers. GPU2 was verified free of compute and graphics contexts. The three resident Rand workers were observed with compute+graphics contexts only on GPU3, and completed-task count advanced from 15 to 16. GPU3 graphics binding uses its own comm-name profile and mask8; it does not modify the GPU2 profile.

`tools/pi05_distillation/gpu3_chain_20261025.py` runs the two accepted queue entrypoints sequentially, preserving their error/timeout cleanup. It releases GPU3 to the existing RLT owner only after the second queue exits with a clear card. The new GPU3 RLT candidate copies the existing GPU4 candidate's complete CP225 into a distinct run/namespace configuration; checkpoint files are referenced, not duplicated. Its private scope allows only physicalGPU3; preload/profile/manifest were checked without creating an extra GPU context. Existing GPU4–7 candidate requests remain unchanged.

Deployment chain: `/data/chenyiteng/deployment-20261025/g3-rand-dojo-rlt`. New deployments: `rand-chunk-sz1-g3-r4` and `dojo-chunk-sz1-g3-after-rand`. Each `prepared.json` gives its exact output. No weights, videos, credentials or other users' data are included in this publication.
