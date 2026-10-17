# U/video collection moved to Shenzhen2 GPU4/5

2026-10-17, user-authorized migration. Shenzhen1 waiting lanes were removed before they ran; its existing policy-budget evaluation continues, followed by original per-card RLT. GRPO U tau2 on Shenzhen2 was closed at 163/200 updates, and its namespace and both compute/graphics contexts cleared.

The original 100 configs and exact seed files are retained: 50 tasks, 32 episodes each, B16 per card, H50/M10 and U from same-noise M5 endpoints, continuous head video at 25Hz simulated time. Only hardware identity, runtime paths and lane pair changed. Existing model, venv and RoboTwin commit 0008ae6 are reused. Runtime code is 30ee66c09.

Output: `/data/chenyiteng/results/pi05-u-video/20261017-sz2-v1`. Transaction: `/data/chenyiteng/deployment-20261017/pi05-u-video-sz2-v1`. First two real valid B16 batches gate expansion. CPU native-controller check passed; initial GPU startup is recorded, real-video validation follows in acceptance-g4/5. Native task errors are skipped; video/U validation or cleanup errors stop the lane. Original RLT fallbacks remain unchanged; the owner is the existing per-host CPU scheduler.

Migration scripts are one-shot receipts/reference, not instructions to replay. The preparation script includes recovery assertions for the empty directories left by its initial Python3.10 path/extraction compatibility failures; no GPU was launched during those failures. Configs, source and CPU check completed before training cutover.


## Startup graphics correction

Initial v1 startup left graphics-only contexts on GPU0. Before accepting or expanding collection, both new U lanes and their exact child processes were stopped; their GPU0/4/5 contexts cleared, while EXPO6/7 stayed unchanged. Outputs from the first startup are retained as incomplete. At 19:12 the same 100 configurations launched in transaction `pi05-u-video-sz2-v2` and output `20261017-sz2-v2`, using the already validated SZ2 GRPO4/5 graphics scope implementation/profile and a private receipt directory. CPU validation bypasses that graphics bootstrap. No policy, simulation, task, seed or recording budget changed.

At 19:14, the copied private scope manifest needed group-write permission removed to match the existing scope contract. Startup exit78 occurred before any v2 batch directory or GPU work was created. Both finite CPU lanes were held; all startup failures were archived in bootstrap-failed, and the original complete 100-job list restarted after actual-environment bootstrap checks passed for both cards. d73a1b5d1 now treats exit78 as a queue-stop condition rather than a native task failure. This avoids consuming the task list on a setup fault.

The startup-recovery archive selection accidentally included lane-plan files; both restarted lanes exited before any batch and RLT briefly took over. The two exact RLT jobs were stopped and their namespaces cleared, the plans restored and all 100 configs retained. At 19:17, lane4/5 PIDs1622180/1622183 and owner1622187 were launched. RLT keeps its original settings and complete CP550/CP300, with fresh dispatch directories to preserve the earlier start/exit logs. The final priority is U/video then per-card RLT; EXPO6/7 stayed active throughout.
