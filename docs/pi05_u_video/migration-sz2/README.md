# U/video collection moved to Shenzhen2 GPU4/5

2026-10-17, user-authorized migration. Shenzhen1 waiting lanes were removed before they ran; its existing policy-budget evaluation continues, followed by original per-card RLT. GRPO U tau2 on Shenzhen2 was closed at 163/200 updates, and its namespace and both compute/graphics contexts cleared.

The original 100 configs and exact seed files are retained: 50 tasks, 32 episodes each, B16 per card, H50/M10 and U from same-noise M5 endpoints, continuous head video at 25Hz simulated time. Only hardware identity, runtime paths and lane pair changed. Existing model, venv and RoboTwin commit 0008ae6 are reused. Runtime code is 30ee66c09.

Output: `/data/chenyiteng/results/pi05-u-video/20261017-sz2-v1`. Transaction: `/data/chenyiteng/deployment-20261017/pi05-u-video-sz2-v1`. First two real valid B16 batches gate expansion. CPU native-controller check passed; initial GPU startup is recorded, real-video validation follows in acceptance-g4/5. Native task errors are skipped; video/U validation or cleanup errors stop the lane. Original RLT fallbacks remain unchanged; the owner is the existing per-host CPU scheduler.

Migration scripts are one-shot receipts/reference, not instructions to replay. The preparation script includes recovery assertions for the empty directories left by its initial Python3.10 path/extraction compatibility failures; no GPU was launched during those failures. Configs, source and CPU check completed before training cutover.
