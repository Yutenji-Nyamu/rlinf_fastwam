# π0.5 U and complete-timeline video

Continue the accepted 2026-10-03 DV50/U collection: all 50 RoboTwin tasks,
32 episodes per task, B16 per physical GPU 6/7, original seeds/noise/episode
limits, H50, M10 main policy and same-full-noise M5 side pass. The existing
policy-budget evaluation has priority; the original per-card RLT runs last.

The previous π0.5 recorder saved only chunk-boundary images. This recorder
samples the **entire simulated execution at 25 Hz**, plus initial and query
end/success images. It records every physics step in a separate JSONL. It
does not claim a video frame for every model action or every 250 Hz physics
step. Main-camera capture uses the native camera directly and does not call
`get_obs`, redraw the scene, advance physics, change TOPP or change actions.
Only two recording calls are inserted into the installed native controller
inside this process; shared RoboTwin files are untouched.

For each query/slot, `*_controller.npz` retains original actions, compressed
paths and exact retained vertex indices, TOPP/downsampled commanded targets,
gripper targets, and per-arm model-h projections with geometric residuals.
TOPP-to-h coordinates are **approximate geometric projections**, not exact
inverse TOPP parameters. Frame sidecars retain exact physics tick, simulated
time, query, slot, seed and commanded-target index. This preserves enough
raw information to inspect U peaks without pretending that frame h equals
action h. Video playback is constant FPS; boundary frames have their true
timestamps in the sidecar.

`query_*.npz` retains per-action U, both endpoint chunks, full initial noise,
main chain and actual submitted action. Observation arrays and task language
are stored alongside it. U uses the existing `ugrow_10_vs_5` implementation
over the 14 real action dimensions. No attention/hidden-state observer is added.

The first valid B16 batch on **each card** must pass exact main-action/RNG
parity and a CPU check that fully decodes each MP4, checks frame coverage,
checks all control-step records and recomputes U. Both cards wait for these
receipts before expanding to their remaining tasks. Native task failures are
recorded and skipped; recording validation/cleanup failures stop the queue.
The inherited 600-second progress / 3600-second batch deadlines remain.

Server paths:

- Source: `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-u-full-video-20261017`
- Output: `/data/chenyiteng/results/pi05-u-video/20261017-v1`
- Transaction: `/data/chenyiteng/deployment-20261017/pi05-u-video-v1`
- CPU test: `CUDA_VISIBLE_DEVICES= ROBOTWIN_PATH=... python -B tools/pi05_u_video/test_recording.py`
- Batch: inherited environment, `python -B tools/pi05_u_video/run_batch.py CONFIG`
- Lane: `/usr/bin/python3 -B tools/pi05_u_video/lane.py 6` (or 7), with
  `U_VIDEO_ROOT` and `U_VIDEO_TRANSACTION` set to the paths above.

CPU tests establish native command/RNG preservation and synthetic recording
coverage. Actual renderer performance and real video acceptance remain pending
until the current inference queue releases these cards. Do not describe a
prepared/queued batch as a completed experiment.
