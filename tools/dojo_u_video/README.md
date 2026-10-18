# RoboDojo π0.5 action U and full video

Opt-in recording around the installed, previously tested official Dojo/XPolicyLab
JAX π0.5. It retains the native M10/H50 action path and RNG progression. A complete
M5 solve uses the same observation and initial noise; `tools/pi05_signals` computes
the existing per-action U10/5 definition over the 14 real normalized coordinates.
No model parameters are trained or modified.

The client wraps native actions and native three-camera video streams. Every
executed action records query/h, command and simulator ticks. Every encoded frame
records its actual frame index, action count and terminal flag. This makes Dojo's
native joint-action interpolation traceable without Robotwin's approximate TOPP
projection. Input RGB/state/prompt, both complete endpoints and full noise remain
available for recomputation and plotting.

## Budget and layout identity

User budget: about 128 episodes per task, in full batches. The prepared schedule is
54 tasks × 5 batches × 25 environments = 6,750 episodes. Success count does not stop
a task. Model training seed is selected from existing historical evaluations for
figure collection, and is recorded separately from the layout seed.

`layouts.py` visits existing official layout files across seeds 0/1/2 before
repeating. An independent episode ID prevents overwrite, while `source_layout`
records the original seed/file/id and repeat number. A repeated layout receives
the continuing policy RNG stream; it is not reported as a new unique layout.
Only the private import overlay overrides the native 25/50 budget and the random
task parallelism cap. Native control, layout loading and success checks stay in
the installed Dojo implementation. Errors can reduce the achieved episode count.

## Runtime

See `docs/dojo_u_video/00_INDEX.md` and `04_EXECUTION.md` for exact source versions,
machine paths, deployment receipts and actual acceptance status.

On the configured Linux server, with its existing policy/simulator environments:

```sh
CUDA_VISIBLE_DEVICES= DOJO_U_REPO=/path/to/RoboDojo \
  /path/to/policy/python -B tools/dojo_u_video/test_recording.py
CUDA_VISIBLE_DEVICES= /usr/bin/python3 -u -B \
  tools/dojo_u_video/run_task.py /path/to/task-config.json
```

The config supplies repository/output paths, both Python executables and captured
environment JSONs, the physical GPU/UID, the existing EGL scope bootstrap, policy
settings, episode/layout plan and timeout. SZ3 requires its existing local NVIDIA
library links and the verified 595 Vulkan compatibility wrapper. GPU scope must
be confirmed in both compute and graphics process tables.

`run_task.py` has a bounded initialization/action timeout and precise own-process
cleanup using the existing operations identity helper. Ordinary task failures
retain useful completed episodes and skip onward. Invalid recording stops the
lane. Native device loss or incomplete cleanup prevents automatic GPU handback.
The finite `lane.py` waits for the current inference identity and release receipt,
then runs the assigned tasks and releases to the unchanged per-card RLT fallback.
No shared Ray or unrelated experiment is stopped.

## Output

`episodes/layout_NNNNN/` is keyed by **episode ID**, not the source layout number:

- `query_NNNN.npz`: U, M10/M5 normalized endpoints, full initial noise, actual
  transformed actions and RNG keys; `query_NNNN_obs.npz`: input images and state;
  matching JSON: prompt, source layout identity and timing.
- `actions.jsonl`, `frames.jsonl`: actual action-to-frame/tick mapping.
- `episode.json`: native outcome, executed length, three complete MP4 paths and
  the native result path. Native MP4s remain under the installed Dojo result tree.

`validation.json` indexes complete episodes; `candidates.json` recommends up to
12 successful trajectories by U range. All successes and failures are retained.
Select peaks/phase changes first, then inspect the corresponding same-episode
real frames; unexecuted chunk tails must not be drawn as executed actions.

Source, configs and compact evidence may be published. Model weights, videos and
large endpoint/observation arrays remain on the servers.
