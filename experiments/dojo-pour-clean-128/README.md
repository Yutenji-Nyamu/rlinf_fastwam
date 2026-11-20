# RoboDojo π0.5 GRPO · pour_liquid_into_cup

Fresh seed-1 base checkpoint, new optimizer, step zero. Two devices each run 32 layouts/slots per wave, grouped into sets of eight attempts per training layout. Two waves collect 128 episodes. The native 400-control-step horizon, 50-action policy chunk, three RGB cameras, reward computation, and physics interval are retained. The actor uses two epochs, global batch 1024 chunk positions, learning rate 5e-6. Evaluation uses the native per-action render cadence and runs at initialization, round 2, then each 10 rounds. The formal target is 200 rounds. Checkpoints rotate only after a complete write; the best trained evaluation is archived separately.

The render trial uses layouts 0, 18 and 1 (two previously solved, one unsuccessful). An original policy supplies shared actions for eight chunks. Two simulators replay these actions at identical seeds and exchange full and query-boundary rendering on the second pass. Policy queries at matched inputs use identical noise. Timing covers the action chunks, including boundary observations, and excludes model inference, reset and RPC. The function also compares three cameras, robot states, generated actions and exact termination traces. The probe uses three physical scenes per card. Rates at full production concurrency need a separate comparison.

The conservative fallback uses the already verified capture mode. The optional training-only render change requires faster execution on both passes, repeatable successes and exact terminal outcomes, and action and camera differences within the documented probe criteria in render_trial.py. The setting is selected before any training rollout. The isolated timing calls charge nested work to the innermost operation. The small probe is excluded from training, evaluation metrics and training seed counts.

The deployment places render_trial.py beside its plan, CPU preflight and trial/layouts.json. The trial lanes use separate native output and authenticated local sockets. The portable configuration uses DOJO_MODEL, DOJO_RUN_DIR and DOJO_RESULTS; the per-card queue helper directory is supplied via DOJO_QUEUE_OPS.

[RoboDojo environment](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/src/eval_client/eval_env.py) · [XPolicyLab π0.5 inference](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/Pi_05/model.py) · [Camera synchronization](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/env/camera_manager/capture/render_sync.py) · [Isaac Sim 5.1 performance handbook](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html)

## Measured run

[Results and formal settings](./measured.md) · [Native observations and episode outcomes](./observed-render.json). Formal training uses per-action rendering.
