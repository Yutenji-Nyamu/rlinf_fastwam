# π0.5 DV50 inference

Original π0.5 RoboTwin base, native 10-step eval flow ODE, 50-action chunks, DV L=3. Fifty official task limits; 16 environments × two batches each. No optimizer, method weighting, DDP, or Ray connection. Each subprocess sees exactly one authorized physical GPU.

`prepare_manifest.py BASE_CONFIG REPO OUTPUT ENVIRONMENT` freezes 100 batch configs. `run_queue.py OUTPUT` runs turn_switch/adjust_bottle batch zero as reusable smoke data, checks DV/episode/video consistency, then runs the remaining queue. A completed batch is skipped; partial batches require explicit retry in a fresh output. Timeouts signal only the controller's own child process group.

Official eval seeds exist for 22 tasks. Other tasks use distinct deterministic seeds without expert success filtering. Both requested seed and RoboTwin's actual `ep_num` are retained; native unstable-scene retries are visible. Initialization failures remain failures, not zero-success policy trials.

Videos are query-aligned head-camera previews (one frame before/after each chunk, played at 4 fps). The simulator interpolates a full action chunk through TOPP and can stop early on success. Submitted action slots are recorded separately from executed masks; terminal-success prefixes are unknown and excluded from executed-only shape ranking. Raw predictions remain in NPZ and the gallery.

`rank_curves.py OUTPUT` writes ranking.json and a self-contained index.html; all valid completed episodes remain accessible. Ranking uses a fixed three-chunk median, sustained high regions, transitions, and residual noise; it is a visualization heuristic, not evidence of causal importance. Raw DV is exactly the existing `compute_endpoint_variance`: population variance over final three endpoints, **sum** across the 14 active dimensions.

CPU checks: `python -m unittest discover -s tools/pi05_dv50 -p 'test_*.py'`. Production acceptance also decodes all sixteen previews and checks trace shapes, finiteness, and distinct actual seeds.
