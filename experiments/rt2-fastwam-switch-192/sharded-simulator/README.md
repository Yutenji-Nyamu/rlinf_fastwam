# RoboTwin simulator recovery

FastWAM GRPO resumed from round 30. Its environment actor had aborted during scene teardown with a CPython thread-state error. Pinning each Python SubEnv to a thread did not keep the native crash out of the training process. Four RLT continuations independently reached the 900-second simulator RPC timeout. The first two had already saved and rotated additional complete checkpoints.

Both methods retain their original RoboTwin tasks, simulator settings, action chunks, rollout budgets and training algorithms. The Ray environment actor sends contiguous slices of the batch to disposable simulator processes. Each process executes its scenes on its main thread and holds its renderer until offload. The parent gathers actions, observations and terminal fields in the original lane order. The two FastWAM ranks each use two processes for 16 environments. Each single-card RLT job uses four processes for 48 environments.

The RPC transport is reused from the RLT deployment. A transport failure during a full, explicitly seeded reset can restart the vector once. A failed action or partial reset propagates to the runner. A simulator-related formal-run failure may retry once from a complete checkpoint before the existing queue advances. RLT restores the replay and trainer state together with the model.

The Python test exercises the real IPC with a deterministic simulator double, including child loss. The GPU probes use real scenes with all three cameras. The small-probe timings include cold starts; the initial four-lane FastWAM probe preceded the renderer-lifetime adjustment. Formal-width probes use the RLinf environment adapter and its native action-chunk path. These timings describe recovery tests, not an evaluation of training speed or success rate.

## References

- [SAPIEN 3.0.1 renderer engine lifetime](https://github.com/haosulab/SAPIEN/blob/3.0.1/src/sapien_renderer/sapien_renderer_system.cpp)
- [SAPIEN 3.0.1 Python renderer binding](https://github.com/haosulab/SAPIEN/blob/3.0.1/python/pybind/sapien_renderer.cpp)
- [Gymnasium process-based vector environments](https://gymnasium.farama.org/api/vector/async_vector_env/)
- [RoboTwin task and scene lifecycle](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/_base_task.py)
- [CPython 3.11 thread-state API](https://docs.python.org/3.11/c-api/init.html#c.PyGILState_Release)
