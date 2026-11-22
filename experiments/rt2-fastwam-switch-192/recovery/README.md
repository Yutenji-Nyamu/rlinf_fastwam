# RoboTwin FastWAM GRPO recovery

The turn-switch environment worker aborted during a scene reset. Each simulation instance now performs initialization, simulation, observation and cleanup on one dedicated thread. Separate instances still run in parallel. The training recipe and reset seeds are preserved. The CPU lifecycle test covers reset, step, observation, seed checking, close and reinitialization.

Training resumes from the complete two-rank DCP at round 15, with a target of 100 rounds. Four RLT continuations retain their full-state checkpoints and share the slot queue with the same no-save fallback. Retention skips externally linked checkpoints before pruning and keeps the existing containment check for local checkpoints. Tested with a real cross-directory symlink and a protected weight file.
