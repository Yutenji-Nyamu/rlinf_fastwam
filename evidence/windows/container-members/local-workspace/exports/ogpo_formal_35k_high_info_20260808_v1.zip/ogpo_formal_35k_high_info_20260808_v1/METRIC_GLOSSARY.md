# Metric glossary

- `rows`: replay-valid primitive RoboTwin transitions, not action chunks or optimizer steps.
- `wave`: one outer runner cycle with eight parallel train episodes followed by the earned updates.
- `paired update`: one actor update plus one critic update.
- `train_success_once`: success fraction among the eight train episodes in one wave.
- `fixed-policy eval`: 20 evaluation episodes using the online actor at a frozen checkpoint in training time.
- `ratio`: mean same-chain probability ratio between online and EMA actors for a whole update burst.
- `success BC`: flow-matching behavior cloning on successful replay trajectories.
- `cgroup_current_bytes`: physical/accounted container memory, including page cache and checkpoint serialization transients.
