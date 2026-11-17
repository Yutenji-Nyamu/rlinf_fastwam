# RLT at the critical phase

The reference VLA executes the initial part of the episode. The RLT actor takes over after the configured simulator condition becomes true, once learning warmup is complete. The condition is latched until reset. Only transitions that start in the critical phase enter RL replay, including reference transitions collected during warmup.

This follows the critical-phase training protocol in [Physical Intelligence's RLT](https://www.pi.website/research/rlt) and reuses RLinf's existing `SimulatorRLTRoute`. A simulation task supplies the phase flag instead of adding a learned handover classifier.

For Dojo pouring, the initial condition uses the native `RewardManager.is_lift` / `Func_Parser.is_lift` predicate for the bottle: height exceeds its reset height by 0.05 m. This is the simulator predicate's default threshold; using it as the handover point is our task-specific adaptation. The reference controls approach and pickup, while RLT trains the subsequent pouring motions. The native task's success criterion and full episode budget remain unchanged.

The final success rate still includes every evaluation episode. A second metric records how often an episode reached the critical phase. Actor/Q/optimizer/replay start fresh; the trained frozen RL-token encoder is reused. Previous fixes (FP32, Q-gradient isolation, output feature normalization, target bounds, fixed reference replay and BC protection) remain enabled.

CPU tests check reference/actor selection before and after warmup and in evaluation, and check that only critical-phase transitions are recorded. They also exercise the actual EnvWorker train/evaluation reset messages through the route: `rlt_switch_flags` travels in the existing dedicated message field. A GPU smoke verifies that the native predicate triggers, the selected transitions reach replay, the actor and critics update, and a checkpoint is written. This change retains the existing checkpoint format.

Native source: [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo), `env/reward_manager/reward_manager.py`, `env/reward_manager/func_parser.py`.
