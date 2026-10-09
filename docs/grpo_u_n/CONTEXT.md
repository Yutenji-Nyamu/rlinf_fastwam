# GRPO U / Norm implementation context

User scope (2026-10-09): implement U/Norm with existing GRPO DVCA intervention, temperature and two independent tricks; server checks, GPU6/7 smoke, push; **no formal training**. Restore original per-card RLT after both smokes and exact release.

Base: SZ2 Clean256, commit cd1753ee63b5f68214cebe383be6c871dc2473bb, turn_switch, original Sidney pi0.5. It ran over100 complete rounds. 64 environments x4 epochs =256 trajectories/round, group8, update epochs2, B512/micro32, C50/D14, 200 physical actions, seed42, original Flow-SDE noise0.5. Preserve task, seeds, reward, GRPO normalization, clipping, optimizer, filters and training budget. Original Clean observed DVCA without weighting its loss.

Later move_can_pot256 used32x8 after host-memory failures. This implementation uses the directly verified earlier turn_switch64x4 Control.

Current ownership: BC/EXPO were stopped by another user-authorized task. Its reserved67 receipt removes SZ2/3 GPU6/7 from the owner. Live SZ2 read found no C/G processes on6/7. Shared Ray stays intact.

References: existing GRPO endpoint telemetry -> forward_inputs -> frozen global scene groups -> two-level action-advantage weights; RLT-Q b1d2d8d/348005d for same-noise ODE10/5 and last-five-step/deepest-three-layer Norm; existing DVCA exp controls for tau/dropout/alpha. No new optimizer, replay, scheduler, resource cap or dataset.
