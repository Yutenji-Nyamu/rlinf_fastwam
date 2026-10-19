# Noise signal, early takeover, and C20 follow-ups

Authorized 2026-10-19 on SZ1 physical GPU4/5, after closing EXPO Clean.

1. Reuse the successful native RoboTwin U/video collector: 50 tasks, 32 episodes
   per task, B16 per GPU, unchanged task seeds/budgets, H50, N10 execution and
   N5 U side pass, head-camera complete video and control/frame mapping.
   Four-noise signal uses the executed native sample plus three additional N10
   samples evaluated together (B48). It is population std/RMS across four
   model-space outputs, averaged over D14, retaining every action position.
   Extra outputs never choose or alter executed actions. Save all four full
   noises, model/env actions, old U, observations, and native video products.
2. CP180 high/low, 15 thresholds each, eligible only at queries0/1; queries2/3
   always weak. Add all-weak, first-two-all-strong, and random eligible controls.
   Refit thresholds on compatible historical first-two U, deduplicated across
   repeated observations, without success labels. Plot actual eligible and
   whole-episode takeover fractions separately. B32/GPU,128/condition,C50.
3. Both weak and strong use C20, with H50/N10/200-action budget unchanged.
   Evaluate original plus the six protected pillbottle RL checkpoints,128 each;
   collect weak-policy U along those trajectories. Select highest successful
   RL candidate (predeclared order breaks ties), freeze C20 thresholds from
   compatible historical/new C20 U, then high/low15 thresholds each. Baselines
   come from the same C20 ranking pass. Selection and same-seed reuse are
   exploratory, not an independent confirmation of the selected winner.
   If no RL candidate beats weak, record that result and skip the dependent
   takeover sweep. EXPO turn_switch weights are not pillbottle candidates.

First real B16 collector batch per card is smoke and remains part of the
1600-episode dataset. Then B32 early/C20 smoke checks new intervention paths.
After acceptance/publication, collect remaining98 batches, early sweep, C20
ranking, and C20 sweep; the unchanged owner restores original per-card RLT.
Noisy native task failures retain outputs and skip without retries. A failed
smoke, signal/video check, or GPU cleanup stops dependent work. Existing lane
limits remain600s stalled/3600s per batch; physics and shared Ray are untouched.

Entry: `prepare.py <pooled-early-calibration.json>`, then
`sequence.py /data/chenyiteng/deployment-20261019/pi05-noise-early-c20-sz1-v2`.
Preparation uses `PI05_FOLLOWUP_VERSION=v2`. The v1 smoke stopped before
simulation because the new archive export cast native float64 actions to
float32. Export now preserves their dtype; v1 evidence is retained separately.
The exact launch needs prepared configs, verified predecessor identity,
graphics/compute environment and the one-time scheduler cutover receipt.
`continue.json` is written by the controlling session only after actual smoke
and public-source publication; it is not a request for another user approval.

Execution and small evidence: [record](../../docs/experiments/pi05-followups-20261019/EXECUTION.md).
Action-level signal/video fields: [plotting handoff](../../docs/experiments/pi05-followups-20261019/PLOTTING.md).
