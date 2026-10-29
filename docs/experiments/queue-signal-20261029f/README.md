# Queue update and teacher/student signal analysis · 2026-10-29

- [Queue protocol](../../small_flow_distillation/31_CP_V2_400_FQL_FIX_200_QUEUE_20261029.md): CP v2 200→400 replaces old CP; FQL-fix 200 budget epochs before RLT. 29 CPU checks passed; existing training preserved.
- [Signal findings](../../student_teacher_signal/05_ALL_TRAJECTORIES_CORRELATION_20261029.md): all 12 successful trajectories; all32/5820-action aggregate analysis. Original relative-dispersion rho0.089; supplementary absolute-dispersion rho0.369. No causal claim.
- [Interactive gallery](plots/index.html): serve this folder with a local static server. PNG/PDF/SVG/CSV are in `plots/`. Local navigation to earlier dashboard/video pages requires the original localhost visualization root.
- `analyze_signals.py` and `supplement.py` reproduce the CPU analysis from the original server batch. Source paths and font path are explicit; no weights, videos or raw model input arrays are included.
- `storage-own-summary.json`: our read-only storage summary and proposed cleanup tiers. No deletion this turn. Named-user inventories remain local.

Formal config `epochs` for FQL counts replay sampling epochs; `method_budget_epochs=200` is the comparable direct-student budget, 2200 updates. All plots use training budget epochs as the user-facing axis.
