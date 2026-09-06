# Stage 2 formal 100-cycle completion evidence

Observation time: `2026-07-30T09:54:24+08:00`.

This directory is a bounded, read-only copy and analysis of the completed
RoboTwin `adjust_bottle` RLT Stage 2 formal pilot. It does not contain the
checkpoint payloads or trajectory replay files.

## Primary evidence

- `driver.log`: complete console log for cycles 1 through 100.
- `metrics.log`: RLinf experiment metric log.
- `events.out.tfevents.*`: TensorBoard event stream.
- `resources.csv`: complete 2-second resource monitor.
- `exit_code.txt`, `finished_at.txt`: terminal run state.
- `rlt_completion_step*.json`: completion manifests for all ten checkpoints.
- `rlt_state_rank_*_step100.pt`: final per-rank trainer state.
- `replay_rank_*_metadata_step100.json`: final replay row counts.

## Derived summaries

- `status_summary.json`: machine-readable headline results.
- `selected_scalars.csv`: selected TensorBoard scalar series.
- `resources_30s.csv`: downsampled resource series.
- `visual_data.json`: values used by the plots.
- `success_and_schedule.png`: train/eval success and RLT phase boundaries.
- `optimization_health.png`: losses, Q values, and gradient norms.
- `resource_profile.png`: GPU, RAM, and cgroup resource profile.
- `checkpoint_growth.png`: total and replay-dominated checkpoint growth.

All derived files were generated locally from the copied raw evidence. The
server run remained untouched.
