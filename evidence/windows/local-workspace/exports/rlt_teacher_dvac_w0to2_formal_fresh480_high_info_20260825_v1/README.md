# RLT teacher-DVAC `[0,2]` fresh-480 high-information archive

- Status: complete, exit code 0, cycle 480; finished `2026-08-25T15:48:56+08:00`.
- Training rollout mean: original RLT `60.86%`, DVAC `53.65%`, delta `-7.21` pp.
- Last-100 training rollout mean: original `91.75%`, DVAC `94.88%`.
- Final fixed20: original `85.0%`, DVAC `85.0%`.
- Peak resources: cgroup RAM `126.66` GiB; GPU0/1 `19.37/19.51` GiB.

`analysis/` contains the final comparison plot, DVAC diagnostics, cycle tables and machine-readable summary. `raw_server_evidence/` keeps logs, resolved config, resource CSV, small telemetry traces and checkpoint manifests. Model/optimizer/replay bodies, videos and full Ray logs are intentionally excluded.
