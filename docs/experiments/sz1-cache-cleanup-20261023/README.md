# Shenzhen1 cache/log cleanup and distillation timing

2026-10-23. User-authorized cleanup removed 2,321 inactive Ray worker log files (21.38 GiB) and 174 exclusively linked UV cache files (20.01 GiB). Total: 2,495 files, 41.39 GiB. Checkpoint, replay and video files were not removed. Existing environments shared via hardlinks were preserved.

Targets were checked against exact paths, UID, inode, size, mtime, link count and open handles before unlinking. Ray log targets were older than 24 hours with absent worker PIDs; active logs and Ray services were preserved. UV deletion was limited to exclusive files at least 16 MiB.

After cleanup: data approximately 184 GiB free, home approximately 196 GiB. Full private exact-path receipt remains with the maintenance transaction; this public receipt contains only counts, sizes and timing.

At 17:14, both formal students were at update 556 while their first evaluations continued: endpoint 18/50 tasks processed (17 valid), OneDP 20/50 (19 valid). Both skipped the existing open_laptop simulation error. Other tasks continued. Missing simulation results are not scored as policy failures.

|Method|Full-budget update time|Evaluation rounds|Estimated evaluation time|Evaluation share|
|---|---:|---:|---:|---:|
|Direct N10 to N1|105.2 h|8|48-64 h|31-38%|
|OneDP|38.5 h|9|54-72 h|58-65%|

The estimate uses measured first-block update throughput and a provisional 6-8 hours per 50-task evaluation. Each round has 50 x 16 episodes, four task lanes with 16 environments each. The first round is still partial; this is a total-budget estimate rather than remaining time. No evaluation settings or training schedules were changed.
