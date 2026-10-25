# C10 distillation update, 2026-10-25 21:58 Asia/Shanghai

The 1113-update evaluations completed: direct endpoint distillation 101/368 (27.4%), OneDP 85/368 (23.1%). Matched original N1/N10 references are 75/368 (20.4%) and 153/368 (41.6%). Each condition attempted 50 tasks with eight episodes; 46 tasks were valid and four simulation failures remain missing, not scored as policy failures. See points-20261025-2158.csv for completed early curves.

Both runs continued: direct 1470/55650 and OneDP 1550/22260. H50/C10, global batch384 (192 per GPU), frozen VLM and no-recording evaluation are unchanged. RLT fallbacks remain waiting behind the two priority jobs. No FQL training was started; data and multi-task planning are documented in ../../pi05_distillation/44_FQL_OFFLINE_MULTITASK_DATA_20261025.md and 45_FQL_MULTITASK_REFERENCES_REINFLOW_20261025.md.
