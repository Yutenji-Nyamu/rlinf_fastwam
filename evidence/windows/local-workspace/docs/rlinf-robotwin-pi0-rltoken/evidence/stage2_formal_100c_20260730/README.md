# Stage 2 formal 100-cycle 启动证据

本目录是 2026-07-30 RoboTwin `adjust_bottle` RLT Stage 2 formal pilot 的小型启动证据副本。

## 文件

- `source_config.yaml`：fail-closed source config；
- `resolved.yaml`：绑定 `max_steps=100` 后的完整 resolved config；
- `exact_command.txt`：实际训练命令；
- `run_provenance.tsv`：Git、配置、artifact 与路径 provenance；
- `stage1_binding_preflight.json`：accepted Stage 1 model/manifest/stats 绑定检查；
- `resources_before.txt`：启动前 Git、进程、GPU/RAM/disk 快照；
- `started_at.txt`、`driver_pid.txt`、`monitor_pid.txt`：实际后台作业身份；
- `launch_failure_self_match.txt`：第一次 launcher 自匹配、未启动训练的失败证据；
- `launch_health_summary.json`：启动后132秒的只读健康摘要；
- `SHA256SUMS.txt`：服务器原始/下载文件的 SHA-256。

增长中的 `driver.log` 与 `resources.csv` 留在服务器 runtime 目录：

`/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1/runtime`

本目录不包含 checkpoint，也不把启动快照描述为训练完成结果。
