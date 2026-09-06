# 深圳服务器操作流水账索引

本索引按小任务拆分服务器操作记录。凭据从不写入文件；账号、时间、工作目录、完整命令、关键输出、退出码、问题、处理和复测如实记录。

| 阶段 | 账本 | 状态 |
|---|---|---|
| 两账号现场刷新 | [`01_LIVE_AUDIT_20260821.md`](01_LIVE_AUDIT_20260821.md) | 已完成 |
| 官方源码与下载 | [`02_SOURCE_AND_DOWNLOAD_LEDGER.md`](02_SOURCE_AND_DOWNLOAD_LEDGER.md) | 当前阶段已完成 |
| RoboTwin 2.0 / ACT 环境安装 | [`03_ROBOTWIN_ACT_ENV_LEDGER.md`](03_ROBOTWIN_ACT_ENV_LEDGER.md) | 已完成 |
| ACT 数据、训练、评估与视频 | [`04_ACT_PIPELINE_LEDGER.md`](04_ACT_PIPELINE_LEDGER.md) | 全部完成；含 H100/CuRobo 根因、窄修复、单条采集/清理、训练、official eval/video |

历史规划期的本地/来源调查记录仍保留在 [`OPERATION_LEDGER.md`](OPERATION_LEDGER.md)，不再向其中追加新的服务器执行流水。该旧文件名中的“脱敏”只表示没有落盘凭据，不代表另做一份冗余流水；本轮实际服务器操作只记在上表分阶段账本中。
