# RLT / DSRL formal 轻量现场包

生成日期：2026-08-24  
用途：快速审阅训练指标、资源、配置和 RLT Step 25 checkpoint 卡点；不用于恢复训练。

## 终态边界

- RLT Stage 1：2,000/2,000 完成；本包含完整标量 CSV、TensorBoard event、resolved config、资源和图。
- RLT Stage 2：完整到 Step 24；Step 25 eval 后卡在首次 FSDP/DCP optimizer-state 保存；checkpoint
  incomplete、不可恢复。本包含 driver、TensorBoard event、resolved config、资源、解析 CSV 和卡点图。
- DSRL：本包原始日志/图表抓取到 Step 146；文档中的后续只读 live 现场已刷新到 Step 154/200、Step 155
  训练中。它仍在服务器自然运行，终态包待 Step 200 后生成。

## 包含

- `docs/`：并发运行层、指标资源和 checkpoint 定因短文档。
- `ledgers/`：逐操作与证据边界。
- `figures/`：五张主要训练/资源图。
- `tables/`：逐步指标 CSV、summary JSON。
- `raw/`：driver、TensorBoard event、resolved YAML、资源 CSV 与只读现场文本。

## 明确排除

- checkpoint/model/replay 正文；
- RoboTwin 视频与数据；
- 整个 Ray session；
- 密码、token、私钥或代理凭据。

