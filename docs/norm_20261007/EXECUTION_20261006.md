# 深圳3机 Norm 执行记录

2026-10-07 01:13：BC Norm、DSRL Norm 均已进入 FORMAL，正在初始化。用户明确要求 smoke 能跑通即可；不再等待额外轮数、warmup 或 smoke checkpoint。两组结束 smoke 后，已核实专属 namespace 及 GPU6/7 释放，再从原 SFT、空回放启动正式任务。

| 任务 | 卡位 | 正式配置 | 输出目录末级 | driver PID/start |
|---|---|---|---|---|
| BC Norm | 物理 GPU6 | N8/U5、300轮、global1024/micro32；固定32条评估每5轮、每10轮保存；成功≤3chunk | `bc-formal300-v1` | 3182779/727781893 |
| DSRL Norm | 物理 GPU7 | N4、C20、200轮、GB/MB256、UTD20、warmup500；固定12条评估每13轮、每65轮保存 | `dsrl-formal200-v1` | 3182804/727782101 |

相对深圳1机实际运行配置，仅换 Norm 信号和部署路径、卡位、独立输出。任务、SFT、种子、动作预算、优化器、评估保持。DSRL 原始 `max_epochs=1000` 非绑定上限保持，实际 `max_steps=200`；BC 正式 `signal_spec` 明确为 Norm。逐叶差异随验收文件发布。两组不继承 smoke 数据或权重。

Norm 定义与原加权机制见 [SIGNAL_CONTRACT.md](SIGNAL_CONTRACT.md)。服务器 CPU 检查 BC52、DSRL29 项通过；BC 回放隔离最后修订复检21项通过。实际模型 probe 两组均通过，确认 Norm 不改变主动作和 RNG，每次仅原有10次前向。BC smoke 完成过真实非零梯度更新，DSRL smoke 已产生265条真实回放；未等待原 warmup500 后的 SAC 更新，也未要求 smoke checkpoint。以上界限不改变正式学习参数。

源码分支已推送并核远端 SHA：

- `codex/sz3-bc-norm-20261007`：`a37b482aa8ad139a4f01eb5562728c00bcae709f`。
- `codex/sz3-dsrl-norm-20261007`：`6c8756b672d991fe21e52cad6023fcdb02c35f5a`。

唯一控制根：`/data/chenyiteng/projects/norm-bc-dsrl-sz3-20261007/control`。owner PID3651109/start727565004/UID20001，boot `47e64b8d-f02b-4dc3-bcdc-e37000390e2b`；以 `status.json`、`heartbeat.json`、`resources.json` 为动态依据，勿重复启动 owner。

入口为 `/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -B /data/chenyiteng/projects/norm-bc-dsrl-sz3-20261007/ops/driver.py`，参数 `--control` 指向上述控制根，`--request` 分别指向其 `requests/bc-formal-v1.json`、`requests/dsrl-formal-v1.json`。已由 owner 执行，不重放。

实际命令及身份在各自 `runtime/launch.json`、`driver-identity.json`；实配和环境在 `resolved.yaml`、`environment.json`。输出根 `/data/chenyiteng/results/norm-bc-dsrl-20261007`，日志在各任务 `runtime/driver.log`，指标在 `tensorboard`。正式 namespace 分别为 `norm-sz3-bc-formal-1007-v1`、`norm-sz3-dsrl-formal-1007-v1`。

原 RLT6/7 已各保留 CP150，恢复合同在项目根 `rlt-after-norm-g67-v1`；开发、smoke、正式期间保持候补，两组正式完成并验证断点、释放专属进程后恢复原累计预算。GPU4/5、其他用户及共享 Ray 不由本任务启停。

运行停止条件：源码/配置 pin 不符、身份或资源归属不明、越界计算/图形上下文、非有限训练值、磁盘不足或最终保存不完整。异常保持租用以便处理，不让低优先级 RLT 插队。

历史说明：最初 smoke 配置为 BC2轮、DSRL14轮，包含完整学习和保存门槛；已按用户最新要求提前结束。主动 SIGTERM 在 Ray 清理阶段产生 driver 退出码15、外层-6；两组清理记录均 `cleanup_error=null`，专属 namespace/卡位释放经过独立确认，随后才提交正式请求。
