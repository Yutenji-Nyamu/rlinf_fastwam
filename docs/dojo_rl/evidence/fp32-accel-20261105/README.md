# 验收证据

- `live-receipt.json`：采集时间、四个正式 owner、完整 smoke/resume checkpoint 回执、六个源码哈希、RLT 预约及 FQL 汇合。
- `final-audit.json`：正式预算与加速开关、训练 rollout 使用 actor.model 的源码确认、进程和后备预约核对、Dojo 物理步数。
- 各槽位 `formal.yaml`、`fp32-model-contract.json`：实际配置和真实模型精度/更新/保存检查。
- 两个 +U 的 `sac-timings.json`、`component-timings.json`：全部候选结果，含失败项和基线重复；`benchmark-input.json` 记录仅供丢弃式测试的 checkpoint/replay 来源。
- `formal-tail.txt` 是末段小日志，发布副本仅去除终端颜色码和行尾空白，便于阅读；完整原日志仍留服务器。

数值和速度结论的适用范围见上级 [执行报告](../../24_DSRL_FP32_ACCEL_20261105.md)。不包含登录凭据、RPC 密钥、权重或回放内容。
