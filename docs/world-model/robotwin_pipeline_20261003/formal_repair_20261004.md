# OpenDW 正式启动修复 · 2026-10-04

上轮 `formal-v1` 在启动检查阶段退出，原归还链已恢复 RLT。故障发生在图形范围检查读取 `/proc/<pid>/environ` 时：系统登录辅助进程的环境不可读，旧检查直接抛出 `PermissionError`。这次修复只允许经 PID/start/UID、父进程、命令和 GPU 上下文核验的既有系统辅助进程；未知或身份改变的进程仍会阻断启动。

服务器 38 项 CPU 检查已通过；随后在 RLT 归还环境中允许并传递图形日志键，13 项针对性回归全部通过。新的 `formal-v2` 使用独立借还周期、输出和 namespace，先按正式并行跑一个动作块，完成训练、原生评估和保存，再由同一 owner 自动进入正式。两阶段之间保留两个 WM 服务并保持 RLT 暂停；运行故障仍由原有精确清理与归还链处理。

| 项目 | 启动 smoke | 正式 |
|---|---|---|
| GPU／并行 | actor、rollout 两 rank 在 4/5；环境两 rank、WM 服务在 6/7 | 同左 |
| WM 采样 | N64/G8/R1，L32/C32，一轮 | N64/G8/R8，L384/C32，最多 200 轮 |
| actor | global64、micro8、U2 | global2048、micro8、U2 |
| 原生评估 | N32/R1/G1、三相机、C32/L32 | N32/R1/G1、三相机、C32/L384 |
| 保存／评估 | 第 1 轮执行 | 每 10 轮执行 |

任务仍为 `adjust_bottle`，使用同一 π0.5 SFT、OpenDW、WorldArena RM 和 GRPO 配置。除了上述串行工作量、保存评估安排及独立输出，smoke 与正式配置必须完整一致。smoke 要求进程正常结束、卡位验收和 driver 完成回执；不增加非零梯度门槛。L32 原生评估用于检查调用链，不作为正式成功率。

**14:37:16（北京时间）已核短 smoke 通过，正式采样运行中。** 同一 owner `3242048/start706364464/UID20001` 健康，当前 `phase=formal`。短 smoke 于 14:15:07 以 exit0 结束，完成 64 条训练轨迹、32 条原生评估、优化器调用及保存；两份分片和完整权重共 3 文件、约 28.83 GB。`grad_norm=0`，本次通过证明训练／评估／保存调用链可运行，尚未证明有效学习。

正式约 14:15:30 自动接上，保持 N64/G8/R8、L384/C32、global2048/micro8/U2、最多 200 轮，保存和原生评估每 10 轮；当前日志已完成首轮 1/8 采样批次，尚未完成正式参数更新。计算及图形进程保持物理 4–7 卡，0–3 无 C/G 进程（各卡仍有约 4 MiB 驱动基础占用）；RLT 保持暂停，WM 真正执行失败时按原链归还。

输出为 `S/runs/formal-v2`，控制目录为 `S/formal-control-v2`，其中 `S=/data/chenyiteng/projects/opendw-robotwin-smoke-20261003`。启动所用 `prepared/plan.json` SHA256 为 `4928516ae72315b8c83ac42ccb9064d41b1b40d6760228fe858bd995b296d893`。[初始启动回执](../publication_formal_repair_20261004/runtime_receipt.json)保留启动身份、CPU 测试和来源摘要；[14:37 验收回执](../publication_formal_repair_20261004/acceptance.json)记录短 smoke 完成及正式采样状态，其原始现场 JSON 为 `opendw-oct4-v2-accepted1.out`。

修复源码、初始文档及轻量回执共 11 文件已推 `codex/robotwin-opendw-wmrl-20261003@cf515063c6ab827f6605b8c5824d9db8ef528664`，远端 SHA 已核同；训练 checkout、检查点和原始日志保持原状。本文 14:37 更新及验收回执使用独立轻量发布包 `publication/formal-accepted-20261004-v2`。
