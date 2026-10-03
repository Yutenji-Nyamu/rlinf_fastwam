# OpenDW 直接正式训练 · 2026-10-04

用户本轮最终决定：停止长 smoke，直接正式；smoke→正式期间 RLT 保持暂停，WM 发生故障后才接回。用户将关闭本机窗口，因此启动、监控与异常归还由服务器进程承担。此前“等待长测／CP1／有效梯度后再转正式”的方案取消。

**00:59:46 已完成直接交接。** 31 项服务器 CPU 测试全部通过；正式 owner `3908112/start701780765/UID20001` 已在 `S/runs/formal-v1` 启动，其中 `S=/data/chenyiteng/projects/opendw-robotwin-smoke-20261003`。交接监督回执 `transferred=true`、`failure=null`、`recovery=null`；RLT 全程保持暂停，没有恢复后重新借卡。截至 01:01:48，阶段为 `service_cpu_load_wm6`，WM6 为 `loading_cpu`、心跳正常、未报错；GPU0–7 上下文列表均为空。正式任务仍在启动加载，尚未核验正式采样或参数更新。[最小运行回执](../publication_formal_20261004/runtime_receipt.json)

| 项目 | 正式配置 |
|---|---|
| 任务／策略／环境 | adjust_bottle；同一 π0.5 SFT；OpenDW＋WorldArena RM＋GRPO |
| GPU | 4/5 actor、rollout；6/7 两个 OpenDW 服务 |
| 采样 | N64/G8/R8，每 runner 轮 512 条轨迹；G 已包含于 N |
| 时间 | H50，执行 C32，最长 384 动作／12 块 |
| 优化 | global2048、micro8、U2；每轮 6 次优化器调度 |
| 预算 | 继承 Control max_epochs1000/max_steps200，实际最多 200 轮 |
| 保存／评估 | 每 10 轮保存；每 10 轮原生 RoboTwin N32/R1/G1，三相机/C32/384 |
| 起点 | 同 SFT 新正式输出；中断 smoke 的部分轨迹不冒充正式首轮 |

今晚取消非零优势／梯度准入，也不新增 GPU 测试轮次。原生环境直接在正式启动中初始化，初始化或运行异常按同一归还链处理。全过滤／零梯度属于明天需分析的学习指标，不作为今晚自动中止的条件；OOM、进程失败和越卡等执行错误仍停止 WM 并归还。

交接沿原已停 RLT 的完整 CP125/125/75/75 和各自源代码／参数，不先恢复再借卡。旧 owner 正常清理 WM；交接监督持借还锁接过责任，新正式 owner 从已停凭据继续管理资源。专用交接意图、清理回执、原 owner 退出和新 owner 接管均单独记录。异常时只处理本任务 PID/start/namespace，保留共享 Ray 和其他任务。

原 N64/L32 smoke 已 exit0（1610 秒），512 条均被奖励门限过滤、梯度为 0，不能称有效学习。已有 10 条演示的 CPU RM 首中末帧核验中，末帧 8/10≥0.1、4/10≥0.9；末帧无独立成功标签，不称分类召回率。详细结果留原始记录。

本轮新实现及轻量运行回执继续推 `codex/robotwin-opendw-wmrl-20261003`；权重、检查点和原始视频不放 Git。
