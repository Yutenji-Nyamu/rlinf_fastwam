# 深圳服务器操作流水账索引

本索引按小任务拆分服务器操作记录。凭据从不写入文件；账号、时间、工作目录、完整命令、关键输出、退出码、问题、处理和复测如实记录。

| 阶段 | 账本 | 状态 |
|---|---|---|
| 两账号现场刷新 | [`01_LIVE_AUDIT_20260821.md`](01_LIVE_AUDIT_20260821.md) | 已完成 |
| 官方源码与下载 | [`02_SOURCE_AND_DOWNLOAD_LEDGER.md`](02_SOURCE_AND_DOWNLOAD_LEDGER.md) | 当前阶段已完成 |
| RoboTwin 2.0 / ACT 环境安装 | [`03_ROBOTWIN_ACT_ENV_LEDGER.md`](03_ROBOTWIN_ACT_ENV_LEDGER.md) | 已完成 |
| ACT 数据、训练、评估与视频 | [`04_ACT_PIPELINE_LEDGER.md`](04_ACT_PIPELINE_LEDGER.md) | 全部完成；含 H100/CuRobo 根因、窄修复、单条采集/清理、训练、official eval/video |
| latest RLinf source、环境与 π0 模型 | [`05_RLINF_SOURCE_ENV_MODEL_LEDGER.md`](05_RLINF_SOURCE_ENV_MODEL_LEDGER.md) | R0/R1已完成 |
| π0 fixed eval、PPO smoke 与 formal-100 | [`06_RLINF_PI0_PPO_4GPU_RUN_LEDGER.md`](06_RLINF_PI0_PPO_4GPU_RUN_LEDGER.md) | 4卡smoke完成；formal完整到Step47、数值finite，因128-env主存持续增长按用户授权停止；最新自然checkpoint/eval为Step40，旧driver/Ray已释放 |
| 双账号、整机、网络与 Fast-WAM 只读刷新 | [`07_SERVER_LIVE_AUDIT_20260822.md`](07_SERVER_LIVE_AUDIT_20260822.md) | 10:28–10:33 CST完成；GPU 0–3空闲、PPO到Step 22并进入23、主存黄灯继续、Fast-WAM断点与quota已刷新 |
| PPO、整机、Git/worktree 只读刷新 | [`08_SERVER_PPO_LIVE_REFRESH_20260822.md`](08_SERVER_PPO_LIVE_REFRESH_20260822.md) | 14:59–15:02 CST完成；PPO完整到Step 33并继续、Step30 checkpoint/eval齐；数值绿，EnvWorker主存同相位增长未平台化，需用户决定是否继续 |
| PPO Step 1–46 曲线与资源复核 | [`09_PPO_CURVES_LIVE_20260822.md`](09_PPO_CURVES_LIVE_20260822.md) | 四张手机可读PNG、CSV/console/event原件；训练成功率约90–92%平台、fixed64最新62/64、优化正常；离散同相位资源点与最新GPU快照显示主存继续逼近极限 |
| GitHub连接准备 | [`10_GITHUB_CONNECTION_LEDGER_20260822.md`](10_GITHUB_CONNECTION_LEDGER_20260822.md) | RLinf repo-scoped key生成阶段的原始账；后续实际认证/remote/worktree见11号总流水 |
| Git认证、remote与隔离worktree | [`11_GIT_WORKTREE_AND_IMPLEMENTATION_LEDGER_20260822.md`](11_GIT_WORKTREE_AND_IMPLEMENTATION_LEDGER_20260822.md) | `rlinf_fastwam` repo-scoped读写认证与3个worktree完成；后续账号级身份、Fast-WAM fork/push闭环见14号账 |
| current π0 GRPO实现与formal | [`12_GRPO_CURRENT_IMPLEMENTATION_LEDGER_20260822.md`](12_GRPO_CURRENT_IMPLEMENTATION_LEDGER_20260822.md) | config-only迁移已测试/commit/push；formal v1因GCS/heartbeat控制面失联终止；同配置v2于10:50 CST最新完整Step29并进入下一步rollout，Step10/20 checkpoint+fixed64齐，数值finite；cgroup约1.55 TiB且尚未平台化。最新现场见16号整机账，曲线报告见上级17号文档 |
| current π0 DVAC观测与扩量 | [`13_PI0_DVAC_CURRENT_IMPLEMENTATION_LEDGER_20260822.md`](13_PI0_DVAC_CURRENT_IMPLEMENTATION_LEDGER_20260822.md) | exact实现已push；fixed-16完成12/16；独立official fixed-64主集自然完成42/64、256 queries、4 NPZ、768 PNG、4 MP4，Ray/GPU释放 |
| GitHub账号级key、fork/push与管理员巡检 | [`14_GITHUB_ACCOUNT_KEY_AND_ADMIN_AUDIT_20260822.md`](14_GITHUB_ACCOUNT_KEY_AND_ADMIN_AUDIT_20260822.md) | 账号key登记/指纹/SSH认证及`gh`登录完成；`Yutenji-Nyamu/FastWAM` fork已创建并普通push`c63dc9b5...`；根/home/data与全用户元数据正常 |
| official Fast-WAM DVAC实现、P1与多任务P2 | [`../../fastwam-robotwin-rlinf-grpo/evidence/SHENZHEN_DVAC_IMPLEMENTATION_LEDGER_20260822.md`](../../fastwam-robotwin-rlinf-grpo/evidence/SHENZHEN_DVAC_IMPLEMENTATION_LEDGER_20260822.md) | 四任务×16全部自然完成，adjust/move/turn/pick=16/11/10/12 success；49/64、503 queries/NPZ、1,509 PNG、64 MP4，GPU3释放 |
| π0/Fast-WAM DVAC离线分析 | [`15_DVAC_OFFLINE_ANALYSIS_LEDGER_20260822.md`](15_DVAC_OFFLINE_ANALYSIS_LEDGER_20260822.md) | 分析器6/6；首版、move-stapler独立phase和最终五source统一分析均exit0，完整派生包核SHA下载/解压，实际结论见上级16号结果文档 |
| 整机管理员巡检、GRPO v2刷新与DVAC教学取证 | [`16_SERVER_WHOLE_AUDIT_AND_GRPO_REFRESH_20260823.md`](16_SERVER_WHOLE_AUDIT_AND_GRPO_REFRESH_20260823.md) | 10:46--10:53 CST只读完成；磁盘/服务正常、GPU0--3空闲、其他用户无GPU作业；GRPO v2完整Step29并继续，数值绿，约1.55 TiB主存/满swap仍为资源黄灯；旧Xid/segfault均与既有已解决事件闭合 |
| GRPO v2 Step36训练、评估、资源与主存归因刷新 | [`18_GRPO_LIVE_REFRESH_20260823.md`](18_GRPO_LIVE_REFRESH_20260823.md) | 13:32--13:38 CST只读完成；完整Step36并进入Step37，Step30 fixed64=62/64，数值finite；cgroup约1.65 TiB、增长变慢但未平台，93.55% PSS来自4个EnvWorker；三张全历史PNG见上级20号文档 |
| DVAC代表原视频取回与细粒度action时间轴重建 | [`19_DVAC_DETAILED_TIMELINE_REBUILD_LEDGER_20260823.md`](19_DVAC_DETAILED_TIMELINE_REBUILD_LEDGER_20260823.md) | 服务器只读盘点+SFTP get；7条Fast-WAM MP4与24张π0 query图闭合；Fast-WAM主口径L5、π0 L3，77张最终图、7 CSV、完整图册与21号教学文档完成；未干预GRPO/推理进程 |
| GRPO v2 Step42训练、评估、资源与主存再刷新 | [`20_GRPO_CURRENT_REFRESH_LEDGER_20260823.md`](20_GRPO_CURRENT_REFRESH_LEDGER_20260823.md) | 15:59 CST只读完成；完整Step42并进入Step43 rollout 2/4，Step40 fixed64=63/64、数值finite；cgroup约1.78 TiB、host available约270 GiB，Step37–42增速回升且未平台；三张全历史PNG见上级22号文档 |
| `[0,5]` Step45收尾与双两卡GRPO并发formal | [`GRPO_DUAL_2GPU_FORMAL_LEDGER_20260826.md`](GRPO_DUAL_2GPU_FORMAL_LEDGER_20260826.md) | `[0,5]`精确停在完整Step45；shared Ray保留。control `4,5/RLinf`与DVAC `[0,2]` `6,7/RLinf_1`均已进入首轮rollout `1/4`，两边fatal=0；pair resolved unexpected diff=0；旧run高信息ZIP已下载 |

历史规划期的本地/来源调查记录仍保留在 [`OPERATION_LEDGER.md`](OPERATION_LEDGER.md)，不再向其中追加新的服务器执行流水。该旧文件名中的“脱敏”只表示没有落盘凭据，不代表另做一份冗余流水；本轮实际服务器操作只记在上表分阶段账本中。
