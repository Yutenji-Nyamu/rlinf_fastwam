# 深圳2只读核验：训练保持、RoboTwin state源

**历史只读快照（2026-10-04补注）：**本页只记录2026-10-03 18:20–18:30的深圳2现场及RoboTwin源码核验，不是当前服务器状态，也不代表深圳3 WMRL正在训练。当前WM工作路由见[组合主上下文](../ROBOTWIN_OPENDW_PIPELINE_CONTEXT.md)；资源变更前必须重新核实现场。本机原始回执和只读脚本路径仅作追溯，未随本页公开。

2026-10-03，时区Asia/Shanghai。用户最新要求：深圳2的4–7卡有我们的训练就行，继续研究。使用既有固定host-key Paramiko，以chenyiteng登录；核UID20001、主机h100-gpu02一致。密码仅进程内使用，未写入脚本/文档/回执。没有训练、资源或配置变更。

## 训练现场

| 项目 | 18:20:44首次刷新 | 18:30:25最终刷新 |
|---|---|---|
| 实验 | EXPO-FT，turn_switch | 同一实验/owner/driver |
| GPU | 物理4–7均为PID1089953，UID20001 | 同一进程占用4–7 |
| owner | 4086166/start381843377，活 | 身份匹配，活 |
| driver | 1089953/start382155251，活 | 身份匹配，活 |
| 已完成回合/策略动作 | 77 / 12030 | 77 / 12030 |
| 已完成学习调用 | 252，253进行中 | 254，保存阶段checkpoint_started |
| status新鲜度 | 当前状态 | 18:30:22更新，采样时约3秒 |
| GPU显存 | 4:51510MiB；5–7各38153MiB | 4:51512MiB；5–7各38153MiB |

十分钟间新增两次学习调用，证明该区间有训练推进。最终状态为保存开始，不据此声称新CP已完成验证。瞬时GPU利用率0%不代表卡空闲；驻留进程、唯一owner及学习进度证据一致。

权威控制目录：`/data/chenyiteng/projects/expo-ft-sz2-20261001/eval10-continuation-20261003`；训练目录：`/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-repair-20261002/run`。active-continuation指向该新owner链，旧current/final不用于接管。维持20k动作总预算、200每回合、每10回合20样本评估等现有配置；本轮无恢复/归还操作。

这是深圳2 EXPO，不能当作深圳3 LIBERO WMRL在训的证据。后者旧r6完成139轮后OOM归还的状态引用同日已核历史，[部署专题](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/21b5d590a9116d94e38c139a3ec4aa9723abbcc3/docs/world-model/WAN_GOAL_DEPLOYMENT_20261003.md)为后续路由；本轮没有刷新深圳3或启动WMRL。

本次不是全机故障审计，未新查内核错误/磁盘，也不将身份与进度核验扩称所有系统指标健康。

## RoboTwin源代码

源根：`/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support`；Git HEAD `0008ae6800df9f75fc8de7098bacb01735fd8fd2`，tracked clean。只读源文件，无import/仿真执行。

| 文件 | SHA256 | 已核事实 |
|---|---|---|
| envs/_base_task.py | aa9d717ad214b9d68c6eeb83ed03f2491db6addc98deaa988a48a3c074178bdd | get_obs调用get_left/right_arm_jointState组成joint_action.vector |
| envs/robot/robot.py | 1f4d0284eaf47fe5de2c0ba98a173b78beb83f2f2cc43409d37072a4774df4c8 | 非real getter读取drive target与缓存夹爪命令；real getter才读取qpos |

这是“策略state不等于实测qpos”的实时证据。另已用本地固定Sidney Git对象核从该字段到策略states的直通链，见[接口审计](interface_audit.md)。它缩小了OpenDW next-state缺口，并不证明末请求action在所有提前终止/控制失败情况下都等于nextstate。

## 证据与复核入口

- 首次状态原始回执（本机留存、未发布：`E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/robotwin-pipeline-20261003/sz2/status-first.out`）
- 最终状态原始回执（本机留存、未发布：`E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/robotwin-pipeline-20261003/sz2/status-final.out`）
- RoboTwin源摘录（本机留存、未发布：`E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/robotwin-pipeline-20261003/sz2/robotwin-state-source.out`）
- 只读脚本：sz2_probe.py（本机留存、未发布：`../../../local_scripts/robotwin_pipeline_20261003/sz2_probe.py`）、sz2_source_probe.py（本机留存、未发布：`../../../local_scripts/robotwin_pipeline_20261003/sz2_source_probe.py`）。没有可重放的训练启动器。

当前研究主入口：[组合上下文](../ROBOTWIN_OPENDW_PIPELINE_CONTEXT.md)。本轮新增研究文档本地维护，未另作Git发布或模型云备份。
