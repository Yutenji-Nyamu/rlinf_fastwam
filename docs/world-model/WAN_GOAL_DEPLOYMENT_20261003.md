# WMRL 授权改动与下次评估约定 · 2026-10-03

状态：**已部署到深圳3独立副本，未启动训练/评估，尚未GPU验收。** 这次讨论中的双相机世界模型、PACE及替换后端均未实施。

## 已部署的最小改动

|项目|旧值/行为|新值/行为|
|---|---|---|
|actor micro batch|128|64|
|global batch|2048|2048，保持；四卡累积4→8|
|保存间隔|40轮|10轮|
|训练阶段交接|先收齐轨迹，env可能仍在卸载|actor训练前等待env_handle完成|
|资源监控|缺连续资源记录|新启动器自动带起本run监控，并把边界日志目录传入Ray worker|

配置逐字段比较仅有 `actor.micro_batch_size` 和 `runner.save_interval` 两项变化。N64/G8/R8、L320/C8/H10、学习率、更新次数、KIR、奖励、相机、模型、1000轮总预算与原seed保持；没有把双相机研究混入现有训练。内置 `val_check_interval=-1` 保持，原生评估另排。

部署目录：`/data/chenyiteng/projects/wan-goal-sz3/deployment-20261003-memory-fix/repo`；独立分支 `codex/sz3-wan-goal-repair-20261003`，从已发布 `23acc5e` 起。

下一次正式运行必须选此目录及 `local_scripts/wan_goal_20261003/private_ray_driver_monitored.py`；使用原正式配置名 `wan_goal_pi05_headonly_formal_sz3`。此启动器只接受新run的资源owner环境、物理4–7和私有Ray地址；本次没有创建新借卡owner、没有设置实际运行端口，也没有重放旧v6或旧评估owner。最近续训候选CP120保留，完整optimizer恢复及GPU峰值仍待下次诊断，不能称已恢复到139轮。

## 监控怎么工作

- 每10秒记录物理GPU总量/已用/利用率、GPU进程显存、自有进程RSS/Swap及系统可用内存；PSS每60秒。
- actor、rollout、env阶段记录torch allocated/reserved/peak及进程RSS；actor/rollout逐轮重置peak，env为进程累计peak。
- `WAN_GOAL_RESOURCE_DIR=<run>/resources` 同时在driver和Ray job的runtime_env显式设置，worker不会只依赖隐含环境继承。
- 外部监控指向内层run（含launch.json/managed-identities.json/wm-exit.json），写`resources.jsonl`，继承同一个owner身份；driver退出后终止自己的监控子进程，owner终态清理仍是原有唯一流程。没有另设永久守护，也不管理其它实验。
- 不初始化CUDA、不做强制同步；监控仅观察。新WMRL未启动，因此当前没有持续WM资源曲线。

## 已记下的评估约定

**C8下分别评单主相机和主＋腕双相机。** 每10轮完整checkpoint触发独立评估安排，资源可用时排队；基于已跑通的原生Pi0Eval/20环境×25批，每种相机500回合、10任务各50、相同初态、L320/H10/M5。两种相机分别报告，不合并成一个成功率。

先核同协议原始SFT基线，再比较训练检查点。旧单图C8原始0/500的问题仍待定位；不重放旧失效评估适配器。旧双图C5成绩保留为历史参考。可机读约定：[evaluation_plan.json](../../local_scripts/wan_goal_20261003/evaluation_plan.json)。本轮仅落盘协议，没有新建或启动评估owner/队列。

## 验证与证据

- 服务器CPU验证：原流程缺屏障可复现；新流程env完成后actor才开始；env异常阻止actor；telemetry不初始化CUDA。
- 新driver的runtime_env提取执行检查通过，新增目录成功传递且原env_vars保留。
- SZ3外部监控只读采样通过；未借GPU进行模型前反向。
- 原r6 checkout受保护文件SHA逐项保持；未改其已有两份输入适配dirty，未启停现有RLT或共享Ray。
- 轻量证据发布入口：[deployment-20261003](../experiments/wan-goal-sz3-20261003/)。完整本地回执位于`E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/wmrl-deploy-20261003/`。

## 研究与历史

- [139轮审计和500回合结果](audit_20261003/REPORT.md)
- [奖励和6.51%的准确含义](discussion_20261003/reward_signal.md)
- [WoVR/PACE与四套件](discussion_20261003/wovr_suites.md)
- [双相机公开方案](multiview_20261003/public_options.md)
- [现有Wan自行扩展可行性](multiview_20261003/local_feasibility.md)

本轮按用户授权补推源码、配置、研究文档及轻量证据；权重、原始视频/日志和凭据不进Git。Git交付不等于大检查点的独立云备份。
