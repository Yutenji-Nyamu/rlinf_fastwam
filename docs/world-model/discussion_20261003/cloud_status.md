# WMRL 云端发布状态 · 2026-10-03 13:09 CST

本页保留补推前的历史快照；用户后续已授权处理，最新改动与交付见[部署记录](../WAN_GOAL_DEPLOYMENT_20261003.md)。

本轮只读检查，不 commit、push、修改训练配置或重启实验。通过既有固定 host-key 的深圳3连接，先核 chenyiteng / UID20001 / h100-gpu01，再在独立 publication checkout 执行 Git 元数据查询；SSH 已关闭。

## 已推内容

- 仓库：[Yutenji-Nyamu/rlinf_fastwam](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz3-wan-goal-20260930)。
- 分支：`codex/sz3-wan-goal-20260930`。
- 当前远端 SHA：`23acc5eaa2ea749ea5c81bbaec65ac442287d27a`，与独立发布 checkout HEAD 一致，发布 checkout clean。
- 最后提交时间：2026-10-01 16:08:49 +08:00；主题为训练机制、奖励过滤和四卡并行说明。
- 已含两份 π0.5 输入适配源码、正式/smoke 配置、执行与资源借还脚本、10月1日故障修复和轻量回执、运行手册及机制文档。不是仅在本地存了一个 Git 分支。

## 尚未补推的本轮成果

- 10月2日原生500回合评估结果及其新版评估入口。
- 10月3日 OOM、训练动态、奖励/起始数据审计与曲线。
- 新准备的阶段等待屏障、资源监控及CPU验证证据；这些也尚未应用到原训练 checkout，未GPU验证。
- 本目录的并行/频率、奖励机制、WoVR与其他套件讨论文档。

以上以本条 WMRL 发布分支的已核文件树为范围；未把其他研究分支上的文件视为本实验交付。CP40/80/120和原始日志按原交付约定不放Git；现有记录定位在深圳3，未取得独立云存储备份回执，不能称已完成异地备份。

## 证据

- [实时远端、发布文件树及历史回执](E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/wmrl-audit-20261003/cloud-status.json)。
- 查询时间：`2026-10-03T05:09:17Z`。
- Git只读操作：`rev-parse`、`ls-remote --heads`、`ls-tree`、`status --porcelain`、`log`；未 fetch/checkout/commit/push。
