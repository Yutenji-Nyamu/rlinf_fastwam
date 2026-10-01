# 两个实验窗口的当前资源安排

更新时间：2026-10-01 13:18（北京时间）。最新人工安排：EXPO放深圳2物理4–7，原RLT先停，结束再回来；WM继续深圳3物理4–7。不同机正常推进不频繁互相轮询，仅资源变更、交接或故障协调。此前所有物理3安排作废，完整历史保留在[归档](archive/EXPERIMENT_RESOURCE_COORDINATION_BEFORE_20261001_1320.md)。

|窗口|服务器/物理卡|当前任务与控制|归还|
|---|---|---|---|
|0928 exp 2|深圳1 4–7|原四RLT继续，累计3000轮|原精确清单，真实空闲且预算未完才恢复|
|0927 exp|深圳2 4–7|EXPO已授权借卡；对方独立root `/data/chenyiteng/projects/expo-ft-sz2-20261001`，对方唯一owner管理|EXPO结束/失败后由同owner恢复原四RLT；统一检查不抢卡|
|0928 exp 2|深圳3 4–7|Wan r6，v6唯一owner，RUNNING_WM；13:18已完成首轮有效更新，下一轮采集|WM完整释放后直接原四RLT，不开Dojo|

深圳2的EXPO配置/实际启动以对方`docs/methods/expo-ft/`、owner/run和现场回执为准，本窗资源表只登记授权边界，不代报未读取的真实GPU进展。深圳3实际actor/env/rollout placement已核4–7，原SFT/单视角mask/GRPO/1000轮/save40保持；新cycle绑定原四RLT完整CP125。首轮有效更新grad0.5559873、mask1.376953%，尚未超过旧4轮故障点，无CP。

- 各窗负责自己的源码、环境、run、namespace、分支和唯一启停/归还；EXPO不接管WM，WM不操作EXPO进程或发布对方dirty。
- 本窗维护HANDOFF与本表；对方维护EXPO研究/运行文档与PROJECT_CONTEXT指针。
- 深圳3新v6原生直接RLT，不重建旧v5 guard，不重放w110/w111或旧stop/prepare/resume。独立核清进程和GPU释放后才归还。
- 所有物理0–3不新增RLT副本。共享Ray、其他用户、无关实验及驱动保持。
- 统一rlt检查只维护当前已授权安排，EXPO/WM借卡或交接期间不抢恢复；正常采集等待或瞬时0%不当空卡。方法/预算新安排不能仅由另一窗口消息扩大。

证据：本窗`w110–w115`唯一准备/启动/placement与`w123-repair-first-update`；13:00人工改EXPO至4–7的授权与跨窗同步。之前三机RLT和旧WM/Dojo交接均见归档。
