# 两个实验窗口的当前资源安排

更新时间：2026-10-01 14:32（北京时间）。最新人工安排：EXPO放深圳2物理4–7，原RLT先停，结束再回来；WM继续深圳3物理4–7。不同机正常推进不频繁互相轮询，仅资源变更、交接或故障协调。此前所有物理3安排作废，完整历史保留在[归档](archive/EXPERIMENT_RESOURCE_COORDINATION_BEFORE_20261001_1320.md)。

|窗口|服务器/物理卡|当前任务与控制|归还|
|---|---|---|---|
|0928 exp 2|深圳1 4–7|原四RLT继续，累计3000轮|原精确清单，真实空闲且预算未完才恢复|
|0927 exp|深圳2 4–7|对方已回执四原RLT精确退出、guardian=PAUSED；EXPO在独立root `/data/chenyiteng/projects/expo-ft-sz2-20261001/runs/smoke-v1`派发，先用物理4|唯一guardian负责结束/失败/租约退出后恢复原四RLT并验证首轮；其余借用卡仍由对方管理，统一检查不抢恢复|
|0928 exp 2|深圳3 4–7|Wan r6，v6唯一owner；14:32完成5轮、3轮有效GRPO梯度，已超过旧4轮故障位置，第6轮采集|WM完整释放后直接原四RLT，不开Dojo|

深圳2的EXPO配置/实际启动以对方`docs/methods/expo-ft/`、owner/run和现场回执为准，本窗资源表只登记授权边界，不代报未读取的真实GPU进展。深圳3实际actor/env/rollout placement已核4–7，原SFT/单视角mask/GRPO/1000轮/save40保持；新cycle绑定原四RLT完整CP125。已超过旧故障位置，尚无CP40；w129四份CP125严格replay/DCP检查通过。

14:00接收EXPO窗口的借卡完成回执：guardian PID3953486、boot87193fef-9734-40ce-ada3-0d084ab9d254，最终冻结CP gpu4/5/6/7=1500/1525/1475/1475并独立修复验证；物理4 UUID GPU-a0a252d6-828d-29e1-1fd2-65187f573f4d。本窗不重复检查/操作对方guardian，不依据这些历史身份发信号，等待对方唯一归还回执再调整路由。以上来自对方现场回执，未由本窗重复独立验证其GPU执行。

- 各窗负责自己的源码、环境、run、namespace、分支和唯一启停/归还；EXPO不接管WM，WM不操作EXPO进程或发布对方dirty。
- 本窗维护HANDOFF与本表；对方维护EXPO研究/运行文档与PROJECT_CONTEXT指针。
- 深圳3新v6原生直接RLT，不重建旧v5 guard，不重放w110/w111或旧stop/prepare/resume。独立核清进程和GPU释放后才归还。
- 所有物理0–3不新增RLT副本。共享Ray、其他用户、无关实验及驱动保持。
- 统一rlt检查只维护当前已授权安排，EXPO/WM借卡或交接期间不抢恢复；正常采集等待或瞬时0%不当空卡。方法/预算新安排不能仅由另一窗口消息扩大。

证据：本窗`w110–w115`唯一准备/启动/placement与`w123-repair-first-update`；13:00人工改EXPO至4–7的授权与跨窗同步。之前三机RLT和旧WM/Dojo交接均见归档。
