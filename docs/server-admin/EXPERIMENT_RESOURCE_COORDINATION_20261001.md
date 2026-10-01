# 两个实验窗口的当前资源安排

更新时间：2026-10-01 14:39（北京时间）。EXPO在深圳2物理4–7的借卡已完成归还，原四RLT继续；WM继续深圳3物理4–7。不同机正常推进不频繁互相轮询，仅资源变更、交接或故障协调。此前所有物理3安排作废，完整历史保留在[归档](archive/EXPERIMENT_RESOURCE_COORDINATION_BEFORE_20261001_1320.md)。

|窗口|服务器/物理卡|当前任务与控制|归还|
|---|---|---|---|
|0928 exp 2|深圳1 4–7|原四RLT继续，累计3000轮|原精确清单，真实空闲且预算未完才恢复|
|0927 exp完成借卡；0928 exp 2统一检查|深圳2 4–7|EXPO归还完成；w141本窗核原四RLT已到1505/1530/1480/1480轮|原guardian唯一恢复，首轮四条已验；统一维护接回，禁止重放旧启停|
|0928 exp 2|深圳3 4–7|Wan r6，v6唯一owner；14:32完成5轮、3轮有效GRPO梯度，已超过旧4轮故障位置，第6轮采集|WM完整释放后直接原四RLT，不开Dojo|

深圳2的EXPO配置/实际启动以对方`docs/methods/expo-ft/`、owner/run和现场回执为准，本窗资源表只登记授权边界，不代报未读取的真实GPU进展。深圳3实际actor/env/rollout placement已核4–7，原SFT/单视角mask/GRPO/1000轮/save40保持；新cycle绑定原四RLT完整CP125。已超过旧故障位置，尚无CP40；w129四份CP125严格replay/DCP检查通过。

14:31对方回报归还完成；w141 14:39本窗固定host-key只读独核guardian=RESTORED、terminal_status=completed、error=null，rlt-first-round四条已验。冻结CP1500/1525/1475/1475，实际更新已到1505/1530/1480/1480、四resume_alive=true、old_alive=false、finished=null、ready_online=1、真实critic更新，watch四条匹配new_run，CUDA未初始化。当前cycle `/data/chenyiteng/projects/expo-ft-sz2-20261001/rlt-cycle`、namespace er-rlt-cycle-g4/g5/g6/g7；`resumed-dispatched.json`及watch为当前路由。guardian正常结束，不要求持续活着；其rlt_paused未复位的旧字段不当作仍借卡。现已取消SZ2跳过，统一检查接回原RLT维护，不重放stop/resume/guardian。EXPO方法与Git结果仍由对方专题记录，本窗不发布其dirty。

- 各窗负责自己的源码、环境、run、namespace、分支和唯一启停/归还；EXPO不接管WM，WM不操作EXPO进程或发布对方dirty。
- 本窗维护HANDOFF与本表；对方维护EXPO研究/运行文档与PROJECT_CONTEXT指针。
- 深圳3新v6原生直接RLT，不重建旧v5 guard，不重放w110/w111或旧stop/prepare/resume。独立核清进程和GPU释放后才归还。
- 所有物理0–3不新增RLT副本。共享Ray、其他用户、无关实验及驱动保持。
- 统一rlt检查只维护当前已授权安排，EXPO/WM借卡或交接期间不抢恢复；正常采集等待或瞬时0%不当空卡。方法/预算新安排不能仅由另一窗口消息扩大。

证据：本窗`w110–w115`唯一准备/启动/placement与`w123-repair-first-update`；13:00人工改EXPO至4–7的授权与跨窗同步。之前三机RLT和旧WM/Dojo交接均见归档。
