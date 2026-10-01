**13:00最新人工安排：exp的EXPO改为深圳2物理4–7，原四RLT低优先级允许先停，EXPO结束后按原配置/最新完整CP恢复；由exp窗口唯一owner负责。物理3筹备方案取消。exp2 WM保持深圳3物理4–7。不同机正常推进不频繁互相轮询，仅资源变更/交接/故障协调。统一检查不能在EXPO借卡期间抢SZ2恢复RLT。**

**12:57实际更新：SZ3新v6已唯一启动r6，RUNNING_WM、真实placement4–7通过，模型加载中；原四RLT已停止，WM结束/失败新owner直接归还。SZ2 EXPO物理3仍由另一窗口筹备，双方窗口实查无冲突。下方12:48表中SZ3准备阶段已被本条覆盖。**

# 12:48更新：WM监控修复与EXPO筹备

用户本轮重新授权修复深圳3 WM并使其稳定训练。目前CPU检查/代码审查，无新GPU进程；准备齐后只借SZ3物理4–7，新唯一v6 owner，WM结束/失败直接原RLT，不开Dojo、不重放v5。专题docs/world-model/WAN_GOAL_REPAIR_20261001.md。

|窗口|服务器/卡|当前安排|控制与归还|
|---|---|---|---|
|0928 exp 2|SZ1 4–7、SZ2 4–7|原8条RLT继续|原精确恢复清单，累计3000轮|
|0928 exp 2|SZ3 4–7|原4条RLT继续；WM修复CPU准备，之后新run r6|v6待准备/启动；WM→原4RLT直接，不自动Dojo|
|0927 exp|SZ2物理3，UUID GPU-07910b32-fd6c-74f1-8fee-5e1b24e6449e|EXPO reserved准备，尚未启动GPU|对方root /data/chenyiteng/projects/expo-ft-sz2-20261001；计划runs/smoke-v1、tools/smoke_owner.py；对方唯一owner|

EXPO若需物理2或4–7，先同步具体资源和精确借卡归还；本窗不停止其进程、不部署其source、不发布其dirty。0–3不自行新增RLT副本。下面为此前历史快照。
# 两个实验窗口的深圳资源安排

更新时间：2026-10-01 11:14（北京时间）。用户明确要求两个 exp 窗口协调并分别推进。现场证据优先，空闲快照不等于已分配资源。

**11:13实查：SZ3四原RLT已从CP125推进至127/127/126/127轮，all_first_rounds_verified=true，四driver活；当前三机4–7均维护原RLT。后续正常推进静默，WM/Dojo不自动重开。**

**当前变更：SZ3 Wan r5完成4轮后监控异常中断，WM完整释放；原v5 owner10:59:31唯一恢复原四RLT，11:03四driver活、CP125、首轮待验。三机4–7目前均维护原RLT；WM/Dojo不自动重开，恢复/监控责任仍由0928 exp 2承担。下面10:04表为历史快照。**

|窗口|本轮任务|当前GPU需求|写入范围|
|---|---|---|---|
|0928 exp 2|WM故障记录；三机原RLT检查与归还|三机原RLT物理4–7；不新启WM/Dojo|Wan独立源码/环境/run、docs/world-model、HANDOFF与本表；Git codex/sz3-wan-goal-20260930|
|0927 exp|EXPO-FT论文、官方开源与RLinf/RoboTwin/π0.5接入调研|本轮0张，暂无拟启动任务|docs/methods/expo-ft、PROJECT_CONTEXT研究入口；本轮未提交/推Git|

另一窗口已明确回复：SZ1只读核源码/实配完成，不改运行；本轮不启动GPU任务，不接管当前归还控制器；HANDOFF与本表由0928 exp 2维护。

## 10:04三机现场

|服务器|物理4–7|最新真实进展|物理0–3|
|---|---|---|---|
|深圳1|原四RLT：手机支架clean/combo、双瓶clean/combo|655/649/661/657轮，四driver活，无新fatal|未分配；显存仅4–69MiB，启动前须重新核进程归属|
|深圳2|原四RLT：多种瓶子clean/combo、电子秤clean/combo|1418/1444/1383/1385轮，四resume活，首轮已验|未分配；显存仅4–194MiB，启动前须重新核进程归属|
|深圳3|Wan Goal π0.5 r5正式，private Ray 63845|第0轮完成，grad norm0.79649、有效mask1.2598%、有限非零优势；已进入第1轮采集|未分配；各4MiB，启动前须重新核进程归属|

深圳3正式目录：`/data/chenyiteng/projects/wan-goal-sz3/runs/wan-goal-sz3-20261001-r5`。仅该任务的actor/env/rollout物理placement4–7已经核验。WM内部奖励结果不等同于真实LIBERO评测。

## 启动与归还的协调

- 当前4–7的唯一启停与恢复责任归0928 exp 2及已部署服务器owner；另一窗口不重放旧Dojo、RLT或WM脚本。
- 用户最新路线：深圳3WM结束或失败后直接恢复原四RLT，不续Dojo。`post-wm-direct-rlt-v5/request.json`与绑定的CPU guard为当前请求；实际RLT resume仍由原v5 owner唯一执行。
- 另一窗口若以后获授权部署新实验，先刷新现场，再同步服务器、物理卡、任务、run、控制器与归还安排；本表登记后启动。优先考虑目前未分配的0–3，不自行扩增原RLT副本。
- 每个任务独立源码/输出、Ray namespace/端口和Git分支；只发布本窗口审过的文件，不发布对方dirty。共享Ray、其他用户、驱动和无关实验保持。
- 心跳按本表与最新人工安排工作；不把瞬时0%利用率或已登记任务的启动等待当空卡。新资源安排覆盖旧恢复安排时，更新本表和心跳后再切换。

证据：`local_logs/wan-goal-20261001/steps/w059-formal-learning-and-guard/`、`w060-sz1-coordination-current/`、`w061-sz2-coordination-current/`；另窗10:05协调回信。以上均只读刷新，未增加GPU任务。
