# 深圳3 WM监控修复后r6运行 · 2026-10-01 12:57

用户确认exp2继续WM，另一exp负责EXPO；当前无冲突。w110 prepare通过，w111 12:53:23唯一启动v6。w114 12:55 owner活、RUNNING_WM，真实actor/env/rollout placement文件已通过4–7，正式模型正在加载；尚无训练标量。新run R/runs/wan-goal-sz3-20261001-r6、cycle P/rlt-cycle-sz3-wan-goal-20261001-repair-v1。原SFT/head-only/GRPO/seed/N64/G8/R8/L320/C8/global2048/micro128/H10/去噪5/1000epoch/save40不变，不重跑smoke。原RLT已精确停止，完整CP125绑定；新owner原生WM→原四RLT直接，不开Dojo、不另起guard/restorer。不得重放旧v5或本轮w110/w111。

w108 CPU检查18项通过，w106新2000短命进程无异常。修复proc FD＋status UID/start核验、旧身份pidfd信号、traceback/最终scan/cleanup错误分记及已知catalog预存；r5原assert根因仍未确定，未虚称实锤。固定d34d4c3已含reset PR1518。当前已发布最新fedc3595，审过修复代码/轻量证据发布中。专题docs/world-model/WAN_GOAL_REPAIR_20261001.md、WAN_GOAL_REPAIR_SOURCES_20261001.md、WAN_GOAL_RUNLOG.md。

下一步用status_wake.sh刷新启动/退出，用repair_runtime_evidence.sh核实原配置和有效有限更新；需超过原4轮故障点并等待原save40完整CP，不把启动/CPU检查当长期稳定完成。仅真实错误继续最小修复，不改方法/seed/预算。统一rlt每15min沿当前active pointer/唯一owner维护WM及SZ1/2原RLT，正常安静。SSH73160仍活，固定pin chenyiteng，密码仅RAM。

另一窗口0927 exp深圳2物理3保留EXPO准备（独立root /data/chenyiteng/projects/expo-ft-sz2-20261001/source与runs/smoke-v1、唯一tools/smoke_owner.py），4–7原RLT继续。本窗维护HANDOFF/共享资源表，不操作对方source/进程，不发布其dirty。0–3不新增RLT副本；任何新增资源按docs/server-admin/EXPERIMENT_RESOURCE_COORDINATION_20261001.md协调。旧完整交接archive/HANDOFF_WAN_REPAIR_BEFORE_20261001_1247.md保留。
