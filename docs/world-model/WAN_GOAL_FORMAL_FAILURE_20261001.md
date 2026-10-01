# SZ3 Wan Goal 正式中断与原RLT归还

更新时间：2026-10-01 11:05。本次r5正式已完成4/1000轮，step0–3均有正梯度、有效loss mask和有限loss，第5轮采集到2/8后于10:58:04中断。首个正式checkpoint原定第40轮，因此没有正式checkpoint；原smoke学习及完整CP证据保持。

## 发生了什么

- `pi05-formal/wm-exit.json`：`outcome=failed`、`exit_code=null`、`error=AssertionError()`。这是内层运行监控捕获的异常，不是已取得的训练driver退出码。
- `wm_stage.py`先令exit_code=None，监控循环正常结束后才赋子退出码；异常处理只保存repr，未保存原断言堆栈/PID/实际值。训练日志中10:58:04的SIGTERM在该失败回执之后，是owned cleanup的结果。近期未见模型计算Traceback，不据此证明模型无任何潜在问题。
- 已比对实际部署与本地已审源码SHA：common.py=`a99b2565…`、wm_stage.py=`2bb73294…`，一致。可达候选是common.py:129的`/proc/<pid>`目录st_uid断言，以及Catalog.scan/alive的调用路径；尚无证据确定具体PID或变化原因，不能声称已证明PID复用或其他用户干扰。
- 先前environ不可读处理不能覆盖所有身份断言；当前缺失的信息是失败瞬间完整traceback及PID/UID/start/boot快照。后续若重新实施，先补精确诊断与CPU身份竞争检查，再决定最小修复；不能放宽归属核查或改GRPO/seed/预算。本次没有重启WM或修改监控源码。

## 归还实际结果

`w078`核内层和外层WM释放回执均为all_workers_stopped/processes_clear/gpus_released=true，物理4–7；原v5 owner唯一归还流程完整。CPU guard等到完整释放后于10:58:53中断短暂Dojo入口，未继续Dojo评测，随后由同一owner于10:59:31唯一派发原四RLT。

- cycle：`/data/chenyiteng/projects/robodojo-openwam-sz3/rlt-cycle-sz3-wan-goal-20261001-wake-v3`。
- owner final：terminal_status=failed（实验终态），error=null（归还无异常），wm_released=true、rlt_dispatched=true。
- guard最终为RLT_RETURN_DISPATCHED并正常结束；owner也已结束，不要求它们在归还后继续存活。
- `w077`11:03：四条原RLT driver均活，从完整CP125恢复，首轮指标尚未写出；这表示派发和加载存活，不等同于首轮已完成。后续只读巡检继续确认实际推进，不重复resume。
- SZ1/2原四RLT仍活并推进；物理0–3未新增安排。当前三机4–7均归原RLT维护，WM/Dojo不自动重开。

## 证据位置

本地细日志：`local_logs/wan-goal-20261001/steps/w073-heartbeat-sz3-1058/`（发现失败）、`w076-wm-failure-live-1059/`（训练与SIGTERM时序）、`w077-sz3-rlt-after-wm-failure/`（原RLT存活）、`w078-formal-failure-receipts/`（源码SHA、释放与唯一派发）。粗日志见WAN_GOAL_RUNLOG.md。原失败、终态和释放回执不覆盖。

核心回执SHA256：外层pipeline-final=`56575342e65399544df0d49b4bebac7aaf1ebfa505113c74c426c37a1c400575`；外层WM释放=`d66d02a1f0d1b4d94f7fdafe886857a9c1498ed2b4847a5bcbaa5a0cefe74b48`；RLT唯一派发=`304477a5ef40fc158c99686315821e1e4d02d7d743c0260c2c79fc85680796db`。
