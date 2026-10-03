# OpenDW smoke：深圳 3 单卡借还接线

2026-10-03。本页是本地源码审查与新组件说明；服务器现场、实际执行和首轮验收以本轮 owner 的回执为准。

## 最小范围

仅借 GPU4 当前 `beat_block_hammer / clean` RLT。当前入口来自 `/data/chenyiteng/deployment-20261002/rlt-next6-beat_block_hammer/plan.json`，源码固定到 `2f484040dbf078fcb6bfcc1da8a300fbc625f87b`。GPU5–7 属于 `rlt-step-timeout-recovery-v1` 的三个恢复任务，冻结其 owner/driver 身份用于保护和事后记录，不向它们发信号。

原来的四卡 `rlt_cycle_sz3.py` 不可直接运行：prepare 指向 9 月 29 日旧任务，stop/release 检查硬编码四卡。本次 [单卡组件](../../../tools/opendw_smoke_gpu4_cycle.py) 复用其中完整 checkpoint 检验、精确恢复和首轮验收；停止环节调用当前 next-six `ops.stop_old([one_target], 'clean-')`。其 GPU 进程检查补为 NVIDIA XML 的计算和图形上下文两类。

## 借还过程

1. `prepare` 在独立 cycle 目录冻结原 driver 的 PID/UID/启动时刻/命令摘要、Ray namespace/job、实配、依赖、源码、启动环境与最新完整 checkpoint。没有完整 checkpoint 就不借；不覆盖原 runtime。
2. `stop` 在发信号前再次确认身份和 checkpoint，仅停止原 driver 及该 namespace/job 的 actor/子进程。释放后重新选择最新完整 checkpoint，记录未保存轮数。
3. 外层 smoke owner 按自己的配方运行 OpenDW。它必须持有独占 owner 锁并在成功、失败或超时的 `finally` 路径，精确回收自己创建的进程，写 release receipt，然后调用 `resume`。单卡组件本身不启动 smoke，也不负责后台超时唤醒。
4. `resume` 重用原 Python、原源码、原 N8/200 动作/3000 轮预算，仅改变 run/log 路径、namespace 和 checkpoint 恢复入口。新建 runtime，并持锁更新监控路由中的 GPU4 一行。GPU5–7 的配置和路由保留。
5. `status` 需要确认 checkpoint 已加载、replay/更新计数未回退、出现 checkpoint 之后的新一轮且 driver 存活，才写 `first-round-verified.json`。`resumed-dispatched.json` 仅表示已派发。

stop 不支持“立即保存到当前轮”的信号；保留的是最新已有完整 checkpoint，不能承诺一轮不丢。恢复沿用原任务的 200 动作预算，本次没有把历史 RLT 改成新讨论的 400。

## 外层 owner 调用合同

以下 `$PY` 是当前 RLT plan 中的 Python；`$SRC` 是已审查部署源码；`$CYCLE` 是本次唯一新目录；helper 必须是现有审过的 `rlt_cycle_sz3.py`。PID 参数必须来自本轮现场核验，不能长期沿用示例。

```bash
"$PY" -B "$SRC" --cycle-dir "$CYCLE" prepare \
  --helper "$CHECKPOINT_HELPER" --expected-driver-pid "$VERIFIED_GPU4_PID"
"$PY" -B "$CYCLE/opendw_smoke_gpu4_cycle.py" --cycle-dir "$CYCLE" stop
# 外层 owner 的 smoke + 精确清理；finally 中继续下面恢复。
"$PY" -B "$CYCLE/opendw_smoke_gpu4_cycle.py" --cycle-dir "$CYCLE" resume \
  --release-receipt "$CYCLE/smoke-release.json"
"$PY" -B "$CYCLE/opendw_smoke_gpu4_cycle.py" --cycle-dir "$CYCLE" status
```

release receipt 至少包含：

```json
{
  "cycle_id": "本次唯一目录名",
  "gpus": [4],
  "terminal_status": "completed",
  "all_workers_stopped": true,
  "managed_processes": [{"pid": 123, "uid": 20001, "start": 123456}]
}
```

`terminal_status` 允许 `completed/failed/timed_out/not_started`；只有 `not_started` 允许空 managed list。每条身份需在启动时登记，包括派生的模型、奖励、训练等 worker；这里只是结构示例。恢复时重新核查所有登记进程已经消失、GPU4 的 C/G 上下文均为空，不能仅信任 JSON 中的布尔值。

若停止或派发留下半途回执，不重放 prepare/stop/launch。先根据 `clean-old-stop-attempt.json`、`clean-old-stopped.json`、`gpu4-launch-attempt.json` 和实际进程处理相同 cycle。特别是 `stop_old` 已完成而后续 bookkeeping 失败时，需要补全这次 checkpoint/stop receipt，不能开始新 cycle。

## 本地审查边界

新组件已完成对 helper 签名、单 key stop、配置白名单及 watch/allowlist 共享锁的源码核对。尚未以此新组件执行服务器借还；实际使用前在深圳 3 做 CPU 语法/准备验证，再按本轮现场 owner 执行。它记录所有卡的上下文，但只把 GPU4 空闲作为借还验收；原 RLT 恢复时的历史图形绑定也按原配置继承，不能据此宣称 GPU0–3 完全无残余。

已补外层 [owner 与 plan 字段](../../../local_patches/opendw_smoke_20261003/tools/OWNER_PLAN.md)，包含 CPU-ready 后借卡、N8→N16 各 45 分钟上限、失败停止下一档、finally 归还和首轮验收。正式预算使用独立 `mode: formal` plan，在读数之后显式确定，不默认停在 smoke 交付。

顺次再借同一卡时，prepare 增加 `--previous-cycle <已归还cycle>`：读取上一 cycle 的 `new_run/namespace`，核当前 driver argv、watch 路由及恢复首轮，从该 run 实配与继承 checkpoint 接续。`finalize_stopped()` 只为已有精确停止完成回执补充 checkpoint/配置记录，不重新发停止信号。
