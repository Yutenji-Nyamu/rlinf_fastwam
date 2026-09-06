# RLT checkpoint 与 RLinf 双 job 调查流水账

日期：2026-08-24  
范围：只读服务器刷新、源码/官方资料核对、本地文档整理；本轮未停止或重启任何训练。

## 1. 操作与结果

| 序号 | 操作 | 结果 |
|---:|---|---|
| 01 | 读取专题 SSOT、并发结论、既有 live 指标和 operation ledger | 锁定 RLT v3、DSRL v2、shared Ray 和授权边界；不复述历史状态为当前 |
| 02 | 10:49--10:52 CST 通过既有只读现场入口刷新两个 run | DSRL 完整 154/200、Step155训练中；RLT仍卡 Step25 save；host/GPU/Ray 健康 |
| 03 | 核 RLT partial checkpoint tree 与 actor/driver 末尾 | 仅 99-byte incomplete manifest；无 DCP shard/metadata；两 actor 均停在 distributed optimizer-state 路径 |
| 04 | 用 DSRL 的并发进展和 checkpoint 作反事实核对 | RLT hang 后 DSRL 继续训练并成功保存 Step65/130；排除共享 Ray/GCS、磁盘、主存或通用 DCP 全局失效作为主因 |
| 05 | 对照 RLT smoke/formal 的 optimizer 状态 | smoke 保存前已有8次真实update；formal首次保存为update0。RLinf已有LR=0 synthetic step，故最强区别变量是synthetic-only/partial state，而非简单空optimizer |
| 06 | 核 PyTorch DCP/FSDP 官方源码和 issue | DCP在写shard前提取Stateful/optimizer state；fresh/partial optimizer与collective hang均有官方已知边界 |
| 07 | 核并发启动器、current RLinf cluster源码与resolved config | 并发只改运行层；算法0 LOC。稳态新增约82行控制，事故探针/恢复约412行，不进入算法热路径 |
| 08 | 形成最小A/B方案 | shared Ray下比较0-update save与1-update save；只有二者不能解释时才做isolated-Ray对照 |

## 2. 当前停止边界

- 没有停止 DSRL：它尚未完整到200，资源正常，最新可恢复点是Step130；建议自然结束。
- 没有停止 RLT：虽然该run已不可恢复且不再推进，但精确停止owned PGID及namespace属于进程控制，等待用户明确
  批准最小复现方案。
- 没有执行全局 `ray stop`、没有修改PyTorch、没有删除partial checkpoint或旧失败目录。

## 3. 下一条有状态操作

若用户批准：

1. 保存当前RLT日志和partial tree清单；
2. 只停止RLT v3 owned PGID并清它的exact namespace；
3. 保持DSRL和shared Ray不动，运行一次0-update save复现；
4. 运行一次至少1次真实update后的save对照；
5. 依据A/B结果决定“延后首次full checkpoint”或进一步查process group；不预先patch barrier。
