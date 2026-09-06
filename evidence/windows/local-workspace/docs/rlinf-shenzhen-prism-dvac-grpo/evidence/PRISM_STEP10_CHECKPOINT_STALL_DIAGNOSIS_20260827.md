# Prism-DVAC Step 10 checkpoint 卡住：只读诊断

现场快照：2026-08-27 23:34 CST。服务器只读检查；未停止、重启或修改任何任务。

## 直接结论

Prism 已完成 Step 10 的 4/4 rollout 和 fixed-32 eval（25/32），于 20:00:54 CST 进入 checkpoint。约 3.5 小时后：

- driver 与两个 actor 仍存活，但训练没有继续；
- `global_step_10` 只有 12 KiB 空目录，没有 `.distcp`、`.metadata` 或 full weights；
- GPU 6/7 保留约 66.5/67.5 GiB，利用率为 0；
- 两个 actor 都处于事件等待，最后日志边界在 PyTorch FSDP optimizer-state 检查路径 `_optim_utils.py:1173`；
- 保存开始时主机仍有约 508 GiB available RAM，磁盘约 2.1 TiB 可用；无 OOM、NVRM/Xid、I/O error、Ray worker death 或 traceback。

最符合的解释是：Prism 自己的两个 FSDP rank 在 optimizer-state 提取所需的 collective 中没有完成会合，卡在 DCP 真正写 shard 之前。最后一条 warning 只定位边界，不足以声称精确停在某一条 Python 语句。

## 为什么不是普通磁盘慢，也不像通用的多 job 故障

1. RLinf 的顺序是 `barrier -> get_state_dict(model, optimizer) -> dcp.save writer`；当前没有任何 shard，说明 writer 尚未开始。
2. Control 与旧 DVAC `[0,2]` 曾在同一 shared Ray 上几乎同时保存 Step 10，双方都产生完整的两份 shard、metadata 和 full weights。
3. 当前 Prism 卡住后，Control 仍在同一 shared Ray 上成功保存 Step 70，并继续到完整 Step 71。共享 Ray、文件系统和全局 DCP 能力没有整体卡死。
4. Prism 两 rank 为独立 Ray job、rank 0/1、world size 2；未发现 Control actor、输出路径或 checkpoint 目录混入 Prism。
5. Prism 方法提交没有修改 runner、optimizer builder、FSDP strategy 或 DCP；actor 的 save/load 也与 Control 相同。Prism 没有新增 optimizer 参数或持久状态。

所以“并发 RLinf”最多是未证实的间接时序因素，不能作为根因；没有证据表明两个 torch process group 被 Ray namespace 合并。

## 旧案例和 smoke 覆盖

- 旧 RLT Stage 2 v3 的首次 Step 25 full checkpoint 有高度相似的现象：eval 后、无 shard、GPU 0%、最后位于同一 FSDP optimizer-state 路径。
- 但旧 RLT 是 actor+critic 两个 optimizer，且当时确认 plural builder 漏了 optimizer warm-up；Prism 是已经完成真实更新的单 optimizer，singular builder 本来就执行 warm-up。旧两行修复不能搬到 Prism。
- RLT 后续 A/B 甚至不能稳定复现原 hang，说明旧事故也包含未唯一定位的 PyTorch collective 时序。
- Prism 自己的两卡 smoke 完成两次 optimizer call，并完整保存 `global_step_1` 后 exit 0；但它关闭 inline eval，未覆盖 formal 的 `eval -> periodic checkpoint`。
- Action-Adv 两步 smoke 在 Control 与 Prism formal 同时运行期间，也完整保存 DCP；再次说明 shared Ray 本身可并发保存。

## 当前判断与下一步

当前概率排序：

1. Prism rank 间的 FSDP optimizer-state collective 偶发失配/挂起；
2. PyTorch 2.11 FSDP/DCP 已知类别的 state-dict collective hang；
3. shared Ray 带来的间接资源或时序干扰；
4. Prism 数学、磁盘满或普通慢写盘——现有证据不支持。

当前 `global_step_10` 不可恢复。若用户授权处理，推荐只停止 Prism 精确 job，保持 Control 不动；给 generic checkpoint 增加少量 rank entry/exit 与 optimizer-state 计数日志，再以完全相同科学配置 fresh 重放。不要移植 RLT warm-up patch，也不要改 Prism 算法或训练预算。

## 依据

- [Prism 实现与 smoke 流水账](IMPLEMENTATION_AND_SMOKE_LEDGER_20260826.md#7-真实两卡-smoke)
- [旧 RLT checkpoint 根因分层](../../rlinf-shenzhen-rlt-dsrl-port/10_RLINF_MULTI_JOB_RUNTIME_AND_RLT_CHECKPOINT_DIAGNOSIS_20260824.md#31-根因分层已确认代码缺口与未唯一确认的内部时序)
- [RLinf runner 同步等待保存](https://github.com/RLinf/RLinf/blob/7d07a4212ee6858cc333e1d4fab7a37256d1f839/rlinf/runners/embodied_runner.py#L644-L653)
- [RLinf FSDP barrier 与 DCP save](https://github.com/RLinf/RLinf/blob/7d07a4212ee6858cc333e1d4fab7a37256d1f839/rlinf/hybrid_engines/fsdp/strategy/base.py#L184-L248)
- [RLinf Stateful/get_state_dict](https://github.com/RLinf/RLinf/blob/7d07a4212ee6858cc333e1d4fab7a37256d1f839/rlinf/hybrid_engines/fsdp/strategy/checkpoint.py#L66-L102)
- [PyTorch FSDP state-dict hang #143536](https://github.com/pytorch/pytorch/issues/143536)
- [Ray multi-job FAQ](https://docs.ray.io/en/master/cluster/faq.html)；[Ray namespace 语义](https://docs.ray.io/en/latest/ray-core/namespaces.html)
