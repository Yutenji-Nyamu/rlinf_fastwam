# RLinf 双训练运行层与 RLT 首次 checkpoint 卡住：短结论

日期：2026-08-24  
机器：SZ-H100  
范围：RLT GPU 4--5、DSRL GPU 6--7；动态状态以服务器现场为准。

## 1. 当前应该怎么判断

1. **双训练本身已经跑通。** 两个独立 RLinf job 长期同时推进；RLT 卡住后，DSRL 仍在同一 persistent
   Ray 上推进并成功保存 Step65/130。A0、B、A′ 三个RLT诊断run也都在DSRL并发时完成保存。
2. **RLT v3 不是学习失败。** 它完整到Step24；Step25 fixed-20完成后卡在首次full checkpoint。此时
   `update_step=0`，在线actor/critic尚未学习。该不可恢复run已按owned PGID + exact namespace精确停止。
3. **已确认的代码缺口是plural optimizer builder漏掉RLinf既有warmup。** RLT构造actor/critic两个Adam后
   直接返回；singular builder和checkpoint自身的注释都要求构造期先初始化optimizer state。A0实测保存前确为
   全空；DCP lazy-init后变成`step=1`。两行修复让两个optimizer在保存前即完整且保持`step=0`。
4. **formal终验已通过。** v4在同样`update_step=0`、fixed-20=`0/20`的条件下写出完整Step25 checkpoint，
   随后完整进入Step26。A0这一次也能成功保存，所以仍不能说“空optimizer必然卡死”；准确说法是两行修复
   消除了已证实的合同违例和Adam-step偏移，并在正式规模绕开该lazy-init分支，但v3那一次间歇stall的更细
   PyTorch内部时序没有被唯一复现。

## 2. RLT checkpoint：已经确定什么

确定的调用边界：

```text
runner Step 25
  -> RLTACFSDPPolicy.save_checkpoint
  -> current SAC/FSDP generic save
  -> torch.distributed.checkpoint.save
  -> Stateful/FSDP optimizer state_dict extraction
  -> _optim_utils.py:1173 附近停止推进
```

- 两个 actor rank 都进入保存路径；01:39:19 CST 后 driver、TensorBoard、checkpoint 均不再增长。
- `global_step_25` 只有 RLT 的 99-byte `complete=false` manifest；DCP 目录没有 `.metadata` 或 `.distcp`
  shard，因此该目录不可恢复。
- GPU 4--5 保持约 17 GiB/card、utilization 0；没有 OOM、磁盘写满、主机内存压力或 Ray/GCS 故障。
- current RLinf 保存链先做 `get_state_dict(model, [actor_optimizer, critic_optimizer])`，再进入 DCP writer。
  PyTorch DCP 会先调用 `Stateful.state_dict()` 并做分布式状态规划，之后才进入 storage writer；因此
  “没有第一个 shard”与“卡在 optimizer-state 收集阶段”是相容的。
- 最后的 `_optim_utils.py:1173` warning 在成功 smoke 中也出现过；它只把范围收窄到 optimizer-state
  一致性检查附近，**warning 本身不是错误，也不能证明正卡在下一条 collective**。

PyTorch 官方 DCP 源码说明了这条 `state_dict -> collectives -> storage` 路径：
[state_dict.py](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/distributed/checkpoint/state_dict.py)、
[state_dict_saver.py](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/distributed/checkpoint/state_dict_saver.py)。

## 3. 根因分层：什么已证实，什么仍不能过度声称

| 层 | 结论 | 证据 |
|---|---|---|
| 直接卡点 | v3两个rank都停在DCP的FSDP optimizer-state提取、writer之前 | 无`.distcp/.metadata`；GPU0 util；两rank现场任务均为`save_checkpoint` |
| RLinf代码缺口 | plural `build_optimizers()`漏掉singular路径已有的`warmup_optimizer_state()` | 实际调用链与checkpoint docstring；A0四组现场计数均为empty |
| 语义后果 | 原路径在首次save内把fresh Adam从step0推进到step1 | A0 `before: empty`、`after: step=1`；A′构造期warmup后前后均为step0 |
| 修复 | 返回两个optimizer前逐一调用既有helper，共2 LOC | A′完整保存；聚焦回归确认zero moments、权重不变、第一次真实update为step1 |
| 仍未唯一解释 | 为什么v3那一次卡、A0这一次成功 | 说明empty/lazy-init不是确定性必挂；可能仍有PyTorch DCP内部偶发时序，现有死进程已无法取得更细栈 |

PyTorch 2.11本身允许iterable optimizer，并会对empty state逐optimizer调用内部`_init_optim_state()`；因此
“两个optimizer不受支持”与“empty state一定非法”都不成立。相关官方代码与已知边界见
[state_dict.py](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/distributed/checkpoint/state_dict.py)、
[PyTorch #164929](https://github.com/pytorch/pytorch/issues/164929)（`get_optimizer_state_dict`会改变fresh Adam行为）、
[PyTorch #192202](https://github.com/pytorch/pytorch/issues/192202)及
[#192307](https://github.com/pytorch/pytorch/pull/192307)。本机RLT实际PG是独立`world_size=2 / rank=0,1`，
因此把物理GPU4--5当作全局rank4--5并套用subgroup问题[#155680](https://github.com/pytorch/pytorch/issues/155680)
不符合现场前提。

两行修复本身直接复用current RLinf已有设计：singular builder、plural builder及其差异见
[RLinf FSDP model manager](https://github.com/RLinf/RLinf/blob/main/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py)，
零LR假step并把Adam step复位0的helper见
[RLinf `warmup_optimizer_state`](https://github.com/RLinf/RLinf/blob/main/rlinf/utils/utils.py)。

## 4. A/B结果与正式恢复

| 运行 | 真实update | 保存前optimizer | 保存结果 |
|---|---:|---|---|
| A0，原代码 | 0 | 两rank×两optimizer全空 | exit0；DCP内部变step1；完整checkpoint |
| B，原代码 | 1组训练 | 两rank×两optimizer完整、step1 | exit0；完整checkpoint |
| A′，两行修复 | 0 | 两rank×两optimizer完整、step0 | exit0；前后step0；完整checkpoint |

因此没有延后checkpoint、跳过optimizer、加barrier或patch PyTorch。正式修复只让plural builder满足RLinf保存
本来就写明的前置合同，同时保持Step25 eval/save及全部科学参数。commit：
`8bbd0216113ec6eca8bbc67e79d9d642d48b8ed1`。formal v4已从fresh Stage2启动；v3的Step25目录仍保留为
事故证据，不用于resume。

formal v4于12:25:52 CST进入同一首次保存边界并完成：

- fixed-20=`0/20`、`update_step=0`，不是通过提前训练optimizer规避问题；
- DCP `.metadata`与两rank shard完整；`full_weights.pt`、两份target-model rank文件与replay均存在；
- RLT sidecar为`complete=true / saved_runner_step=25 / update_step=0`；
- driver随后完整打印Step26，fatal/error/exit marker为0，训练继续。

## 5. DSRL：现在停还是继续

12:31 CST 最终刷新：

| 项目 | 当前值 |
|---|---:|
| 完整进度 | 180/200；继续运行 |
| 最近训练success | 75%--100%波动；末值均finite |
| 最新 fixed-12 | Step169：10/12=`83.33%` |
| 最新可恢复checkpoint | Step130（另有Step65） |
| GPU 6 / 7 | 各约34.1 GiB |
| host available | 约1.8 TiB；OOM/错误为0 |

它可以称为“行为已进入高成功平台，工程训练正常”，但fixed-12每次只有12条，且尚未完成200步，不能称严格
统计收敛。RLT v3精确停止、A/B/A′与v4启动/Step25保存期间，DSRL从Step160连续推进到179，PID、namespace
与GPU映射不变；12:31 CST时RLT也已继续到Step28。这也是并发方法有效的现场证据。**本轮不停止DSRL，让它自然完成。**

## 6. `172.17.0.1:6389` 到底是什么

它是 **Ray 控制面地址**，不是代理、数据路径，也不表示训练跑进 Docker：

```text
                         persistent Ray head
                           172.17.0.1:6389
                            /             \
 RLT driver/worktree A/namespace A/GPU4-5/output A   DSRL driver/worktree B/namespace B/GPU6-7/output B
```

两个 driver 通过它向同一个 Ray head 注册 job、申请 placement 和启动 actor。需要显式 persistent head 的原因是：
隐式 local Ray 的生命周期属于单个 driver，不适合作为两个独立 driver 的共同控制面。

本机 `localhost/127.0.0.1` 会被 Ray 2.57 的 `resolve_ip_for_localhost()` 替换成探测到的 node IP；现场被
替换成公网地址后，本机无法通过公网地址 hairpin 回连。于是选择机器上已经存在且本机可达的
`docker0=172.17.0.1`。这只是选接口，没有改系统网络。Ray 的源码行为见
[services.py](https://github.com/ray-project/ray/blob/ray-2.57.0/python/ray/_private/services.py)，RLinf 对 external
Ray 的正式入口见 [multi-node guide](https://rlinf.readthedocs.io/en/latest/rst_source/guides/multi_node.html)。

触发条件是“使用外部 persistent Ray 且 advertised node IP 不可回连”；顺序单跑并使用隐式 local Ray 时
一般不会碰到这个边界。

## 7. 为什么需要两个 `RLINF_CODE_WORKING_DIR`

current RLinf 会把该路径下的 `rlinf/` 作为 Ray per-job runtime package 同步给远端 actor。external Ray
head 先于两个 driver 存在，不会自动知道后来每个 driver 位于哪一棵 checkout：

```text
RLT  -> RLINF_CODE_WORKING_DIR=<RLT worktree>
DSRL -> RLINF_CODE_WORKING_DIR=<DSRL worktree>
```

这不是我们新写的包机制，而是 current RLinf 对 Ray runtime environment 的既有入口；第一次未设置时，
真实出现了 `NodeProbe ModuleNotFoundError: rlinf`。Ray 官方把 runtime environment 定义为 per-job/actor
依赖隔离机制：[dependency handling](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html)。

- 若两项训练使用**同一 worktree**，两行都可指向同一路径，代码身份更简单。
- 但只要仍是两个 external-Ray job，就应保留显式 code package；同一树不会让 head 自动继承 driver cwd。
- 若把同一固定 RLinf 安装进所有 worker 并关闭 code sync，也能运行，但会弱化本次对 branch/commit 的精确
  source lock，不推荐用于算法迁移实验。

## 8. 为什么官方 `save_path=./data` 在这里必须覆盖

official RoboTwin YAML 使用相对路径 `./data`，而 external Ray worker 的现场 cwd 是
`/home/chenyiteng`。两个 job 因而都会解析成 `/home/chenyiteng/data`，与 worktree 是否相同无关。

我们没有改 RoboTwin 或 RLinf 源码，只在每个 launcher 用 Hydra 参数覆盖四条 I/O 路径：

```text
env.train.task_config.save_path=<run_root>/robotwin_data/train
env.eval.task_config.save_path=<run_root>/robotwin_data/eval
env.train.video_cfg.video_base_dir=<run_root>/video/train
env.eval.video_cfg.video_base_dir=<run_root>/video/eval
```

这只隔离数据/视频产物，不改变 loss、采样、batch 或 optimizer。它看起来低级，是因为 official 单 run
示例的相对路径在单次执行中通常没问题；组合成同机多 job 后，worker cwd 才变成共享命名空间。

## 9. RLinf 是否默认只能一次训一个

不是硬限制：Ray 支持一个 cluster 上多个 job；current RLinf 也有 namespace 冲突回退和显式 placement。
[Ray multi-tenancy FAQ](https://docs.ray.io/en/latest/cluster/faq.html#do-ray-clusters-support-multi-tenancy) 同时强调它不提供
完整的物理多租户隔离。

更准确的说法是：**RLinf 的公开 embodied 示例主要按单 driver、单实验写，尚未提供“两项完整训练并发”的
一键隔离模板。** 所以 code package、输出根、GPU placement 和单 job cleanup 需要 launcher 显式解析；这
是运行编排缺省不足，不是算法只能单跑。

## 10. 我们到底改了多少，是否臃肿

| 层 | 本轮并发改动 |
|---|---|
| RLT/DSRL 科学算法源码 | 0；objective、loss、采样和预算未改 |
| RLinf通用FSDP checkpoint合同修复 | 1 file / +2 LOC / commit `8bbd0216`；plural optimizer构造期复用既有warmup helper |
| scientific config | 0；batch、env、UTD、预算均未因并发改变 |
| venv/依赖/系统网络 | 0 |
| 稳态外层控制 | 两个 launcher 合计约 16 个功能行 + 66 行 persistent-Ray 启动器 |
| 一次性事故证据/恢复 | 约 412 行探针、精确 stop/namespace cleanup 和 Stage2-only 恢复入口；不进入算法热路径 |

正式 launcher 各约 162--224 行，但大多数是 source hash、resolved config、预算、PID/exit、observer 和实验
证据，不是为了双训练新增的框架。稳态并发机制实际只有：

1. 一个明确的 `RAY_ADDRESS`；
2. 每 job 一条 code working dir；
3. 每 job 四条绝对 I/O override；
4. 停止时按 owned PGID + exact namespace 清理。

所以目前不是“在 RLinf 外又造了一层训练框架”。等当前 formal 收束后，可把两个特定 cleanup 脚本合并为
一个约 40--50 行的参数化工具；事故探针归档，不再继续堆新 wrapper。

## 11. 这次暴露的方法论缺口

缺口不在“两个训练不能并发”，而在 smoke 覆盖：

- 以前只验证了RLT **发生过optimizer update后**可以保存；
- formal首次暴露的是 **plural optimizer仍为fresh时**保存；
- 并发 smoke 验证了两个 job 能同时 step，却没有覆盖 pre-update full checkpoint；
- 停 driver 后只看 PID 也不够，必须同时检查 exact namespace actor。

以后这类current-RLinf多optimizer port只增加一个高信息量合同检查：`pre-update save`与
`post-first-update save`各一次，并确认fresh optimizer保存不会改变step语义。无需增加多套Ray、任意barrier、延迟
checkpoint或大量诊断分支；env-gated日志只保留在事故诊断worktree，不进入formal分支。

## 12. 相关本地入口

- 并发问题与现场解决历史：
  [`08_RLT_DSRL_CONCURRENT_FORMAL_RESOLUTION_20260824.md`](08_RLT_DSRL_CONCURRENT_FORMAL_RESOLUTION_20260824.md)
- 训练指标、资源与 RLT 卡点：
  [`09_FORMAL_LIVE_METRICS_AND_RESOURCE_REFRESH_20260824.md`](09_FORMAL_LIVE_METRICS_AND_RESOURCE_REFRESH_20260824.md)
- 逐命令账本：
  [`evidence/FORMAL_CONCURRENCY_OPERATION_LEDGER_20260824.md`](evidence/FORMAL_CONCURRENCY_OPERATION_LEDGER_20260824.md)
- 本轮调查账本：
  [`evidence/RLT_CHECKPOINT_AND_MULTI_JOB_INVESTIGATION_LEDGER_20260824.md`](evidence/RLT_CHECKPOINT_AND_MULTI_JOB_INVESTIGATION_LEDGER_20260824.md)
