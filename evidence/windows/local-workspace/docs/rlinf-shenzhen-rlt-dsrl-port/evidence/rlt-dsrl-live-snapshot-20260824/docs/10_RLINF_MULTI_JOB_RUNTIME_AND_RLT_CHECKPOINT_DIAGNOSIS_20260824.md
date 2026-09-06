# RLinf 双训练运行层与 RLT 首次 checkpoint 卡住：短结论

日期：2026-08-24  
机器：SZ-H100  
范围：RLT GPU 4--5、DSRL GPU 6--7；动态状态以服务器现场为准。

## 1. 当前应该怎么判断

1. **双训练本身已经跑通。** 两个独立 RLinf job 曾长期同时推进；RLT 卡住后，DSRL 仍继续推进并在同一
   persistent Ray 上成功保存 Step 65 和 Step 130 两个约 33.59 GB 的 DCP checkpoint。
2. **RLT 当前不是学习失败。** Stage 2 只完整到 Step 24；Step 25 eval 已完成，但首次 full checkpoint
   卡住。此时 `update_step=0`，在线 actor/critic 尚未发生一次 optimizer update。
3. **最强假设是“首次真实更新前的 synthetic-only/partial optimizer state 进入 FSDP+DCP 提取路径”。** 这是有现场
   对照和 PyTorch 已知问题支持的假设，还不是最终定因。
4. **DSRL 已进入高成功率平台，但不宜叫严格统计收敛。** 10:52 CST 完整到 Step 154/200，learned
   train success 最近 20 步均值 86.25%；fixed-12 最近五点均值 80%，单点仍在 58.33%--91.67% 波动。
   数值、显存和主存正常。建议自然跑完，不在这里中断。

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

## 3. 根因候选：证据权重

| 候选 | 当前权重 | 支持或反对证据 |
|---|---:|---|
| synthetic-only/partial optimizer state | **最高** | formal 首次保存时 `update_step=0`；同代码、同 torch 的 smoke 在 8 次真实 update 后保存成功。RLinf 已用 LR=0 synthetic step 建过部分 optimizer state，所以它不是简单的“完全空 optimizer”；PyTorch 官方已有“非空但部分初始化状态被误当完整”的 issue |
| FSDP process-group / collective rank 假设不一致 | 中低，待最小复现 | PyTorch 有 optimizer-state gather 因 process group/rank 假设而 hang 的官方 issue；但物理 GPU 4--5 不等于 torch global rank 4--5，本次两个 actor 可能仍是独立 PG 的 rank 0--1，不能据 GPU 编号套用该 issue |
| 通用 FSDP collective 顺序不一致或 PyTorch 2.11 回归 | 中低 | 官方有 FSDP save collective hang 报告；本次双 rank 同时停在 optimizer-state 路径，仍需栈或复现进一步区分 |
| DSRL、共享 Ray、系统资源或输出冲突 | **低，现场反证强** | RLT 卡住后 DSRL 继续训练并两次成功 DCP save；两 job 的 GPU、namespace、code package、输出目录均隔离，host/disk/Ray 健康 |

相关 PyTorch 官方问题：

- 非空但部分初始化的 optimizer state 可能绕过完整初始化：
  [PyTorch #192202](https://github.com/pytorch/pytorch/issues/192202)；对应的逐 parameter 初始化修复仍是
  未合并 PR [#192307](https://github.com/pytorch/pytorch/pull/192307)。
- unused parameter 缺 optimizer state 的官方案例：
  [PyTorch #164257](https://github.com/pytorch/pytorch/issues/164257)。这些报告主要表现为 missing key/加载错误，
  不是本次 hang 的直接复现，所以只能支撑“这条路径有已知边界”，不能替代现场 A/B。
- FSDP optimizer-state gather 的 process-group hang 类问题：
  [PyTorch #155680](https://github.com/pytorch/pytorch/issues/155680)；它只说明一种已知故障类别，**尚不能直接
  定性本次事故**。
- 一般 FSDP checkpoint collective hang 报告：
  [PyTorch #143536](https://github.com/pytorch/pytorch/issues/143536)。

因此当前准确措辞是：**RLT 卡在首次真实更新前的 FSDP/DCP optimizer-state 提取到 writer 之间；
synthetic-only/partial optimizer state 是
主假设，尚需一个最小 A/B 才能定因。** 现在不应直接改 barrier、patch PyTorch 或宣称 DSRL 并发导致。

## 4. 最小定因与恢复方案

需要先精确停止已经不可恢复的 RLT v3 owned PGID，并只清它的 exact namespace actor；不能执行全局
`ray stop`，否则会伤到仍在运行的 DSRL。之后只做一个小矩阵：

| 复现 | Ray 条件 | optimizer 状态 | 问题回答 |
|---|---|---|---|
| A | 当前 shared Ray，DSRL 仍运行 | 0 次真实 update 后立即 full save | 能否稳定复现本次边界 |
| B | 同一 shared Ray | 至少 1 次真实 update 后 full save | synthetic-only 与真实 initialized optimizer 是否决定结果 |
| C（仅 A/B 不能解释时） | DSRL 完成后的独立 Ray | 0 update 后 full save | shared-control-plane 是否参与 |

首选正式修复不是增加一串防御代码，而是：

- eval 仍可按 Step 25 执行；
- 在 optimizer 尚未初始化时不做包含 optimizer 的 full DCP；
- 首个 full checkpoint 放到第一次真实 optimizer update 之后。

如果必须让 warm-up 阶段也可恢复，再单独设计“模型/target/replay/RLT sidecar 已保存、未真实更新的 optimizer 显式缺省”
合同；不能把缺 shard 的目录标成 complete。这个方案会改变恢复合同，只有 A/B 证实后再实现。

## 5. DSRL：现在停还是继续

10:49--10:52 CST 现场：

| 项目 | 当前值 |
|---|---:|
| 完整进度 | 154/200；Step 155 训练中 |
| learned success mean | 73.06% |
| recent 30 / 20 / 10 / 5 | 85.83% / 86.25% / 85% / 85% |
| 最新 fixed-12 | 91.67%；last-5 mean 80% |
| replay resident | 4,495 |
| 累计已规划并执行 update | 80,300 |
| GPU 6 / 7 | 34.1 / 34.4 GiB |
| host available | 约 1,898 GiB；PSI/OOM 为 0 |

它可以称为“行为已进入高成功平台，工程训练正常”，但 fixed-12 每次只有 12 条，且还有 23% formal 预算，
不能称严格统计收敛。若现在强停，最新可恢复点仍是 Step 130，会丢掉 24 个完整 cycle；自然跑完预计约
2.8 小时，还会得到 Step 156/169/182/195 四次 fixed-12 和 Step 195 checkpoint。**因此本轮不停止 DSRL。**

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
| RLT/DSRL 算法源码 | **0 files / 0 LOC / 0 commits** |
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

- 以前验证了 RLT **发生过 optimizer update 后**可以保存；
- formal 首次暴露的是 **optimizer 尚为 fresh 时**保存；
- 并发 smoke 验证了两个 job 能同时 step，却没有覆盖 pre-update full checkpoint；
- 停 driver 后只看 PID 也不够，必须同时检查 exact namespace actor。

因此以后这类 current-RLinf port 的最小检查应增加一项：`pre-update save` 与 `post-first-update save` 各一次。
这是一个高信息量的合同检查，不需要膨胀成大量防御流程。

## 12. 相关本地入口

- 并发问题与现场解决历史：
  [`08_RLT_DSRL_CONCURRENT_FORMAL_RESOLUTION_20260824.md`](08_RLT_DSRL_CONCURRENT_FORMAL_RESOLUTION_20260824.md)
- 训练指标、资源与 RLT 卡点：
  [`09_FORMAL_LIVE_METRICS_AND_RESOURCE_REFRESH_20260824.md`](09_FORMAL_LIVE_METRICS_AND_RESOURCE_REFRESH_20260824.md)
- 逐命令账本：
  [`evidence/FORMAL_CONCURRENCY_OPERATION_LEDGER_20260824.md`](evidence/FORMAL_CONCURRENCY_OPERATION_LEDGER_20260824.md)
- 本轮调查账本：
  [`evidence/RLT_CHECKPOINT_AND_MULTI_JOB_INVESTIGATION_LEDGER_20260824.md`](evidence/RLT_CHECKPOINT_AND_MULTI_JOB_INVESTIGATION_LEDGER_20260824.md)
