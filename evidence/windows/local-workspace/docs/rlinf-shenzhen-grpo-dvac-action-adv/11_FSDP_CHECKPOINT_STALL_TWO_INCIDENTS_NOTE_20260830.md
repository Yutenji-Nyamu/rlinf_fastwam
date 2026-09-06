# 两次 FSDP checkpoint 卡住：问题与处理

日期：2026-08-30；机器：SZ-H100；范围：两卡 current RLinf GRPO 系列。

## 1. 两次发生了什么

| 事件 | 现场 | 结果 |
| --- | --- | --- |
| Prism v1，2026-08-27 | 完整到Step9；Step10评估后卡在默认DCP/FSDP optimizer-state保存。目录没有shard，GPU util归零；同一shared Ray上的Control继续运行 | Step10不可恢复 |
| ST-DVAC `[0.8,1.2]` v1，2026-08-30 | 完整到Step9；Step10评估后同样卡在默认DCP保存。目录仅12 KiB、0文件，无fatal/OOM/磁盘压力；并行Action继续到Step14 | Step10不可恢复 |

共同边界很明确：训练与评估已经完成，卡点都在**第一次分布式checkpoint协调**，不是训练数值崩溃、
显存OOM、磁盘写满或shared Ray整体故障。

## 2. 为什么会卡

默认DCP需要多个FSDP rank共同进入PyTorch distributed-checkpoint的optimizer-state提取与保存协议。
两次现场都是rank没有完成这次会合，因此还没生成`.distcp`或`.metadata`文件。

我们没有把更底层原因唯一归结为某一个PyTorch collective bug；实际处理是绕开这条不稳定的DCP协调
路径，改用RLinf已有的`local_shard`保存路径。

`local_shard`让每个rank保存自己的checkpoint shard，避免本次卡住的跨rank DCP保存协议。它只改变
checkpoint格式，不改变rollout、advantage、loss、batch、optimizer update或模型训练数学。

## 3. 第一次怎么解决

Prism专用commit `306ce2e98a06b6f439a1070d8942e20132e48d49`只修改一个文件：

- `rlinf/hybrid_engines/fsdp/fsdp_model_manager.py`，`+14/-1`；
- 读取并校验`actor.fsdp_config.checkpoint_format`；
- 在save/load两侧对称透传`dcp | local_shard`；
- 默认仍为`dcp`，只有实验配置显式选择`local_shard`。

Prism v2随后成功保存Step10/20/30/40/50，每次生成两份rank-local checkpoint。因此这不是只通过一次
偶然保存，而是同一软件栈上连续五个保存周期的实证。

## 4. 为什么第二次又发生

第一次修复只提交在`codex/sz-prism-dvac-rank-rloo`专用分支，没有进入ST使用的
`0e28ac6f...`分支；ST的manager没有格式透传，resolved也没有`checkpoint_format`，启动清单仍明确写着
`default DCP every10`。

所以第二次不是`local_shard`失效，而是ST根本没有使用此前的修复。并行Action的默认DCP碰巧成功保存
Step10，也不能证明ST这次DCP协调一定会成功。

## 5. 第二次怎么解决

1. 从ST原head建立独立分支`codex/sz-st-dvac-local-shard`。
2. 只移植上述一文件补丁，新head为`f2a543da87afb7d5e3aa030ef52e53df9a266b28`并已push。
3. resolved只新增`actor.fsdp_config.checkpoint_format=local_shard`及新source/run路径；科学参数逐叶不变。
4. 精确停止旧ST的owned PGID与`RLinf_1`；Action和shared Ray不动。
5. fresh v2已在GPU6/7进入首个rollout。真正的保存闭环证据是Step10出现
   `checkpoint_rank_0.pt`与`checkpoint_rank_1.pt`。

## 6. 以后固定怎么做

- 这套两卡FSDP GRPO正式训练显式使用`checkpoint_format=local_shard`，不要依赖默认DCP。
- 启动前在resolved中直接确认该叶子；不能只看某个旧分支曾经修过。
- local-shard恢复必须保持相同world size和FSDP拓扑。
- checkpoint目录没有任何shard时，该step不能声称可恢复；保留事故目录，从上一个完整checkpoint或fresh运行。

详细操作证据：

- 第一次：[Prism checkpoint修复流水账](../rlinf-shenzhen-prism-dvac-grpo/evidence/CHECKPOINT_FIX_AND_FORMAL_V2_LEDGER_20260827.md)
- 第二次：[ST切换与修复流水账](evidence/ACTION_MID_ST_NARROW_CUTOVER_LEDGER_20260830.md#6-st-step10-dcp-卡住与-local-shard-v2)

