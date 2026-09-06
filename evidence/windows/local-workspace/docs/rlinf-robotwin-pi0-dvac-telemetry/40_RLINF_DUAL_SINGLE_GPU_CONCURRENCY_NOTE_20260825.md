# RLinf 双单卡并发：参数语义与运行经验

日期：2026-08-25  
范围：AutoDL 两张 A800；一条单卡原始 RLT control 与一条单卡 RLT success-episode DVAC-BC。

## 1. 一句话结论

单卡装得下 RLT。两条单卡实验并发时，最合适的结构是：

```text
一个共享 Ray head（看见 GPU0、GPU1）
  ├─ job A：RLinf namespace RLinf，placement GPU0，独立代码/输出
  └─ job B：RLinf 自动换到 RLinf_1，placement GPU1，独立代码/输出
```

模型容量不是主要问题。单卡相对历史双卡主要损失吞吐；两条同时运行时，还要观察 16 个 train env 合计的
CPU、RAM 和 I/O。

## 2. 双卡改成单卡，训练 batch 怎样保持一致

历史双卡和当前单卡都保留：

```text
global batch = 512
microbatch = 128
```

梯度累积次数由 RLinf 按下面的关系自动得到：

```text
gradient accumulation = global batch / microbatch / world size
```

因此：

| 配置 | world size | 每个 rank 的 microbatch 次数 | 一次 optimizer update 汇总的样本 |
|---|---:|---:|---:|
| 历史双卡 | 2 | 2 | 2 ranks × 2 × 128 = 512 |
| 当前单卡 | 1 | 4 | 1 rank × 4 × 128 = 512 |

`world size` 是共同训练同一个模型的 GPU 进程数；`microbatch` 是一次能送入单卡的样本数；梯度累积表示先做
若干个 microbatch 的 forward/backward，把梯度加起来，最后才做一次 optimizer step。

所以单卡没有把有效 batch 减半，也没有修改学习率。高层上，一次更新仍由 512 个样本决定；差别是双卡能
并行算两份，单卡要把四份依次算完。浮点归约顺序、rank 随机流和 FSDP 拓扑会变化，因此不能要求逐 bit
一致，但优化预算的主语义一致。

## 3. 为什么单卡更慢

双卡时，每张卡只需串行处理两个 microbatch，然后做一次跨卡梯度归约；单卡不需要跨卡归约，但必须独自处理
四个 microbatch。模型 rollout 也从两个 actor rank 分摊变成一个 rank 服务全部 8 个 env。理想情况下，
optimizer 段接近约两倍计算量落在一张卡上；实际减去通信开销后通常小于严格两倍，但仍明显更慢。

两条单卡实验同时运行还会共享 CPU simulator、主存和磁盘，所以整步耗时不只由 GPU 决定。

## 4. replay 的 per-rank 容量减半意味着什么

历史配置中的 replay `cache_size=50000`、`sample_window_size=50000` 是每个 actor rank 各自持有：

```text
双卡：2 × 50k，机器上合计最多约 100k 个 rank-local 槽位
单卡：1 × 50k，机器上合计最多约 50k 个槽位
```

同时保持 8 个 train env 后，单卡的一个 replay 接收全部 env 数据，旧 transition 相对更快被新数据替换。
这可能减少很早期经验的保留时间和跨 rank 的总多样性；它不改变单次 optimizer batch 仍为 512。

`warmup_min_size=10000`也是rank-local合同。历史双卡通常要两个rank各自达到10k，机器级大约已收集20k；
单卡只需唯一buffer达到10k。因此按机器级transition数看，单卡可能更早开始online update。这是单卡相对历史
双卡的真实行为差异，不是梯度累积能够抵消的。

当前 control 与方法版都保留 `50k/50k`，因此两条单卡实验彼此公平，也保持了原代码的 rank-local 参数。
若将来目标是严格匹配历史双卡的机器级总 replay 容量，应单独批准把单卡改为 `100k/100k`；这会改变 replay
年龄分布和主存，不应悄悄混入当前 A/B。

## 5. 当前相对历史成功双卡 RLT 改了什么

两条 smoke 共同只做必要运行变化：

- placement 从两张卡改为一张卡；world size 随之 `2 -> 1`；
- 梯度累积由 RLinf 自动 `2 -> 4`，global batch 仍为 512；
- smoke 把 `max_steps/save/eval interval` 改为 1；
- 为让一轮 smoke 真正越过 warmup 并走到 actor/critic update，临时缩短 replay warmup/update schedule；
- train/eval env、rollout epoch、batch、actor/critic loss 权重、replay 字面容量及 fixed-eval 规模均在
  control 与方法版之间一致。

正式 480-cycle 配置不采用 smoke 的短 warmup覆盖；它继承历史成功 RLT 的正式 schedule。

## 6. 同一机器上多个 RLinf job 怎样隔离

从用户提供的深圳并发材料和本轮 AutoDL 现场共同保留下来的最小做法是：

1. 使用一个可达的 persistent Ray head，并让它看见两张 GPU；
2. 每条 job 用 `cluster.component_placement` 指定不同 GPU；
3. 每条 job 显式设置自己的 `RLINF_CODE_WORKING_DIR`，让 Ray worker 收到正确 worktree 的 `rlinf/`；
4. train/eval data、video、logger、checkpoint 使用 run-scoped 绝对路径；
5. 由 RLinf 自己处理 namespace：第一条通常为 `RLinf`，冲突时第二条自动改为 `RLinf_1`；
6. 停止某一条时处理其 owned process group，并核对该 exact namespace 的 actor；共享 head 上不能用全局
   `ray stop` 清一条任务。

本轮还得到三条 AutoDL 具体经验：

- Ray temp 路径要短，避免 AF_UNIX socket 超过约 107 字节；
- 不要为两个 driver 手工导出不同 `CLUSTER_NAMESPACE`。RLinf manager 在内置 namespace 创建，而 worker
  按外置值查找时会找不到 `DeviceLockManager`；
- 两条冷启动同时发生会产生大量 TorchInductor helper。运行层设置
  `TORCHINDUCTOR_COMPILE_THREADS=1`并错峰启动，只改变初始化并发，不改变训练数学。

## 7. 为什么不用两个各自独立的 Ray head

本轮真实试过两个独立 head。每个 head 只看见一张经 `CUDA_VISIBLE_DEVICES`裁剪后的 GPU，Ray 会把该设备
重新编号为逻辑 GPU0；两个 job 的物理卡映射因而不再与 RLinf 的 placement 编号一致，现场出现两条都落到
物理 GPU0。

共享 head 看见物理 GPU0、GPU1，placement `0/1`才有一个共同坐标系。namespace 隔离 actor 名称，placement
隔离 GPU，绝对路径隔离产物；三者职责不同。

## 8. 深圳材料中哪些不能直接照搬

可迁移的是上述结构，不是现场常数。深圳的 `172.17.0.1:6389`、GPU4--7、2 TiB RAM、特定 venv/worktree
和 checkpoint 修复都属于那台机器。本轮 AutoDL 使用自己的容器 IP、端口、两张 A800、240 GiB cgroup、
既有 venv 和 source lock。

深圳材料另有一个 RLT plural-optimizer 首次 checkpoint warmup 修复；它解决的是对应 current-base 的
optimizer-state 合同，不是“多 job 必须打的并发补丁”，不能仅因并发就复制到当前旧 RLT 基线。

## 9. 本轮 smoke 的判定标准

两条都需要完成：

```text
GPU placement正确
train rollout完成
fixed eval完成
replay写入成功
至少一次真实 critic/actor update
global_step_1 checkpoint完整
driver exit 0
无CUDA OOM、cgroup OOM/OOM-kill或worker fatal
```

资源监控只记录，不触发自动停止。

## 10. 本轮真实结果

v9最终满足全部判定项：两条都自然`exit 0`，各完成8 train、20 fixed eval、8 critic/4 actor update和完整
`global_step_1` checkpoint。方法版权重mean/p05/p95=`1.000/0.741/1.264`、ESS=`0.975`，确认新分支进入
apply。GPU0/1峰值=`21,223/21,194 MiB`，双job cgroup RAM峰值约78.1 GiB，所有memory/OOM事件为0；
结束后shared head与两条namespace actor均已退出。

详细结果与原始轻量材料见
[smoke evidence](evidence/rlt_success_bc_dual_single_gpu_smoke_v9_20260825/README.md)。正式480-cycle配对训练
已于2026-08-26使用同一shared-head/placement/自动namespace合同启动；当前路径与初始健康现场见
[41号短说明](41_RLT_SUCCESS_BC_FORMAL480_LAUNCH_AND_DELTA_20260826.md)。
