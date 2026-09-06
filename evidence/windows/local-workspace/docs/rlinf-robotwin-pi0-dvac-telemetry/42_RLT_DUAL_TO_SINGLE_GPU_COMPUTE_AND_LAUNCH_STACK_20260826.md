# RLT 双卡迁移与双单卡并发：计算和启动层简明说明

日期：2026-08-26  
范围：历史双卡 RLT、当前单卡 control、当前单卡 success-episode DVAC-BC。

## 1. 先给结论

显存占用低只说明模型和一个 microbatch 能装进单卡，不表示第二张卡没有参与计算。当前 RLT 的双卡训练是
数据并行：两张卡同时各算一半样本；迁成单卡后，唯一的一张卡要依次算完全部样本。因此单条实验约慢一倍
是合理的。

本轮并未把所有物理行为保持不变。保持不变的是一次 optimizer update 仍使用 512 个样本、学习率和 loss
配置；变化的是 world size、每卡串行 microbatch 数、rollout rank 数，以及 rank-local replay 的机器级容量
和开始更新的时点。

当前两条单卡任务同时占用 GPU0/GPU1，所以单条约 41 小时，但两条也会大约同时完成。历史上一条双卡
480-cycle 约 20.2 小时；若用同两张卡顺序跑两条约 40.4 小时。两种安排的机器级总吞吐接近。

## 2. 显存、GPU利用率与计算速度不是同一个量

- 显存是容量：保存模型、optimizer state、activation 和 batch。
- GPU利用率是采样窗口内有没有 kernel 在执行，不等于显存比例，也不直接等于 Tensor Core 的完整利用率。
- CPU RAM同样主要是容量；空闲 RAM 不会自动替GPU完成矩阵计算。

历史双卡显存峰值约 19.5 GiB/card，当前单卡任务约 25.7 GiB/card，都远低于80 GiB。但两组运行的GPU
利用率都是“等待环境时较低、训练或推理时冲到100%”的突发结构：

| 运行 | GPU长期均值 | GPU P95 | 含义 |
|---|---:|---:|---|
| 历史双卡RLT | 每卡约29%--30% | 100% | 活跃阶段两卡会同时满载，其他阶段等待环境/通信 |
| 当前两条单卡并发 | GPU0/1约27.4%/26.5% | 100% | 两个独立job各自在自己的卡上突发运行 |

所以“平均只有约四分之一”更接近“100%工作一段、随后等待一段”，而不是“一直只用了四分之一算力”。

当前两条单卡并发截至Step128/129的资源分段为：

| 阶段 | 两卡平均利用率 | 两卡显存P95 | pair cgroup RAM |
|---|---:|---:|---:|
| cycle1--66，尚未online update | 32.1% / 30.3% | 25.3 / 25.3 GiB | 均值103.7、峰123.5 GiB |
| cycle67以后，online SAC | 24.0% / 24.3% | 25.3 / 25.2 GiB | 均值141.6、峰158.5 GiB |
| 最近1小时 | 26.2% / 27.6% | 25.3 / 25.2 GiB | 均值156.1 GiB |

显存阶段间变化不大；RAM上升主要跟replay、环境对象和cache累计有关。历史双卡单job在cycle200时显存峰约
19.5 GiB/card，但cgroup曾到240 GiB，其中约160 GiB是file cache。当前两条job暂时只有约156.9 GiB且
memory event为0，但尚未跑完，不能用当前中途值断言最终一定低于历史。

## 3. 一个optimizer update实际怎样计算

`global batch`是一次optimizer真正更新参数前共同参与求平均的样本总数；`microbatch`是一次放进一张GPU做
forward/backward的子批大小。拆microbatch主要是为了限制activation显存。

共同参数是：

```text
global batch = 512
microbatch = 128
```

RLinf源码按下面关系自动计算梯度累积次数：

```text
gradient accumulation = 512 / 128 / world size
```

| 运行 | world size | 每个rank串行次数 | 同时工作的rank | 最终样本数 |
|---|---:|---:|---:|---:|
| 历史双卡 | 2 | 2 | 2 | 2 × 2 × 128 = 512 |
| 当前单卡 | 1 | 4 | 1 | 1 × 4 × 128 = 512 |

之所以能分开再合并，是因为总loss是各样本loss的平均，梯度也可写成各样本梯度的平均：

$$
\nabla_\theta\left(\frac{1}{512}\sum_{i=1}^{512}\ell_i\right)
=\frac{1}{512}\sum_{i=1}^{512}\nabla_\theta\ell_i.
$$

每个microbatch先backward，把梯度累加到同一参数；双卡再用all-reduce把两个rank的梯度求和/平均；全部512个
样本到齐后才执行一次optimizer step。除浮点求和顺序和随机流外，它保持了同一个global-batch目标。

如果一个 microbatch 的 forward/backward 暂记为一份工作：双卡各做两份并行完成；单卡要连续做四份。
单卡省掉了跨卡梯度归约，但这通常抵不过多出来的两份串行计算。因此一次 update 接近两倍时间。

这里没有人工把梯度累积从2改成4；我们只把 placement 从两卡改成单卡，RLinf依据world size自动得到4。
没有把microbatch提高到256，是为了保持已成功配置的训练语义。显存余量意味着以后可以单独测试更大
microbatch是否提高吞吐，但它不是“免费恢复双卡速度”，也不应悄悄混入当前A/B。

`microbatch=256`会让单卡从4轮变成2轮，但每轮样本也翻倍，总FLOPs并未减半；收益只来自更好的kernel
并行度和更少的启动开销。当前稳定普通cycle中update约占26%，即使假设update能神奇地快一倍，整步也只
约缩短13%，因为主要时间仍在rollout。

## 4. rollout与replay也不是逐点相同

总train env仍是8，但物理分配变化了：

```text
历史双卡：两个actor/env/rollout rank，大致各服务4个env
当前单卡：一个rank服务全部8个env
```

因此policy inference、环境数据收发和eval也少了一层跨卡并行。

另外，`warmup_min_size=10k`和replay `50k`是每rank参数：

```text
历史双卡：两个rank都到10k后开始更新，机器级约20k；总槽位约100k
当前单卡：唯一rank到10k即开始更新；总槽位约50k
```

当前单卡在cycle67开始online update，历史双卡约cycle135才开始。因此“同为Step100”的墙钟不是完全
相同阶段：当前已经计算大量SAC更新，历史当时主要仍在收集。当前control与方法版共享这一单卡语义，因此
它们彼此仍是matched A/B；但不能把它们称作历史双卡逐点复刻。

### 4.1 实测每步时间结构

| 运行/阶段 | rollout | actor/critic update | eval | cycle墙钟 |
|---|---:|---:|---:|---:|
| 历史双卡cycle68，纯收集 | 122s | 0 | 0 | 123s |
| 当前单卡cycle66，纯收集 | 229s | 0 | 0 | 230s |
| 当前单卡cycle68，首次update追账 | 222s | 189s | 0 | 411s |
| 历史双卡cycle200后普通cycle | 未单独保留 | 未单独保留 | 0 | 约150s |
| 当前单卡control cycle129 | 217s | 77s | 0 | 295s |
| 历史双卡cycle100评估点 | 125s | 0 | 268s | 417s |
| 当前单卡control cycle125评估点 | 198s | 70s | 540s | 867s |

所以当前稳定普通step并非主要都在反向计算：cycle129约74%是rollout、26%是update。双卡变单卡后，
rollout本身也从两个rank并行服务8个env变成一个rank服务8个env；纯收集阶段实测已从约122秒变为229秒。

阶段也整体前移：历史为cycle1--135收集、136--154开始SAC、155--191 student ramp、192后稳定student；
当前为1--66收集、67开始SAC。2026-08-26 11:15时control/method在129/128，update约75.1k/74.8k，已超过
20k warmup加50k ramp，属于online SAC与student/reference稳定混合执行阶段。

## 5. 基于历史双卡RLT到底改了什么

### 5.1 为单卡共同改变

- `component_placement: 0-1 -> 0`或`1`；world size随之`2 -> 1`；
- 梯度累积由RLinf自动`2 -> 4`，global/micro batch仍为`512/128`；
- 用户选择fresh 480，所以历史resume入口改为`resume_dir: null`；
- 每条任务使用独立绝对日志、数据、视频与checkpoint路径。

8 train env、20次fixed eval、loss系数、学习率、update schedule、字面replay容量、seed、Stage1与π0权重
均未改。

### 5.2 DVAC-BC相对单卡control额外改变

并发本身没有要求修改算法Python。方法版只增加已有文档冻结的三件事：成功episode回填标签、成功样本
切换到executed-action BC target、成功C10内部使用mean-one DVAC权重；原actor-Q和critic TD保持原RLT。
control与方法resolved除方法字段、GPU placement和run路径外，意外差异为0。

## 6. 当前启动栈到底有几层

实际运行链是：

```text
本机Paramiko command-file                 只负责把一次启动命令送到服务器
  -> pair launcher                       一次性启动shared Ray、两个job、监控和结束清理
      -> shared Ray head                  给两个job统一GPU坐标系
      -> run-one wrapper × 2              集中设置环境/路径，resolve配置，再调用原生入口
          -> train_embodied_agent.py      RLinf真正的训练入口
              -> Ray actor/env/rollout    正常RLinf训练循环
```

旁路的resource monitor每5秒只读一次RAM/GPU；cleanup watcher只在两条job都结束后关闭shared Ray。它们不包住
每个训练step，也不参与loss、batch或梯度。训练期间长期多出的运行层实际上只有一个等待子进程退出的bash
wrapper，对吞吐没有实质影响。

当前正式运行使用3个主脚本：

- `autodl_start_shared_ray_formal480_v2_20260825.sh`：39行；
- `autodl_run_one_rlt_success_bc_formal480_v2_20260825.sh`：92行；
- `autodl_launch_rlt_success_bc_dual_single_gpu_formal480_v2_20260825.sh`：94行。

大量`preflight`、`health`、`live_probe`和v1--v9脚本是开发/取证时的一次性命令，不是当前训练层层调用的
生产栈。它们数量看起来多，但不会在每次step中执行。

## 7. 本轮并发遇到的问题与最小解决

| 问题 | 原因 | 最终处理 | 是否改训练数学 |
|---|---|---|---|
| 早期launcher没有进入repo | Python入口使用相对路径 | run-one统一先`cd`到source root | 否 |
| 两个独立Ray head都落到物理GPU0 | 每个head把唯一可见卡重编号为逻辑0 | 一个shared Ray看见0/1，placement分别0/1 | 否 |
| 手工namespace后worker找不到manager | manager和worker进入不同查找空间 | 不手工设置，让RLinf自动使用`RLinf/RLinf_1` | 否 |
| Ray Unix socket路径过长 | AF_UNIX路径长度有限 | Ray临时目录使用短`/tmp`路径 | 否 |
| 两任务同时冷启动helper过多 | TorchInductor并发编译 | 编译线程设1，第二条错峰120秒 | 否，仅启动慢一点 |
| formal v1训练前import失败 | launcher漏了smoke已有的RoboTwin `PYTHONPATH` | v2在run-one集中恢复同一环境 | 否 |
| replay终态`next_obs`键不一致 | auto-reset附带了RLT不消费的键 | replay只保留core observation schema | 是代码修复，但不是并发算法改动 |

这些问题主要集中在第一次让两个RLinf job共享一套Ray/GPU坐标系，并不意味着每次训练都要重新诊断一遍。
当前真正值得消除的脆弱点是`run-one`手工抄了一遍RLT/RoboTwin环境变量；formal v1漏`PYTHONPATH`已经说明
环境合同应该只有一份来源。

## 8. 启动层还能怎样精简

当前运行不应中途改。以后建议把已验证知识收敛成两层，而不是继续复制日期版脚本：

1. 一个稳定的`start_shared_ray.sh`：容器生命周期只启动一次；
2. 一个通用`launch_rlinf_pair.sh`：内部一个`run_job()`函数，集中保存环境变量、两份config/run path、
   错峰、轻量monitor和结束清理。

更理想的是再把source、RoboTwin、Stage1、norm stats等export收敛到唯一`rlt_runtime_env.sh`，两个job只
`source`它，不再复制环境。三个当前脚本可以折成一个约百行的长期pair launcher；GPU1的薄YAML可选择保留
以便复现，也可改成显式placement override。

正式入口仍直接是RLinf的`train_embodied_agent.py`。保留的最小记录只有exact command、resolved YAML、PID、
exit code和resource CSV。以后不把版本化preflight/health/live-probe复制进生产启动包；这些只作为历史证据或
临时只读命令。

不能删除的三件事是：shared Ray统一GPU坐标、每job独立placement/namespace、每job独立绝对输出路径。
它们是两个RLinf任务不互相踩卡、撞actor名或混写checkpoint的边界，不属于多余算法包装。

## 9. 对当前耗时的准确表述

截至2026-08-26 10:38，control/method累计约`5m09s/5m11s`每cycle；历史双卡完整480平均
`2m31.6s`每cycle。约2.04倍不是“计算与资源无关”，恰恰说明历史第二张卡确实承担了并行工作。

更准确地说：单条任务的高层batch语义尽量保持，但物理计算从“两rank并行”变成“一rank串行”；同时
rank-local replay时点也改变。两条单卡并发后，机器级总实验吞吐与两卡顺序跑两条历史实验大致相当。
