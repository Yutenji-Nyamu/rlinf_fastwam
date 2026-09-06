# Fast-WAM current plain GRPO formal100 启动账本（2026-09-02）

## 授权与边界

- 用户明确授权启动此前讨论并完成smoke的 Fast-WAM plain GRPO。
- 使用刚由 π0.5 DVAC 释放的GPU6/7；保留GPU4/5上的π0.5 Control与shared Ray。
- 不改变已经审定的模型、采样、优化、评估或保存参数。

## 启动前现场

- 08:58 CST：GPU6/7均约11 MiB、无compute process，`RLinf_1=0`。
- π0.5 Control wrapper仍为PID `1477572`，namespace `RLinf=15`，完整Step28并继续运行。
- Fast-WAM worktree clean，HEAD=`7b2331c55d14397cfb4cb16181470ddc8afae44a`；official Fast-WAM
  HEAD=`7faa71108368fbb3b6885649f112af607427a2d4`。
- 主机available约1.15 TiB，`/data`约1.7 TiB可用。

## 精确正式合同

- 任务/模型：`move_stapler_pad` / official Fast-WAM release。
- GPU6/7；32 train env，rollout4，128 trajectories/step；G8，16 groups。
- `H32/C24/M10/D14`，episode192，最多8 query/trajectory，即1024 query records/step。
- actor-only chunk-level GRPO；`GB1024/MB2/update2`，2 optimizer calls与2048 presentations/step；LR=`5e-6`。
- fixed32/eval5，eval video开启；DCP/save10；fresh formal100。
- 总预算：12,800 trajectories、最多102,400 query records、200 optimizer calls、204,800 presentations、
  640 fixed eval episodes。

resolved packet：

`/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1`

运行目录：

`/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1`

## 启动与初始健康状态

- 约09:06 CST启动；wrapper PID=`3589666`，observer PID=`3589667`，Ray job=`62010000`。
- current RLinf检测已有Control后自动使用`RLinf_1`；09:10 CST为15个actors。
- 两个rollout rank与两个actor rank均从正确release checkpoint/stats完成构建；每份均报告824个可训练tensor与
  `H/N/S=32/24/10`。
- 09:10:48 CST已进入`Generating Rollout Epochs: 0/4`，即首个正式rollout开始；GPU6/7约41.1 GiB/card。
- wrapper存活、exit marker不存在、fatal匹配0；主机available约1.12 TiB。
- 同时π0.5 Control仍存活并保持`RLinf=15`，完整Step28；未停止或重启shared Ray。

本轮在确认真实rollout开始后停止主动观察；尚未产生完整Step1，因此不报告训练效果或ETA。

## 15:52 CST 终态刷新

- 训练完整到Step14；末步success=`38/128=29.69%`，MA5=`32.50%`，MA10=`31.48%`；
  Step5/10 fixed32均为`14/32=43.75%`。
- Step14的KL/clip/grad=`0.000856/0.01074/10.41`，均为有限值；没有GPU OOM、Ray内存阈值或磁盘写满。
- 随后进入Step15第三次fixed32，两个EnvGroup在reset/renderer阶段先反复出现
  `OIDN Error: invalid handle`，约13秒后出现`pthread_key_create failed`，最终rank0触发
  `Fatal Python error: PyGILState_Release`，下游actor/TCPStore/NCCL错误均为连锁结果。
- wrapper和Fast-WAM actors现已退出，GPU6/7释放；最新完整可恢复checkpoint为Step10。这个现场支持
  “SAPIEN/svulkan2/OIDN评估渲染线程/句柄路径故障”的近因判断，不支持把它写成训练数值或资源耗尽。
- 本轮只读刷新，没有自动重启。完整图和小体积原始材料见
  `docs/rlinf-shenzhen-experiment-expansion/evidence/current-pair-live-20260902-1552/`。

## 18:38--18:53 CST：按 outcome 数翻倍后 fresh 重启

### 为什么不是从Step10恢复

- 用户本轮明确选择把每步独立trajectory与G8 group翻倍，属于新的科学配置，因此从release checkpoint
  fresh启动，未把旧Step10 optimizer状态混入新实验。
- GPU6/7启动前均空闲；GPU4/5的π0.5 Control和shared Ray保持原样。主机available约1.1 TiB，
  `/data`约1.6 TiB可用。

### 新合同及唯一科学参数变化

- 仍为32 train env，即16 env/GPU；`rollout_epoch: 4 -> 8`。
- 每步`128 -> 256 trajectories`，G8从16组变32组；每条仍最多8个C24 query，故
  `1024 -> 2048 query records`。
- `actor.global_batch_size: 1024 -> 2048`；MB2、update2不变，因此仍是每个update epoch完整看一遍
  当前数据、每步2次optimizer call；presentations从2048变4096。
- 模型、任务、H32/C24/M10、LR、GRPO advantage/loss、32-env并发、fixed32/eval5、save10/DCP均不变。

resolved packet：

`/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets/fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2`

运行目录：

`/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2`

### 启动过程与窄修

- 第一次bootstrap在模型初始化前自然失败：新wrapper漏传official Fast-WAM `src`到`PYTHONPATH`，
  报`ModuleNotFoundError: No module named fastwam`；GPU未占用、未产生训练状态、named actors已自然退出。
- 同时发现experiment name仍保留旧`32x4/B1024`字样。v2仅恢复已验证的joint runtime路径并修正命名：
  exact current RLinf worktree + official Fast-WAM source + RoboTwin source；科学配置未改。
- v2约18:48 CST启动，wrapper PID=`826980`、observer PID=`827061`；Fast-WAM使用独立
  namespace `RLinf_1=15`，π0.5 Control仍为`RLinf=15`。
- 18:53 CST两个rollout rank和两个actor rank已从正确source构建6.02B MoT（video 5.00B、action 1.02B），
  进入`Generating Rollout Epochs: 0/8`；18:58 CST已完成第一轮真实wave `1/8`，用时5分29秒。
  fatal/exit marker均不存在；同时π0.5 Control完整到Step52，两个namespace仍各15 actors。

本轮在真实rollout开始后停止主动观察；尚未完成Step1，因此不报告效果。预期普通step约42--44分钟，
fixed-eval step约49--51分钟；这是由旧128-run实测外推，不是已完成实测。

## 2026-09-03：16-env pi0-style offload formal100 fresh启动

用户判断前一晚的单步资源smoke已足够，并授权在原GPU6/7重启Fast-WAM plain GRPO。03:25 UTC启动前
现场没有Fast-WAM/RLinf训练：GPU6/7均为0 MiB；GPU0仅有liwenbo的StarVLA服务，GPU1--7其余为空；
主机`MemAvailable`约1.93 TiB，`/data`余约1.5 TiB，shared Ray `172.17.0.1:6389`健康。目标worktree
clean且HEAD仍为`7b2331c55d14397cfb4cb16181470ddc8afae44a`，official Fast-WAM仍为
`7faa71108368fbb3b6885649f112af607427a2d4`。

新run和packet同名：

`fastwam-grpo-control-formal100-2gpu16x16-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-pi0style-v3`

完整路径位于`/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/{runs,packets}/`。新formal直接继承
成功的16-env单步smoke；相对该smoke的resolved逐叶差异只有`runner.max_steps: 1 -> 100`以及六个
run-scoped命名/路径叶。相对停止的32-env×rollout8 formal，resolved逐叶差异只有：

- `env.train.total_num_envs: 32 -> 16`；
- `env.train.rollout_epoch: 8 -> 16`；
- `env.train.enable_offload: true -> false`；
- `env.eval.enable_offload: true -> false`；
- `actor.enable_offload: false -> true`；
- 六个run-scoped命名/路径叶。

自动审计输出为`FORMAL_RESOLVED_DIFF_OK unexpected=0`。`rollout.enable_offload=true`保持不变；256条
trajectory、32个G8 group、最多2,048 query records、GB2048/MB2/update2、H32/C24/M10/D14、LR5e-6、
fixed32/eval5和DCP/save10均未改变。100步总预算为25,600 trajectories、最多4,915,200 action slots、
最多204,800 query records、200次optimizer call、409,600 record presentations和640个fixed-eval episodes。

03:33 UTC fresh启动，wrapper/owned PGID=`2153012`，observer=`2153166`。启动脚本、resolved packet、
SHA256和资源observer均已落在上述run-scoped目录；未停止或重启shared Ray，也未触碰其他用户进程。
03:37 UTC两个rollout rank与两个actor rank均完成6.02B MoT构建，日志进入真实
`Generating Rollout Epochs: 0/16`。wrapper仍存活、exit marker不存在、fatal/OOM扫描为空；GPU6/7约
`34.96/35.32 GiB`，主机仍约1.86 TiB available。到此停止主动观察；尚无完整Step1，不报告训练效果。
