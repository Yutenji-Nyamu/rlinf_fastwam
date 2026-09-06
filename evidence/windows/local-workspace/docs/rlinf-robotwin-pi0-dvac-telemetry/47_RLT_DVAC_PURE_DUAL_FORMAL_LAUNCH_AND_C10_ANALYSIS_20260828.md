# RLT-DVAC-Pure 双单卡 formal 启动与 C10 数据分析

日期：2026-08-28  
状态：两条fresh 480-step formal均完成Global Step 1并继续运行；已停止主动盯跑。

## 1. 正式实验

| 项目 | GPU0 | GPU1 |
|---|---|---|
| 方法 | Pure reference-BC | Pure reference-BC |
| 用户简称 | `[0,2]` | `[0,5]` |
| 精确参数 | `strength=.5` | `strength=2.0` |
| 实验名 | `...pure_reference_bc_s0p5_formal480_20260828_v1` | `...pure_reference_bc_s2p0_formal480_20260828_v1` |
| 首轮rollout | 232.42秒 | 223.78秒 |
| 启动证据 | `Global Step 1/480` | `Global Step 1/480` |

二者都继承matched-width单卡干净RLT：train env8、eval`4×5=fixed20`、global/micro batch
`512/256`、warmup20k、replay80k、UTD5、每cycle最多1600次critic update、actor 20k/50k
warmup/ramp、480 cycles、eval/save25、fresh入口，以及相同Stage1、π0、norm、seed、loss和LR。

相对干净RLT，只有Pure方法字段和必要输出路径不同；两条Pure之间只有`strength`、GPU placement、名称与
输出路径不同。源码HEAD=`a2ae5cbe81049fb7027c43ea85483ed3ffc3ce2f`，两条compose均为0，placement
现场为GPU0/GPU1，OOM/OOM-kill均为0。

这里的`[0,2]/[0,5]`是沿用讨论时的强度简称。由于每个C10最后会归一到均值1，最终实际weight并不被严格
限制在这两个区间；正式产物用`s0p5/s2p0`保留精确含义。

## 2. 单卡是否实现了原目标

| 量 | 历史双卡干净RLT | matched-width单卡干净RLT |
|---|---:|---:|
| GPU峰值 | 19.37/19.56 GiB，合计38.9 | 一张卡25.17 GiB |
| 平均cycle | 151.6秒 | 262.0秒（到Step476） |
| 墙钟 | 480步20小时12分49秒 | 476步34小时38分49秒 |
| train env | 2 rank×4，总8 | 1 rank×8，总8 |
| global/micro batch | 512/128 | 512/256 |
| 梯度累积 | 每rank 2轮，两卡并行 | 单rank 2轮 |
| fixed eval | 2 rank×2×5=20 | 1 rank×4×5=20 |
| replay/warmup | 2个rank-local 50k；约20k总warmup | 1个80k pool；20k warmup |

结论是“容量和总预算语义基本实现，速度与数据拓扑不是逐点复刻”：

- 一张A800确实能同时放下完整8-env、B512训练；单实验总显存由约38.9降到25.2 GiB，估算GPU-hours由
  约40.4降到约34.5，且两张卡可以并行跑两个独立实验。
- 单实验仍约慢`1.72×`。原因不是显存，而是双卡原本有两份物理计算和两个rollout rank；单卡虽把
  microbatch加到256，仍只有一份GPU算力和一个rank服务8个env。
- 两条单卡并发的cgroup RAM峰值约239.84 GiB，与历史双卡单job曾触240 GiB同量级；没有证据说明RAM峰值
  被明显降低。
- 训练高层形态保留：后期都进入高成功率区。但到共同Step476，历史双卡/单卡control的累计train success
  为`60.56%/55.49%`，19次fixed20累计为`213/380 vs 200/380`。因此不能称为数值或随机过程完全一致；
  主要剩余差异是rank-local replay变统一pool、worker/rank数和随机流/样本混合。

所以它达成了“用一张卡承载原来的总env与optimizer batch，并提高两项实验并发吞吐”；没有达成“单条实验
与双卡同速、同轨迹”。

## 3. RLT的H50生成与C10训练链

```text
当前观测
  -> frozen π0做4步ODE，生成action [B,50,14]
  -> 同时得到endpoint preview [B,4,50,14]
  -> 保存L2/L3/L4 DVAC [B,3,50]

π0 reference取h0..9             [B,10,14]
student输出                       [B,10,14]
环境执行当前route的C10

Pure训练选择L3并再次显式切h0..9   [B,10]
  -> mean-one weight              [B,10]
  -> 乘成功episode的逐h reference-BC误差
  -> 对query、h、14D求平均为BC loss
```

因此实现已经正确考虑“π0推理H50、RLT只执行/训练C10”：`h10..49`完整保存在replay中用于诊断，但不会
进入Pure BC权重。它们也不能接成后续四个query，因为执行10步后环境状态已改变，student/π0会重新查询。

与旧C50推理数据相比：每次query内部的M4/H50/14D DVAC计算相同；旧评估每次执行50格，一条200-slot
episode通常只有约4个query；RLT每次执行10格，最多约20个query，所以状态时间轴密约5倍，但每次query
只有前10格真正对应本次执行/BC。

## 4. 完整rollout能恢复什么

旧Step475完整replay有57,957个query row，均含：

- `teacher_dvac_v=[L2,L3,L4,H50]`；
- `ref_chunk/actions=[C10,14]`；
- reward/done、`episode_success`和student/reference route。

按replay的权威ID顺序和done可恢复完整数值episode。扫描最后4,000行得到329条完整student episode：
成功281、失败48，并导出成功/失败各低、中、高DVAC共6条。

主要现象：

- 87.39%的query中，未执行的`h10..49`均值高于实际C10；平均高`0.222 log10`，约`1.67×`。
- 成功episode多在10--12个query结束，正reward在最后query；失败episode多跑满20个query。
- 329条聚合曲线没有显示C10 DVAC在成功前单调上升：成功均值从约`-1.96`降到末端`-2.15`，失败降到
  `-2.25`。个别轨迹确有明显早期/中期峰值，但旧replay不含相机图，不能仅靠数值判断对应抓取、接触还是放置。
- 旧数据上的幅度预览：s0p5 weight ESS=`.922`、top20质量=`27.7%`；s2p0 ESS=`.593`、top20质量
  `45.0%`，约19.7%位置归零。因此新两条方法强度差异是实质性的。

当前正式配置不保存训练图片/视频，replay只有`z_rl/proprio/reference/DVAC`等数值，无法事后还原画面。
若要给峰值精确贴任务阶段，只需以后做一轮fixed-reset、C10 control-trace评估，不需要改变当前formal。

## 5. Step146现场刷新

2026-08-28 10:31 CST两条driver均alive，最新完整Step为`s0p5=147`、`s2p0=146`，共同轴取Step146。
共同Step1--146累计training-rollout success为`15.33%/15.15%`；共同Step146的5步均值为
`15.00%/7.50%`，10步均值为`16.25%/10.00%`。到Step125两条fixed20仍均为`0/20`。

最新`actor_switch_rate=0`，说明训练rollout仍走π0 reference route；当前曲线主要描述采集分布，尚不是
student接管后的闭环效果。方法本身已生效：s0p5/s2p0的weight ESS为`.908/.586`，后者重分配显著更强。
两条均无OOM/OOM-kill，RAM现场/历史峰值约`186.24/190.29 GiB`，两卡显存峰值约`25.12/25.09 GiB`。
图、CSV与摘要见[现场证据](evidence/rlt_dvac_pure_dual_live_g146_20260828/README.md)。

## 6. 入口

- [四条单卡RLT共同Step195成功率对比](evidence/rlt_four_single_gpu_success_live_g195_20260828/README.md)
- [Step191现场成功率、fixed20与资源](evidence/rlt_dvac_pure_dual_live_g191_20260828/README.md)
- [数值时间线、四张图与CSV](evidence/rlt_step475_c10_episode_probe_light_20260828_v1/README.md)
- [正式启动逐指令流水](evidence/RLT_DVAC_PURE_DUAL_FORMAL_LAUNCH_LEDGER_20260828.md)
- [Pure实现语义](46_RLT_DVAC_PURE_REFERENCE_BC_PLAN_20260827.md)
- [单卡计算与启动说明](42_RLT_DUAL_TO_SINGLE_GPU_COMPUTE_AND_LAUNCH_STACK_20260826.md)
