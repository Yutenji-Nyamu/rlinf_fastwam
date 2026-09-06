# RLT success-episode DVAC-BC：代码增量与正式480步启动

日期：2026-08-26  
状态：单卡原始RLT control与单卡success-episode DVAC-BC均已正式启动，当前只确认健康启动，尚无训练效果结论。

## 1. 代码增量到底多不多

相对历史成功RLT提交，当前分支还包含上一版teacher-DVAC telemetry与兼容路径，因此生产Python总差异为
5个文件、约`+973/-25`；这不是本轮BC方法本身的真实体量。

只看本轮success-episode BC相对其直接父版本：

- 生产Python：3个文件，约`+286/-15`；
- 正式配置：2个YAML，约`+52`；
- 测试：约`+90/-1`。

其中真正改变算法语义的主体是：

1. episode结束后把success回填给本episode的replay rows；
2. 成功episode把BC target从π0 reference切换为实际executed action；
3. C10逐位置MSE乘mean-one DVAC权重；
4. 新方法恢复原始RLT的actor-Q路径；
5. smoke发现后，把`next_obs`固定为core observation schema。

其余主要是已有teacher-DVAC的采集/统计、方法指标、shape检查、raw trace和旧`q_gradient`复现实验兼容。
所以结论是：**方法本体是窄增量，但当前分支整体不算小；不是每一行都为新BC公式所必需。** 当前已通过
10项测试与双任务smoke，不在正式训练前为了减行数重新扰动。以后可独立删除GPU1 overlay、部分重复指标、
raw trace和旧Q-gradient兼容路径。

没有新增第二个模型forward、RoboTwin控制逻辑、critic TD、optimizer、checkpoint格式或第三方依赖。

## 2. 方法实际改了哪里

原RLT actor loss仍由两部分组成：沿critic给出的方向提高Q，以及靠近BC target。新方法只改第二部分：

```text
失败/普通episode：student C10 -> 模仿π0 reference，十格权重均为1
成功episode：     student C10 -> 模仿实际成功executed action，十格按DVAC重分配
actor-Q、critic TD、π0 teacher、reward、route：不变
```

`strength=0.25`的映射是：

$$
w_h=1+0.25\left(\operatorname{clip}(z_h,-2,2)
-\operatorname{mean}_j\operatorname{clip}(z_j,-2,2)\right).
$$

`0.25`是“DVAC差异变成BC倍率差异”的斜率：某格比同一C10平均高1个截断后的$z$单位，约为`1.25x`；
低1个单位约为`0.75x`。十格均值恒为1，因此它只在成功C10内部重新分配BC，不放大整条成功样本的平均BC，
也不影响失败BC、actor-Q或critic TD。理论范围约`[0.1,1.9]`；smoke实测p05/mean/p95为
`0.741/1.000/1.264`。

## 3. 为什么过程较长

算法改动本身没有反复重写；时间主要花在两条RLinf任务同机并发的运行层。遇到的问题与窄修复如下：

| 问题 | 原因 | 处理 |
|---|---|---|
| Windows本地没有项目torch/pytest | 本地只保存代码副本 | 在服务器既有venv做测试 |
| replay终态`next_obs`字段不一致 | auto-reset wrapper混入额外键 | 只保存RLT消费的core字段 |
| 两个独立Ray head都落到物理GPU0 | 各head把唯一可见卡重新编号为逻辑0 | 改为一个shared Ray head，placement用0/1 |
| 手工namespace后worker找不到manager | manager与worker查找空间不一致 | 不手工设置，让RLinf自动用`RLinf/RLinf_1` |
| Ray Unix socket路径过长 | run目录太长 | Ray临时目录改为短`/tmp`路径 |
| 双任务同时冷启动helper过多 | TorchInductor并行编译 | 编译线程设1，方法任务延迟120秒；不改训练数学 |
| 正式v1在训练前导入失败 | launcher漏了成功smoke已有的RoboTwin `PYTHONPATH` | 保留v1现场，v2原样恢复已验证运行环境 |

因此过程长主要是“两个RLinf job如何正确共存”，不是BC方法膨胀。最后一次v1问题只改launcher环境，
没有再改代码或YAML。

## 4. 单卡与历史双卡的参数关系

历史成功RLT resolved与本轮单卡control逐叶比较，差异只有：

- placement：`0-1 -> 0`；
- 历史文件是`250 -> 480`的resume，本轮按用户决定为fresh 480，故`resume_dir -> null`；
- source内seed文件路径和run-scoped日志、数据、视频路径迁到当前干净worktree/run目录。

训练主体不变：8 train env、1 rollout epoch、4 eval env × 5 epoch=`20`次fixed eval、global/micro batch
`512/128`、update epoch 5、critic:actor=`2:1`、每cycle最多1600 updates、warmup `10k + 30k`、
actor warmup/ramp `20k/50k`、replay字面容量`50k/50k`、学习率、loss系数、seed和Stage1/π0均一致。

world size从2变1后，RLinf自动把梯度累积从2改为4，所以optimizer每次仍看512条样本；一张卡只是串行多做
两个microbatch，通常更慢。replay是per-rank的，保留`50k/50k`意味着单卡机器级总容量是历史双卡的一半；
这是已知单卡语义差异，没有暗中改成100k而同时改变样本年龄和RAM。

本轮control与方法版resolved比较，除独立产物路径/命名和placement `0 -> 1`外，唯一算法差异是
`rlt_dvac`的13个方法字段及OpenPI telemetry开关；其他训练字段意外差异为0。

## 5. 正式启动现场

共同source：`64f2779f266b7d7019895c7aee1ebc222312b7d3`，服务器worktree clean。

```text
control / GPU0
/root/autodl-tmp/experiments/rlt_single_gpu_control_formal480_20260825_v2

success-episode DVAC-BC / GPU1
/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_formal480_20260825_v2

shared Ray
172.17.0.9:52001
```

2026-08-26 00:06 CST只读确认：两条wrapper均alive；control与方法分别解析到hardware rank `[[0]]`和
`[[1]]`；两卡模型均已装载并进入rollout初始化/生成；显存约14.7/21.2 GiB；cgroup
`high/max/oom/oom_kill`均为0。按用户要求到“正常启动”即停止盯守，训练继续后台运行。

正式v1失败目录完整保留；它在训练前因漏设RoboTwin import path退出，不纳入实验结果，也没有checkpoint。

逐指令与问题现场见
[正式启动流水账](evidence/RLT_SUCCESS_BC_FORMAL480_LAUNCH_LEDGER_20260826.md)。
