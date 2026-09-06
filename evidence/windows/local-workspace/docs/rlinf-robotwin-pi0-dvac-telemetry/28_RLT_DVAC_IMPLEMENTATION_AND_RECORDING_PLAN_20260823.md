# RLT × teacher-DVAC：实现与训练记录小计划

日期：2026-08-23  
状态：**方案、实现、代码审查、真实两卡smoke与云端同步均已完成；正式fresh-480已启动并进入连续rollout。**
DSRL本轮不进入实现范围。

实现事实（2026-08-24）：

- AutoDL worktree：`/root/autodl-tmp/RLinf_rlt_teacher_dvac`；
- branch/upstream：`codex/rlt-teacher-dvac-weighting` / `personal/codex/rlt-teacher-dvac-weighting`；
- commits：主体`275ab4535f8363e15b839cbb64e7d997a056d01e`，rank-stable telemetry修复
  `513dbcb7f31ebb639afa5267dff218028d07187b`；worktree clean，ahead/behind=`0/0`；
- 检查：ruff、diff、Python语法、YAML解析通过；CPU-only单测`5 passed`；
- 真实smoke：8 train episodes、20 fixed eval episodes、8 critic/4 actor update、完整g1 checkpoint，driver
  `exit=0`；GPU峰值约17.1/17.2 GiB，cgroup峰值57.18 GiB，OOM/OOM-kill=0；
- 完整实施账：[RLT_DVAC_IMPLEMENTATION_LEDGER_20260824.md](evidence/RLT_DVAC_IMPLEMENTATION_LEDGER_20260824.md)。

## 1. 一句话目标

首版按下面这个假设实现：

> 冻结π0 teacher在某个future action位置的去噪终点越不稳定，RLT student在该位置越多地依据Q学习。

这里的信号来自冻结π0，真正更新的是MLP student。它不是student自身的概率不确定性，因此方法的准确名称是
`teacher-DVAC-guided RLT student Q weighting`。

## 2. 现有RLT与新增路径

成功RLT Stage 2当前主路径是：

```text
observation
  -> frozen pi0
       -> z_rl [B,2048]
       -> reference chunk [B,10,14]
  -> MLP student -> pi_student [B,10,14]
  -> twin-Q gives one value for the whole C10 chunk
  -> actor loss = -lambda_Q * Q(s,pi_student)
                  + lambda_BC * BC(pi_student, reference/executed action)
```

新增旁路是：

```text
the same frozen pi0 ODE, M=4
  -> endpoint previews z_i [B,4,50,14]
  -> V_L2 / V_L3 / V_L4 [B,50]
  -> training selects V_L3[:,0:10]
  -> global z-score -> w [B,10] in [0,2]

pi_student
  -> Q branch sees a forward-identical weighted-gradient view
  -> BC branch still sees the original pi_student
```

所以首版只改student actor的Q分支：

- `w_h=0`：第h格不从Q分支学习，但仍受BC约束；
- `w_h=1`：与原RLT完全相同；
- `w_h=2`：第h格从Q返回student的梯度贡献变为两倍；
- π0继续冻结，critic TD loss、reward、route、BC和Stage 1都不改。

实现形式是：

```python
pi_chunk = pi.reshape(batch_size, 10, 14)
pi_for_q = (
    pi_chunk.detach()
    + weight.detach().unsqueeze(-1)
      * (pi_chunk - pi_chunk.detach())
).reshape_as(pi)
```

forward时`pi_for_q == pi`，因此同一次forward的动作、Q值和loss数值不变；backward时
`Q -> action_h -> student`这一段按`w_h`缩放。default Q和CrossQ两条路径都使用`pi_for_q`，BC继续使用原`pi`。

## 3. 首版默认参数

| 项 | 默认值 | 含义 |
|---|---:|---|
| Stage | `stage2` | Stage 1不运行action denoising，不接DVAC |
| mode | `off / observe / apply` | 基础配置默认`off`，实验overlay用`apply` |
| teacher denoise | `M=4, flow_ode` | 复用原reference生成，不增加π0 forward |
| 保存的L | `[2,3,4]` | 离线分析可比较 |
| 训练selected L | `3` | 与GRPO-DVAC首轮一致 |
| active action dim | `14` | 不把π0内部padding维度混入方差 |
| teacher horizon | `H=50` | 完整信号保留用于分析 |
| student applied horizon | `C=10` | 只有student实际输出的前10格进入训练权重 |
| normalization | `frozen initial-replay global-z` | 使用干净raw DVAC，不做per-h residual |
| `log_v_eps` | `1e-12` | 先取`log(V_L3+eps)` |
| z clip | `[-2,2]` | 控制极端值 |
| strength | `0.5` | `w=1+0.5z`，所以范围为`[0,2]` |
| apply target | `student_q_only` | BC与critic更新不加权 |

“global”只统计真正参与student训练的`h=0..9`。完整H50仍保存，但不让未训练的`h=10..49`改变C10的
均值和标准差。

## 4. 为什么RLT不用GRPO的recent-5窗口

GRPO主要训练刚采集的on-policy rollout；RLT会从replay反复抽取旧transition。如果均值和标准差一直随最近
5个collection step移动，同一条旧transition在不同actor update里会得到不同权重。

首版采用更适合off-policy replay的简单做法：

1. 在成功formal配置原有的reference warmup期间收集raw `log(V_L3+eps)`；
2. 统计域为全部actor rank、全部warmup transition、每条的`h=0..9`；
3. 到原配置的`warmup_min_size=10000`首次满足时冻结`count/mean/std`；
4. 冻结前`w=1`；冻结后同一个raw V始终映射到同一个z和w；
5. baseline随RLT trainer state保存和恢复。

这不是新增一段预训练；它复用RLT本来就存在的reference collection warmup。raw V保存在replay，weight在
actor update时由冻结baseline派生。

## 5. 训练数据记录合同

### 5.1 每条collection/replay transition

建议用一个可选aux字段保存：

```text
teacher_dvac_v [3,50] float32   # 顺序L2,L3,L4
teacher_policy_version scalar
collection_step scalar
```

训练时派生而不重复存储：

```text
log_v_l3 [10]
z_clipped [10]
weight [10]
replay_age = current_policy_version - teacher_policy_version
```

三条H50方差每个transition约600 bytes。按历史57,410 replay rows估算约32.9 MiB；即使curr/next各保留一份
也约65.7 MiB，远小于RLT checkpoint与现有replay主体。

### 5.2 抽样raw trace，不塞进replay

规划中的完整collection-side trace是：

```text
z_endpoint [4,50,14]
x_chain    [5,50,14]
V_L2/L3/L4 [50]
ref_action [10,14]
student_action [10,14]
reset_id / episode / query / collection_step / route
```

默认建议每个collection cycle、每rank抽4条，总计8条；这用于复查endpoint公式和画图，不参与训练。

首版实际实现先保存actor-update replay样本中的`V_L2/L3/L4`、selected V、z、weight、`[10,14]`
student/reference action、collection version与route。`endpoint/x_chain`以及reset/episode/query对齐尚未进入首版writer；
它不影响训练或smoke，后续若要复核完整teacher去噪链，再在collection侧单独补充。

### 5.3 每个actor update记录

沿用RLT现有Q/BC/replay指标，再增加三组：

1. **信号与权重**：`V_L3`和`log V`的p05/median/mean/p95，baseline
   `count/mean/std`，weight的min/p05/median/mean/p95/max，命中0/2比例、weight ESS、top-20% weight mass、
   每个h的mean weight。
2. **RLT专属关系**：按reference-route/student-route、reward-positive、done/success分组的V和weight；
   `student-reference abs/MSE per-h`与DVAC的相关；twin-Q gap、TD-error绝对值与DVAC的相关；replay age。
3. **训练作用链**：现有`q1/q2/q_pi/q_data`、critic loss、BC loss、student-reference error、actor/critic
   pre-clip grad norm，以及当前`q_weight`和`bc_weight`。后两项必须记录，因为RLT本身会改变Q与BC的相对强度。

RLT没有GRPO advantage、PPO ratio、KL或PPO clip，因此不复制这些字段。对应的替代分析轴是route、reward、
TD error、Q分歧和replay age。

## 6. 最小代码修改清单

正式实现以AutoDL clean成功RLT source
`/root/autodl-tmp/RLinf_rlt_pi0_robotwin@2b8199d8ab2e7b110994fd3234bf7007196c3af9`为base，新建一个独立
worktree和一个branch，不编辑本地已有dirty `.rlt-impl-worktree`。

1. `rlinf/models/embodiment/openpi/openpi_action_model.py`
   - 在现有M4 ODE循环旁路收集endpoint；
   - 计算H50的L2/L3/L4；
   - `mode=off`时不分配、不返回DVAC字段。
2. `rlinf/algorithms/rlt/rollout.py`与`transition.py`
   - 把可选DVAC aux字段从query透传到compact transition/replay；
   - 不扩大原来的必填`RLT_OBS_KEYS`合同。
3. 新建`rlinf/algorithms/rlt/dvac_weighting.py`
   - population variance、冻结global-z baseline、`[0,2]`映射、Q-action straight-through helper与轻量writer。
4. `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py`
   - 全rank累计并冻结warmup baseline；
   - actor batch取L3前C10产生weight；
   - default/CrossQ两条Q路径接`pi_for_q`，BC接原`pi`；
   - baseline进入现有RLT trainer-state save/load；
   - 写训练指标。
5. 新增一个从成功`8env250`配置继承的opt-in YAML；除DVAC块、输出目录和后续实验步数外，不改成功RLT参数。

## 7. 当前训练、实现、测试和push的边界

2026-08-23 21:32 CST现场：当前global-z `[0,2]` GRPO已完整到Global Step 4，Step 5 rollout为13/16；
PGID=`820638`，六个核心worker alive；GPU0/1约`27.1/26.6 GiB`，cgroup约`166.0/240 GiB`，
`oom=0, oom_kill=0`。

新RLT worktree与当前GRPO worktree、branch、Python进程完全分开。编辑、commit和普通push不会改变当前训练已经
加载的代码，也不占GPU。项目import/test可能加载模型、Ray或模拟器，并与当前作业共享cgroup RAM，因此当前只做
实现和静态diff审阅；几项server测试留到当前作业结束或停止后。

截至2026-08-24，本轮已经完成原顺序中的1--5：

1. 已在隔离worktree完成主体；
2. 当前global-z训练运行期间没有启动RLT/Ray/环境/模型；
3. 已完成公式、global-z、Q-action前向恒等/反向倍率、optional transition往返与空组固定schema共5个
   CPU-only测试，并由独立
   代码审查复核off/async兼容、resume合同、shape、baseline冻结/恢复、default/CrossQ与BC/critic边界；
4. 已commit并push主体`275ab453...`和metric schema修复`513dbcb7...`；
5. 真实两卡smoke已自然exit0，训练、fixed-20 eval、DVAC trace、资源和完整g1 checkpoint均通过。

完整smoke结果见[32号文档](32_RLT_DVAC_REAL_SMOKE_RESULT_20260824.md)。用户随后选择新的单段预算：
**Stage 2 fresh启动并由同一进程直接跑到绝对cycle 480**。正式resolved合同、精确命令、输出目录、预算与
启动现场见[33号文档](33_RLT_DVAC_FRESH480_FORMAL_LAUNCH_20260824.md)。

## 8. 历史成功RLT的实际训练长度

这里需要把“成功RLT是480步”说准确：历史run确实以绝对`cycle=480`自然结束、`exit0`，但它不是一次
fresh 480-cycle训练，而是两段连续合同：

1. `8 env` fresh Stage 2先完成`cycle 1--250`，`250/250`自然结束；
2. 从完整`global_step_250` checkpoint严格resume，继续`cycle 251--480`，最终`480/480`自然结束。

合并后共有3,840个train episode、57,410条global replay row、215,055次critic update和107,528次
actor update；续训阶段10个fixed-20评估合计`178/200=89%`，最终点`17/20`。完整证据见
[历史480-cycle结果](../rlinf-robotwin-pi0-rltoken/17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md)。

因此当前teacher-DVAC overlay“继承成功`8env250`配置”描述的是算法、并发、replay、更新和评估参数合同，
不是强制复用历史两进程边界。用户已明确批准本次fresh单进程直接到480；它对齐历史的**总cycle预算**，但
cycle250没有进程重启，随机轨迹与内存高水位不会逐点相同，因此不称为历史bitwise复现。

## 9. 还需要用户决定什么

实现本身没有阻塞性歧义，首版默认已经按下列合同完成：

- 只做RLT Stage 2；
- teacher raw/global-z `V_L3`；
- 完整H50记录、前C10训练；
- `[0,2]`；
- 只缩放student的Q分支；
- 初始10,000条reference warmup冻结baseline；
- base配置默认`off`，实验overlay显式`apply`。

本次正式实验的运行决定已经冻结：fresh直接480；训练期仍沿历史合同每25 cycle做fixed-20 eval。后续若要
做最终模型间结论，再单独规划更大的同fixed-ID held-out评估；它不阻塞当前formal。

## 10. 当前正式训练状态

2026-08-24 18:52 CST正式run已启动，source=`a85b101b...`。19:03 CST只读复核时已完整到Global Step 2/480、
正在第3轮rollout；2 actor、2 rollout、2 env worker存活，replay正常增长，cgroup约44.2 GiB，memory event
与精确fatal匹配均为空。前两轮actor/critic updates为0是原有10,000-transition reference warmup的预期行为。

- [正式fresh480启动结果](33_RLT_DVAC_FRESH480_FORMAL_LAUNCH_20260824.md)
- [逐指令实施账](evidence/RLT_DVAC_FORMAL480_IMPLEMENTATION_AND_LAUNCH_LEDGER_20260824.md)
- [AutoDL内存锯齿3分钟说明](34_AUTODL_MEMORY_SAWTOOTH_AND_SHENZHEN_TRANSFER_20260824.md)：cgroup主约束、
  nested offload配套，训练并发合同不变
- [AutoDL内存机制技术附录](35_AUTODL_MEMORY_SAWTOOTH_TECHNICAL_APPENDIX_20260824.md)：源码与运行证据
