# RLT DVAC new：随BC/Q课程反转方向的实现规划

2026-09-13。状态：**独立实现已推送，深圳93项CPU测试及GPU7跨切点smoke通过；18:01正式启动，18:04验收已进入采集。** 提交`2be9a6d542b0688f3092ea67a4f7b4c07944107e`，基于原new提交`5e5882e6`；保留scale1/scale2、Pure04与clean既有实现。以下保留设计依据，实际交付状态见[实现与启动验收](RLT_DVAC_DIRECTION_IMPLEMENTATION_20260913.md)。

本轮17:02新曲线：[MA15/20/25/30，一图一行及共同区间放大](E:/Codex/home/visualizations/2026/09/07/01a07be9-03a5-7562-a01a-ed379a0171f5/rlt-direction-distribution-20260913/rlt-large-windows.html)。同R373的MA30为clean93.75%、new1 90.83%、new2 95.42%；共同R150–373采集均值78.63/76.17/78.18%，后段R200–373为88.15/87.14/89.80%。曲线描述阶段差异，尚未检验反向方法。

## 1. 要检验什么

早期优先模仿高V位置；当原clean的BC/Q课程达到最终BC系数后，转而优先模仿低V位置。改变的是成功来源BC内部的分配方向，原BC/Q课程、teacher、Q目标、回放、采样量与更新次数均沿用目标对照。

V是冻结teacher单次去噪链末L=3次终点预测的波动；它尚不是经任务成败校准的teacher错误率。方法假设是：学生较弱时，这类位置值得补课；学生较强后，不稳定teacher目标可能成为约束。老师没有随训练变差，改变的是学生能力及其访问状态。

**已确定首版：两层同时反转、原课程结束为锚点、硬切、success_scale=1。** 原new scale1的600轮完整预算保留，从同一Stage1全新Stage2启动；仅反一层和平滑过渡已留参数，本次不启用。

## 2. 已核实的数据链与改动位置

| 位置 | 原行为 | 已实现内容 |
|---|---|---|
| `teacher_dvac_v`与成功标记入池 | 保存V，取L3和前10个监督动作 | 无需改缓存格式或重算teacher |
| `_prepare_global_batch` | 每个actor更新槽，完整512条query算权重，再切两个256微批 | 从同一`update_step`解析两层方向，传给纯权重函数 |
| `build_two_level_success_weights` | 成功query内层、成功query之间外层；失败权重1 | 各层中心化残差乘一个[-1,1]方向系数 |
| actor BC损失 | student与reference teacher动作的平方误差乘权重 | 不改目标、不改Q项 |
| checkpoint | 保存critic计数及方法配置契约 | 由恢复计数重建方向，契约记录启用的方向课程 |

源码：[V选取](C:/Users/86136/Documents/rl/worktrees/rlt-dvac-new-20260912/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py:192)、[完整batch接点](C:/Users/86136/Documents/rl/worktrees/rlt-dvac-new-20260912/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py:255)、[权重函数](C:/Users/86136/Documents/rl/worktrees/rlt-dvac-new-20260912/rlinf/algorithms/rlt/dvac_two_level.py:49)、[actor目标](C:/Users/86136/Documents/rl/worktrees/rlt-dvac-new-20260912/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py:578)。

### batch不会跨更新缓存旧方向

真实调用顺序为：

1. 读取本次完整回放batch。
2. 根据`update_step % critic_actor_ratio == 0`判断本槽是否更新actor。
3. 若是actor槽，准备完整batch权重；随后才切microbatch。
4. 先更新critic，再按相同计数更新actor；两个微批用同一组已准备权重。
5. 本槽完成后，`update_step += 1`。

`rlt_dvac_new_weights`只放在本次batch浅拷贝中，未写回回放；下一actor槽重新抽样并重新计算。故课程切点不会因旧权重缓存延迟一轮，只受actor每2个critic槽更新一次的离散频率影响。

源码：[采样、准备与切批](C:/Users/86136/Documents/rl/worktrees/rlt-dvac-new-20260912/rlinf/workers/actor/fsdp_sac_policy_worker.py:558)、[actor更新条件](C:/Users/86136/Documents/rl/worktrees/rlt-dvac-new-20260912/rlinf/workers/actor/fsdp_sac_policy_worker.py:614)、[计数递增](C:/Users/86136/Documents/rl/worktrees/rlt-dvac-new-20260912/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py:1483)。

## 3. 反转的数学定义

令成功query的动作数为C=10。两支都从原始`log(V+eps)`出发，沿用现有MinMax范围、常数域保护和成功来源选择：

$$
u_{b,a}=\operatorname{MinMax}_{a}(\log(V_{b,a}+\epsilon))-\operatorname{mean}_{a}[\operatorname{MinMax}_{a}(\log(V_{b,a}+\epsilon))],
$$

$$
v_b=\operatorname{MinMax}_{b\in S}(\operatorname{mean}_{a}\log(V_{b,a}+\epsilon))-\operatorname{mean}_{b\in S}[\operatorname{MinMax}_{b\in S}(\operatorname{mean}_{a}\log(V_{b,a}+\epsilon))].
$$

$$
W_{b,a}=s\,[1+d_L(t)\alpha_Lu_{b,a}]\,[1+d_G(t)\alpha_Gv_b],\qquad b\in S.
$$

失败query继续`W=1`。`s=success_scale`，`alpha_local/chunk`仍各在[0,1]；方向系数`d_L/d_G`在[-1,1]。

| 方向 | 分配语义 |
|---|---|
| +1 | 保持当前高V优先 |
| 0 | 指定层回到均权，另一层仍可重分配 |
| −1 | 指定层低V优先 |

内层每个query均重1，外层成功query均重1，因此成功来源最终均重仍为`s`。重复抽中的query按实际batch出现次数计入，不改采样方式。常数信号、单个成功query的外层自然回到1。

单层例子：`[0.5,1,1.5] → [1.5,1,0.5]`。这是正权模仿中的份额交换，不是负BC或让student远离teacher。两层反转是各自比较域内的相对降权；不是保证任意绝对V较大的动作在全batch中权重都更小。

纯函数可新增 `direction_local=1.0, direction_chunk=1.0`，只在现有中心化残差前相乘。旧调用不传参数时结果保持；不把alpha改成负值，不改现有alpha语义。方向±1、alpha≤1时，有限域的因子仍为正；保留现有最终float32有限/正值检查。

**不采用**`1/W`或`2−最终W`：前者会改变均值、放大近零值；后者对两层乘积可能产生负值。逐层反向后，两个偏好的乘积也不等于最终权重的线性镜像。

## 4. 切点必须和原课程使用同一个计数

当前`_actor_objective_weights()`用累计critic更新计数`t=self.update_step`。原配置是前20k保持BC/Q=7/0.05，再50k线性到2.5/0.45。源码中的进度是：

$$
p(t)=\operatorname{clip}((t-20000+1)/50000,0,1),\quad t\ge20000.
$$

所以理论进度在`t=69999`首次为1；actor只在偶数计数槽更新，**第一份实际使用最终课程系数的actor batch是`t=70000`**。该槽先做一次critic，actor和DVAC仍读取同一个未递增计数；本槽结束才加1。历史clean在R192轮内跨点，R193起整轮处于最终系数，但不要把R193写进方法。

实现宜提取一个很小的纯课程进度解析函数，供BC/Q原公式和方向解析共同读取，保持原数值次序；或者使用同一份解析出的课程进度作为方向条件。不能另写一个近似轮次判断。对一般配置，课程结束阈值是`warmup_updates + max(ramp_updates - 1, 0)`，但仍以首个实际actor槽生效。

[真实BC/Q公式](C:/Users/86136/Documents/rl/worktrees/rlt-dvac-new-20260912/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py:352)。若以后整体半量使课程变10k+25k，则首个实际切换actor槽相应约35k；自动跟随配置，无需另传35k或70k。

### 硬切与线性过渡

- `transition_updates=0`：首次原课程进度达到1的actor槽，指定层由+1变−1。最直接对应“到这个点后反过来”。
- `transition_updates=K>0`：从课程结束点开始，方向按critic计数经过K次更新由+1线性到−1；中点0表示该层暂时均权。不要从20k就开始反转，这会改变用户指定的时序。

推荐先把硬切作为清晰的方向消融。平滑不是数学必需项；它减少目标的瞬时变化，但增加一个过渡长度。不得因窗口平滑把真实切点误报告为整个R193。

## 5. 建议配置与兼容范围

以下字段**已实现并纳入新独立recipe**：

```yaml
algorithm:
  rlt_dvac:
    direction_schedule:
      enable: true
      anchor: actor_weight_schedule_end
      scope: both             # both / local / chunk
      transition_updates: 0   # 0硬切；正整数为切点后线性过渡
```

不额外增加起点轮次、成功率触发、随机开关或success_scale课程。旧配置不含此段时，解析为disabled并保持两层方向+1；不改旧YAML，**也不将默认字段写回旧配置对象**。仅`two_level_batch`支持此课程，原Pure04路径保持原样。

新配置单独继承现有new基线，方法字段之外逐项保持目标对照实际配置。若anchor选择原课程结束而BC/Q课程未启用，初始化会报配置错误；不会默默猜70k。本轮已授权GPU7从头训练，600轮、8env、B512/micro256、UTD5和scale1对照一致。

### scope和success_scale是独立选择

| 设置 | 结果 |
|---|---|
| both | 内部位置和整个成功chunk均转向低V，最贴近此次提议 |
| local | 只换chunk内部偏好；外层仍可能整体加强高Vchunk |
| chunk | 只换整段偏好；段内仍优先高V动作 |
| success_scale=1固定 | 只改变成功监督分配，不增加其平均系数 |
| success_scale=2固定 | 成功监督仍平均翻倍，即使某位置反转降权，也可能高于clean |

例如反向两层因子0.8和0.9，scale2最终仍为1.44。若希望同时把总模仿强度从2降1，那是另一项改动，应另列，不随方向切点改变。本次直接对照原scale1，固定`success_scale=1`。

## 6. 恢复状态：无需新计数，但必须登记课程身份

现有checkpoint已经保存并恢复`update_step`；方向由计数与配置确定，不需要再保存“已反转”布尔量，否则两种状态可能不一致。终止后恢复跨过切点，直接得到正确方向。

当前恢复契约对整个`rlt_dvac`字典及哈希逐字匹配。两个实际注意点：

1. 给历史配置自动补一段`enable:false`，即便算出的权重相同，也会改变字典哈希。因此默认只用`get`读取，不写回；旧配置、旧checkpoint继续用旧契约。
2. 新启用方向课程时，除新方法字段外，契约还应记录用于推导切点的BC/Q课程字段（enable、warmup、ramp与解析后的结束计数）。现有默认契约没有完整记录这组课程字段；仅记录字符串`actor_weight_schedule_end`无法阻止恢复时悄悄换了锚点。

仅在新方向课程启用时附加这一确定性契约片段，旧分支不受影响。新方法同配置恢复可用；拿现存new checkpoint直接改enable属于改变方法后分叉，现有严格检查会拒绝。若将来要做这种分叉，应单独定义一次显式迁移和来源记录，不能删除匹配检查来“兼容”。首版从相同Stage1全新Stage2启动最简单。

源码：[契约与方法字典](C:/Users/86136/Documents/rl/worktrees/rlt-dvac-new-20260912/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py:1070)、[契约校验](C:/Users/86136/Documents/rl/worktrees/rlt-dvac-new-20260912/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py:1141)、[恢复计数](C:/Users/86136/Documents/rl/worktrees/rlt-dvac-new-20260912/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py:1257)。

## 7. 实现工作与少量必要验证

已使用独立`codex/sz-rlt-dvac-direction-20260913`分支。改动集中于权重函数、纯课程解析、RLT worker与恢复契约、新配置和相关单元测试；未改teacher、仿真、replay格式或通用SAC的采样/优化顺序。

必要测试聚焦真实边界，不额外建大量镜像测试：

1. **旧行为相同**：默认不启用与原方向+1权重/BC损失一致，原Pure04/off/observe路径保持；旧配置恢复契约不变。
2. **数学语义**：两层反向、只反一层、方向0；成功均重s、失败1、常数域、单成功、无成功；全程正权且不回传V梯度。
3. **真实计数**：现有真实更新循环测试上覆盖69998/69999/70000附近actor槽；BC/Q末值与方向同槽对应，两个microbatch不各算外层；切点后每次batch重新准备。
4. **恢复与过渡**：同配置在切点前后恢复结果一致；改变课程锚点或方向配置必须被契约识别；若支持K>0，验证起点、中点、终点。

已扩展两级数学测试、完整batch/真实循环/恢复测试，并新增纯方向课程测试；深圳服务器93项全部通过。GPU learner使用512条真实回放跨69998/70000，2次actor、4次critic更新通过。测试与正式启动回执见[实现验收页](RLT_DVAC_DIRECTION_IMPLEMENTATION_20260913.md)。

增加的关键日志只需：本actor槽计数、课程锚点、两层实际direction、是否已经进入反转期。沿用当前inner/outer/std/ESS和成功比例指标。轮级日志会对很多更新取平均，因此跨点轮的direction可能在−1与1之间；这代表本轮混合了两个阶段，不等于实际用了中间权重。

## 8. 当前结论与待正式实验选择

代码接入是局部、清楚的；主要工程细节是**同计数切点和恢复契约**，不是重建信号链。建议配置保留scope与过渡长度，但第一个实验固定both、硬切、scale不变即可得到解释清楚的对照。

本轮运行选择已确定：直接对照scale1，保留完整预算，从原Stage1全新Stage2启动。93项CPU与GPU smoke均通过，正式实验已在GPU7进入初始teacher采集；后续状态集中维护于[验收页](RLT_DVAC_DIRECTION_IMPLEMENTATION_20260913.md)。

支持这条思路的前人证据主要是后期放松或选择性模仿，尚不证明DVAC应反号：[Q-filter](https://arxiv.org/pdf/1709.10089)、[Adaptive BC](https://github.com/zhaoyi11/adaptive_bc/blob/main/main.py#L196-L206)。前一轮的信号定义、文献界限和预算诊断见[综合讨论](C:/Users/86136/Documents/rl/docs/rlinf-shenzhen-rlt-dvac-pure-port/RLT_DVAC_DIRECTION_AND_BUDGET_DISCUSSION_20260913.md)及[方向讨论证据](C:/Users/86136/Documents/rl/local_scripts/experiment_followup_20260913/RLT_REVERSE_DISCUSSION.md)。
