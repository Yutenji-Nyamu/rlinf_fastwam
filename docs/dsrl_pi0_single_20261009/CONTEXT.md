# DSRL π0 单卡 U / Norm：锁定上下文

2026-10-09。用户授权实现、服务器测试、GPU smoke、推送既有 Yutenji-Nyamu/rlinf_fastwam；仅深圳3物理6/7，临时让原候补RLT退让，完成后归还；不启动DSRL正式训练。

## 审计结论与历史版本

成功参照：深圳1 `formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2`。算法实现4b609178，轻量证据后HEAD 4ec52bd8（本次只读核同、工作树干净）。模型 `RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50`，不是Sidney π0.5。

历史是两卡数据并行：6/7各有actor、rollout、env，各2环境；不是推理/训练各占一卡。200轮、800采集回合中578成功，72.25%；learned阶段577/748=77.14%，最终周期评估8/12、最好12/12。单次高点不是稳定100%。历史Adam更新104120，单seed实验，不能作U/Norm收益结论。

| 项目 | 成功π0 | π0.5原C10 | π0.5 C20 | π0.5修复重启 | 本次π0单卡 |
|---|---|---|---|---|---|
| 模型 | 任务专用π0 | Sidney e49e2ab6 | 同左 | 同左 | 原π0 |
| H / C / ODE | 50/20/4 | 50/10/10 | 50/20/10 | 50/20/10 | 50/20/4 |
| train/eval环境 | 4/4 | 4/4 | 4/4 | 8/4 | 4/4 |
| 学习rank / global batch | 2/256 | 1/256 | 1/256 | 1/256 | 1/256 |
| micro / 梯度累积 | 64/2每rank | 256/1 | 256/1 | 256/1 | 256/1 |
| warmup / UTD | 500/20 | 500/20 | 500/20 | 2000/20 | 500/20 |
| replay | 2×12500 | 25000 | 25000 | 25000 | 25000 |
| 指令环境 | 旧reset | 旧reset | 旧reset | 已修复 | 原π0参照环境 |
| 真实动作 | delta joint适配 | absolute qpos | 同左 | 同左 | 原π0适配 |

共有：冻结VLA，只学32D噪声actor与10头Q；latent重复H50；learned actor tanh±1；SAC输入主图64×64+raw state14；actor LR1e-4，critic/entropy LR3e-4；γ=.999按C折算，target EMA τ=.005，target entropy=-16；成功chunk奖励0、否则-1，超时bootstrap。BF16 SAC与FP32 target shadow为共有条件。

## 为什么π0.5没学起来

C10曾同时改变控制、奖励尺度、warmup物理覆盖与每动作更新量；后续C20已经纠正。旧场景指令错配是真bug，但后续N8/warmup2000重启已采用固定vector SHA bc21d004，并通过真实reset。修复后Clean仍只有随机阶段9/208、learned阶段7/560；已计划112680更新，超过历史π0。故不能把原因简化成卡少、没更新、只有prompt错误或预算太少。

既有逐函数审计：critic/alpha/优化器/target/更新数主干一致，Clean没有进入U分支；strict load、三相机、state、动作转换、回放投影未发现新的明确错接。π0.5已成功更新且无OOM，单卡没有少算global batch。

最有依据的未决问题是底座变化后的噪声可控性与Q指导是否有效：Gaussian切换至重复32D的tanh actor时成功率退化；这个切换同时改变分布与参数，尚不能区分表达空间限制和学习退化。原生iid H50×32、通用prompt、逐动作执行、图像resize与RLinf存在差异；原生五例成功不证明DSRL完整入口等价。当前不为此再开π0.5长训练。

本次为了可解释地迁移成功π0，保留其旧环境路由；已知reset文字缺陷明确保留，不把新smoke当修复后基线。独立修复环境留作后续受控对照。本轮不同时改控制器、prompt、精度、噪声范围和采样预算。

## 方法合同与温度

U：原π0主ODE4不变，额外ODE2共用cast后行为latent和prefix cache；比较完整solve最终结果的前C20×14，逐动作mean(population std/RMS)。额外求解保存恢复RNG。

Norm：原ODE4的全部4步、最深3层各取post-residual/pre-final-norm hidden L2，再平均12个标量，保留动作轴，之后截C20；无额外求解。新标识norm_residual_t4_l3与旧π0.5 t5_l3严格区分。U同样用新标识ugrow_ode4_vs2。

历史采集信号写入回放，表示行为latent对应生成，不冒称当前sampled latent的因果credit。每chunk有效动作上s=mean(log(signal+1e-12))；全global batch做minmax，再w=exp(unit/τ)/mean(exp(unit/τ))；detach后乘完整actor项(alpha_entropy*logπ-Q)。critic TD、entropy温度目标、奖励和均匀回放不变。

这里只有外层chunk温度，先沿用τ=2.5；minmax后的理论最大/最小比exp(1/τ)≈1.49，τ1/2/3分别2.72/1.65/1.40。内层只有固定mean-log聚合，没有可分别分配的物理动作loss，故不另设内层τ。该τ也不同于target EMA τ=.005或SAC entropy alpha。

双trick独立可关，默认不改变旧实现/恢复合同：chunk dropout p=.2把抽到的chunk权重置1，不丢数据、不除1-p、不再次归一化；对成功与失败均适用。私有CPU RNG由seed42和持久化update_step决定。权重强度λ从R1的1线性降至R200的0，w=1+λ(w-1)；轮数跟随runner version，非SAC更新数，且与entropy alpha不同。200终点对应本次原200轮参照预算。所有新增开关及τ进入trainer恢复合同，拒绝不兼容恢复。

## 实施及验收边界

源分支codex/dsrl-pi0-single-u-n-20261009从已发布6c8756b6（U+Norm）建立；保留现存dirty工作树。只扩展π0信号标识、可选双trick、单卡实配与针对性检查。

单卡能保留总环境/global batch/更新密度，但双池等额抽样→单池均匀、seed分片及BF16归约顺序发生变化；不是逐位复现。提供Clean/U/Norm正式预算配置但不执行正式。

Smoke两路各1轮：仍N4、H50/C20/ODE4、200动作、B256/MB256；仅warmup改1、UTD改1使首轮真实更新最多40次；评估关闭，末轮保存。覆盖真实采集→信号回放→非均匀actor权重/梯度→断点→退出释放。退火终点和开关组合由CPU测试覆盖，不额外跑长smoke。

操作使用统一queue精确6/7 request及PID/UID/start/namespace，暂移6/7槽，4/5等待WM的槽保持；所有compute/graphics验证在指定卡。按最新完整RLT检查点恢复，不伪造中途保存。模型/身份/越界/非有限值/输出合同异常即停本次精确任务。命令与resolved/environment/request将在EXECUTION中落盘后才启动。

证据来源：工作区docs/methods/dsrl/{COMPREHENSIVE_REVIEW_20261007,LEARNING_PATH_COMPARISON_20261008,INSTRUCTION_AND_INTEGRATION_FOLLOWUP_20261007}.md；docs/methods/ugrow/DSRL_PI05_AUDIT_20261006.md；历史resolved与逐轮CSV；本轮pi0_20261009读回证据。

## 本轮直接回读的结束记录

SZ1/SZ3 TensorBoard原事件于本轮重新读取，按time/step完整轮次对齐：

| 路线 | 完整轮 | 采集成功/回合 | 计划更新 | 最后周期评估 |
|---|---:|---:|---:|---:|
| π0双卡 | 200 | 578/800 | 104120 | 8/12 |
| π0.5 Clean C10 | 83 | 11/332 | 121140 | 1/12 |
| π0.5 U C10 | 83 | 6/332 | 121900 | 0/12 |
| π0.5 Clean C20 | 200 | 24/800 | 146700 | 0/12 |
| π0.5 U C20 | 200 | 14/800 | 148880 | 0/12 |
| π0.5 Norm C20 | 118 | 11/472 | 83880 | 0/12 |
| π0.5 Clean N8/reset修复 | 96 | 16/768 | 112680 | 0/12 |
| π0.5 U N8/reset修复 | 95 | 16/760 | 111220 | 1/12 |
| π0.5 Norm N8/reset修复 | 67 | 14/536 | 66960 | 1/12 |

这些更新数来自日志累计；不把每个最终轮数称作已保存CP。不同底座/种子调度/样本量不是方法收益的配对比较。旧smoke数据不混入正式表。原事件读回原文保存history-sz1.json/history-sz3.json。
