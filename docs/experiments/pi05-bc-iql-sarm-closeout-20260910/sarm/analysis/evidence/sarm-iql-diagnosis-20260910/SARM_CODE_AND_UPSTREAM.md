# SARM / RynnValue RA-BC：代码、信号与官方方法复核

2026-09-10。仅本地源码、已下载现场元数据及官方网页审计；未运行模型、训练、服务器命令或改生产代码。标签：**事实**=源码/回执/实际数据；**推导**=由公式得到；**假设**=待验证原因；**建议**=尚未执行的消融。

## 结论先行

**事实：当前更准确的结论是“SARM未显示收益”，不是“训练坏了”。** R100正常完成500次更新；同R5–100的20个fixed32检查点均值42.34375%，clean43.125%，差−0.78125pp。R100为13/32 vs14/32。单seed、反复使用同32个评估种子，不能把检查点平均当独立重复试验。[数值对照](/C:/Users/86136/Documents/rl/docs/server-admin/evidence/experiment-refresh-20260910-current/analysis.json)

**本轮未发现能解释效果的已证数值实现错误。** 已对上符号、评分边界、总体矩、1024-query分母和micro32梯度累积；最终552条数据可独立重建统计。更强的实际线索是：**权重明显偏向第一chunk、削弱第四chunk；而该偏好是否对应动作质量，现有信号验证没有回答。** 这套实验应称“冻结RynnValue + SARM加权规则的在线适配”，不是完整复现SARM的奖励模型和训练条件。

## 1. 原方法与本项目：复制了什么，替换了什么

| 环节 | 官方依据 | 本项目事实 / 影响 |
|---|---|---|
| 信号模型 | SARM把任务阶段和阶段内进展组合成0–1进度；先针对任务训练奖励模型 | 替换成冻结RynnValue-8B通用预计剩余秒，不含本任务stage监督；语义替换大于单位替换 |
| 训练数据 | 原RA-BC用200小时、质量不一的遥操作示范；并比较20小时长度筛选BC | 我们仅使用在线成功episode，原有成功门槛已去掉整条失败；不应把原论文描述成“必然含失败episode” |
| 干预粒度 | 每个chunk一个进度差、一个非负权重、加权平均BC loss | 相同；没有逐动作DVAC、critic或优势指数变换 |
| 统计 | 论文描述运行矩；固定版LeRobot对离线全数据预算总体均值/标准差 | 全run新入池unique query累计一次；旧样本的r固定，权重随整个成功池矩变化 |
| 采样边界 | LeRobot原始frame index加chunk_size，末端截到本episode最后帧 | 实际执行前后query边界；每个原始50×14命令整体得到一个权重 |
| 冻结评分 | 官方RynnValue支持absolute/relative，输出原生秒；官方adapter提供因果前缀末槽路径 | 只用主视角、4帧历史边界、absolute末槽；不融合relative，不重新归一成0–1 |

官方来源：[SARM论文§3–4](https://arxiv.org/html/2509.25358v4#S3.SS2)、[LeRobot SARM使用文档](https://huggingface.co/docs/lerobot/en/sarm)、[RynnValue官方仓库](https://github.com/alibaba-damo-academy/RynnValue)。论文§4.3控制加权框架、仅替换reward model时，ReWiND明显弱于SARM；这是“复制权重函数还不够”的直接先例。[同论文§4.3](https://arxiv.org/html/2509.25358v4#S4.SS3)

**限定：** “成功池已有过滤，因此收益空间可能小”是假设，不等于成功轨迹每段都最优。原方法依赖可靠的段质量排序；通用RynnValue基准成绩不能替代本任务、同阶段动作的排序检验。官方文档也明确指出，数据本来较一致、质量较高时，RA-BC可能没有明显收益；κ=.01针对原任务调过，建议依据自己数据的均值/标准差定标。[官方κ和适用条件](https://huggingface.co/docs/lerobot/en/sarm#tuning-ra-bc-kappa)

## 2. 本地数据链与低层审计

本地生产packet的8个核心文件SHA256与已通过服务器测试的`tests-receipt.json`逐一一致；formal配置diff只含RA-BC组件和路径/卡号迁移，BC4/U5、1024/micro32、LR2.5e−5、Adam、clip1、种子、初始化、成功长度过滤off均沿clean。[正式diff](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-online-bc/evidence/bc-sarm-formal-20260909/prepared-formal/config_diff.json)

| 检查点 | 代码事实与精确接点 | 结论 |
|---|---|---|
| 原始RGB/任务 | collector在step前clone主RGB与原任务文案；按env取reset前真实final_obs；相邻post/pre完全一致才接收。[data:60](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/data/online_bc.py:60)、[data:110](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/data/online_bc.py:110)、[data:147](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/data/online_bc.py:147) | 未见reset图误接；真实终图没有保存在最终replay，不能凭当前数值独立重新看终图 |
| success-only | success为true才输出完整episode，终止后不继续收；原动作命令flatten存入，mask全1。[data:127](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/data/online_bc.py:127)、[data:171](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/data/online_bc.py:171) | 没有把失败动作混入SARM训练 |
| 评分输出 | 锁定HF8738c5e4；模型已把分布解码回秒，读取[1,4]的最后槽，未把4帧平均；不再额外exp/log。[scorer:144](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/utils/rynnvalue_scorer.py:144)、[scorer:290](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/utils/rynnvalue_scorer.py:290) | 符合锁定模型真实维度；与README泛化示例的mean写法有意区别 |
| 差值与缓存 | N个chunk需N+1个边界值，r=v_before−v_after；校验有限性、0–512范围、差值符号、模型/请求哈希。[client:62](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/utils/rynnvalue_client.py:62)、[client:96](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/utils/rynnvalue_client.py:96) | 未见错位或反号；请求失败显式失败，不静默编造分数 |
| μ/σ | float64 Welford合并，新unique入池时更新；包含成功内部负r。存原始mean/M2，只在读取时μ≥0、σ≥ε。[algorithm:64](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/algorithms/online_bc_rabc.py:64)、[actor:113](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/workers/actor/fsdp_online_bc_policy_worker.py:113) | U5重复采样不会重复更新统计；无最近五轮或每micro统计 |
| SARM映射 | soft=clip((r−(μ−2σ))/(4σ+ε),0,1)；r<0置0，r>κ置1；等于0/κ仍走soft。[algorithm:119](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/algorithms/online_bc_rabc.py:119) | 和官方公式一致，负进展只是停止模仿，并非反向优化动作 |
| 分母和mask | 先算g=1024w/(Σw+ε)，再拆micro32；每query先按50×14 mask平均FM误差，乘g后求micro均值，累积时再除32。[actor:156](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/workers/actor/fsdp_online_bc_policy_worker.py:156)、[loss:15](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/data/online_bc.py:15)、[accumulate:482](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/workers/actor/fsdp_dagger_policy_worker.py:482) | 等价完整batch weighted mean；没有每micro重新归一、未漏乘/多乘 |
| 全零保护 | 全1024零权时在Adam前返回skip，scheduler与update_step不前进。[algorithm:128](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/algorithms/online_bc_rabc.py:128)、[worker:483](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/workers/actor/fsdp_dagger_policy_worker.py:483)、[counter:668](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/workers/actor/fsdp_dagger_policy_worker.py:668) | R100实际500步，无缺预算证据 |

**mask适配的未决边界：** 50个原始命令经TOPP插值后，真实成功可能发生在执行中途。现有mask表达“提交的命令标签”，不是逐动作已执行证明。末端进度差权重作用整个50动作chunk；这是继承clean标签合同的粗归因，而非SARM新增错mask。若要改变标签，应让clean与方法组同时换，并保留真正命令执行映射，不能拿模拟步数直接当50动作的前缀长度。

**已有反证范围：** 55项数值/数据/缓存测试 + 4项actor测试通过；含单位缩放、精确κ边界、zero skip、完整batch与micro梯度等价。真实smoke10次Adam、两个1024张量经独立NumPy复算。它们证明“代码按规则工作”，不证明RynnValue知道哪些动作更有用。[55项回执](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-online-bc/evidence/rynnvalue-rabc-implementation-20260909/tests-receipt.json)、[4项输出](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-online-bc/evidence/rynnvalue-rabc-implementation-20260909/source-finalize.txt:2)、[smoke](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-online-bc/evidence/rynnvalue-rabc-implementation-20260909/smoke-verification.json)

## 3. κ=2的真实含义：不是单位换算错误，也不是官方最优值

**事实：** 当初仅用3条历史成功片段、6个delta（0.543–3.497预测秒）检查full/soft两分支，κ2使3个full、3个soft。随后用户选择保持2并全程冻结。没有依据任务fixed评估调κ。这是已授权可复现起点，统计依据偏少。[正式参数讨论](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-online-bc/BC_RYNNVALUE_RABC_FORMAL_DISCUSSION_20260909.md)

**推导：** 若仅做全局单位转换r′=cr，并同时令μ′=cμ、σ′=cσ、κ′=cκ、ε′=cε，则权重完全不变。单纯把秒除以一个常数不产生新信息。若只换r却不换κ，才改变了方法；若每episode用不同分母，则是在改变跨episode偏好，已经不是全局单位转换。本项目用秒本身不是bug；关键是预测秒的差是否代表该动作的学习价值，以及κ2与这批数据是否合适。

**推导：** 因v_after≥0，r≤v_before。v_before≤2的chunk不能触发r>2的硬满权；soft仍可能自行达到1。R100 soft上界μ+2σ=3.794秒，高于κ2，因此本池170个满权全部来自硬覆盖。将κ附近的微小分数差放大成soft→1的跳变，是SARM规则本身，在噪声较大时可能加重误排序。

## 4. R100真实池：第一段偏重、第四段偏轻已得到数据支持

基于新下载`replay-metadata.json`中175条成功episode、552条unique query；148条长度3，27条长度4。独立stdlib复算每个边界差、episode连续性及count/mean/M2均吻合：μ=1.564636秒，σ=1.114469秒。raw均重0.57709，零权31/552=5.62%，满权170/552=30.80%，按unique池计算ESS428.94/552；30/175条成功episode至少有一段零权，无整条全零。[可复算JSON](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-online-bc/evidence/sarm-iql-diagnosis-20260910/sarm-weight-analysis.json)

| query序号（从1计） | n | 平均r/预测秒 | raw平均权重 | 零权 | 满权 |
|---|---:|---:|---:|---:|---:|
| 第一chunk | 175 | 2.120 | 0.801 | 10.86% | 73.71% |
| 第二chunk | 175 | 1.465 | 0.518 | 0% | 13.71% |
| 第三chunk | 175 | 1.343 | 0.487 | 0% | 9.71% |
| 第四chunk | 27 | 0.049 | 0.098 | 44.44% | 0% |

只看同样长度3的148条，第一/二/三段raw均重仍为0.800/0.535/0.480，排除了“仅因混入第四段”这一解释。终止chunk raw均重0.421，非终止0.649；65.71%终止chunk的before≤2，不能触发硬满权。真实成功终图仍预测平均剩余0.618秒（中位0.567秒）。

**事实边界：** 这是命令序号偏好，未看这些视频就不能把第一段全部叫“接近”、第三段全部叫“接触/放下”。低权第四段可能是低效补动作，也可能是必要的最终成功动作；当前数值单独区分不了。低平均权重也不等于同等比例的实际梯度贡献，仍取决于FM误差、梯度方向和Adam。

**额外推导：** 每条轨迹Σr=v_initial−v_terminal；当两条的端点值相近，长轨迹每chunk平均r天然较低。这里长度4平均r1.243、raw权重0.462，长度3为1.643、0.605。是否有效过滤慢动作，要由真实行为/任务阶段证明，不能仅凭该差得结论。

**日志口径修正：** `train/rabc/*`的batch权重诊断来自每轮**最后一次1024采样**，不是五次更新均值，因为每次prepare覆写`self.rabc_metrics`，轮末直接union返回。[actor:168](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/workers/actor/fsdp_online_bc_policy_worker.py:168)、[actor:223](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/workers/actor/fsdp_online_bc_policy_worker.py:223)。上表使用552条unique query固定R100矩，production则每次有放回抽1024后独立归一，两者不能混称。R100 raw均重在官方0.3–0.8诊断范围内仅说明没有退化成全1，不证明有效。

## 5. 信号质量：已有支持、尚缺的验证

**已有支持：** 官方demo能输出原生秒；3条正常成功片段的6个delta全正，倒序2个全负、冻结输入相同输出，说明粗方向/确定性成立。真实smoke另有一条最终成功轨迹首段r=−1.075；R100同样有19条成功轨迹首段零权。不能只凭“整条成功”断言这些首段都应高权，也不能只凭负r断言它们有害。

**假设A：进展大小偏好与动作质量不同。** 快速大幅推进可能更容易，也可能已有充分学习；精细收尾的剩余时间差小，却可能决定最后是否成功。SARM函数不自动纠正这一点，且本任务缺少原SARM语义阶段定标。

**假设B：历史前缀影响相邻差值。** 4帧索引依次为[0,0,0,0]、[0,0,0,1]、[0,0,1,2]、[0,1,2,3]、[0,1,2,4]。因此不同边界既改变最后图，也改变历史/重复情况。官方adapter确实使用linspace因果前缀；这不是偏离该入口的索引bug。[本地prefix:79](/C:/Users/86136/Documents/rl/local_scripts/bc_rynnvalue_rabc_packet/src/rlinf/utils/rynnvalue_scorer.py:79)、[官方adapter](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-online-bc/evidence/rynnvalue-rabc-plan-20260909/upstream_snapshot/robometer/robometer/evals/baselines/rynnvalue.py:348)

需同时保留反证：我们已开官方`pred_slot_isolated_eager`，没有遗漏这个重要组件。其代码屏蔽跨预测slot的key，仍保留正常因果语言/视觉上下文，不能由“isolation”推出输入历史完全无影响。[锁定attention代码:45](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-online-bc/evidence/rynnvalue-rabc-plan-20260909/upstream_snapshot/RynnValue-8B/attention_impl.py:45)。真正检查应固定同一末图和任务、只换历史前缀，观察差值/排序是否稳定，而不是再做全视频倒放便宣布标定充分。

**假设C：跨场景/指令尺度。** r会消去同一episode内恒定的value偏置，却不会消去场景相关增益、随动作变化的误差或必要准备动作造成的短期value上升。跨全成功池μ/σ只调整整体尺度；不会把不同指令、不同阶段的预测自动变成可比较的质量。直接改成每episode MinMax还会把微小噪声撑满，不应作为默认修复。

**官方解码没有缺一步：** 锁定`value_tokenizer.py`先对bin logits softmax，在symlog空间求期望后inverse_transform回秒；本地读取pred_value，未误拿bin下标或symlog值当秒。[锁定解码:263](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-online-bc/evidence/rynnvalue-rabc-plan-20260909/upstream_snapshot/RynnValue-8B/value_tokenizer.py:263)、[官方源码URL](https://github.com/alibaba-damo-academy/RynnValue/blob/10e0d333f5f3811d0d130587e50f1faf48da49e5/rynn_value/value_tokenizer.py#L263)。absolute/relative融合是可选信号消融，不是当前必修缺漏；使用relative时必须确认相邻采样帧就是对应执行chunk，第四边界的linspace会跳过第三边界，不能直接取最后relative当单chunk分数。

## 6. 优先做的五个有界对照

| 顺序 | 建议 | 想分辨什么；固定什么 |
|---|---|---|
| 1 | 用现成池/回放给少量同阶段、同初始场景片段做盲排；另检查同末图不同前缀 | 核查r是否知道同阶段哪个动作更值得学；先不训练，不用fixed评估挑信号 |
| 2 | `g_mix=(1−λ)+λg_sarm`，首个λ=0.5，其他κ2/μσ/池/预算不变 | 保留每段至少0.5的clean监督，单独测试强重权/硬删是否伤害成功链；这是新消融，非声称原SARM默认 |
| 3 | 单独κ=1与κ2比较，保留负r置0和所有其他参数 | 当前κ2硬覆盖偏向第一段；κ1使前三段权重接近，但不修复第四段/负r信号。只调κ不能证明信号质量 |
| 4 | 在相同池做位置对照：只用query序号平均权重，或在同序号内打乱r | 若原权重不胜只看序号，说明主要学到阶段/长度偏好；如要去偏，后续再单独实施位置内比较，勿一次叠加 |
| 5 | 小批比较更密因果帧、paired relative与原absolute差；保留每个边界轻量图用于核验 | 寻找稳定且真实的动作进展信号；变信号之前校验任务文案/左右臂/接触代理。未核准相邻帧与relative符号前不接正式训练 |

用R100同一552条池、固定μσ离线重算权重，能预知“分配怎么变”，不能预知训练胜负：

| 仅改权重（未训练） | 池ESS / 552 | 第一/二/三/四chunk平均归一系数 | 零权 |
|---|---:|---|---:|
| 当前κ2 | 428.94 | 1.388 / 0.897 / 0.843 / 0.170 | 5.62% |
| κ1 | 470.79 | 1.038 / 1.050 / 1.047 / 0.121 | 5.62% |
| κ3 | 454.23 | 1.308 / 0.936 / 0.881 / 0.192 | 5.62% |
| 保留负侧0、关闭硬满权覆盖 | 461.69 | 1.281 / 0.948 / 0.894 / 0.195 | 5.62% |
| κ2＋50% clean混合 | 515.06 | 1.194 / 0.948 / 0.922 / 0.585 | 0% |

均重约1只控制loss整体系数，不保证梯度范数或更新方向不变。以上一次只比较一项；没有证据支持把κ、归一域、数据筛选、信号模型同时更换。

## 来源与复算身份

- 本地代码：`local_scripts/bc_rynnvalue_rabc_packet/src`，8项源码hash与服务器通过回执一致；本轮未重新跑项目测试。
- SARM源码精确行号使用已下载锁定LeRobot`3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e`：[global统计:143](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-online-bc/evidence/source-audit-20260904/huggingface__lerobot/source/src/lerobot/rewards/sarm/rabc.py:143)、[权重/归一:214](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-online-bc/evidence/source-audit-20260904/huggingface__lerobot/source/src/lerobot/rewards/sarm/rabc.py:214)。本轮官方固定网页访问失败，改读可访问的[官方main](https://github.com/huggingface/lerobot/blob/main/src/lerobot/rewards/sarm/rabc.py)，核心公式/global域一致；没有声称main SHA已固定。
- RynnValue固定GitHub`10e0d333f5f3811d0d130587e50f1faf48da49e5`、HF8B`8738c5e4ce4418ea0266e9fbeffae0eb9bbb230e`；[原来源锁](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-online-bc/evidence/rynnvalue-rabc-plan-20260909/SOURCE_LOCK.json)。官方adapter在线已重读，模型解码/attention按本地锁定实际文件核对。
- 552条数据来源与sha见[sarm-weight-analysis.json](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-online-bc/evidence/sarm-iql-diagnosis-20260910/sarm-weight-analysis.json)；复算器[analyze_sarm_replay_20260910.py](/C:/Users/86136/Documents/rl/local_scripts/analyze_sarm_replay_20260910.py)只用stdlib和本地JSON，未加载Torch、RGB、checkpoint或模型。
