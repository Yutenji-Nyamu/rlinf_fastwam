# 2026-11-08 配方回归的来源与边界

| 选择 | 一手依据 | 本轮落地 |
| --- | --- | --- |
| RLT 恢复成功配方 | 本机旧 click_bell Clean CP625：20/20；`successful-rlt-config.json` 是实际 resolved.yaml 快照，非按论文猜参数 | C10、B512/micro256、20k预采集、30k接管前critic更新、BC 20k强约束+50k渐变、UTD5、actor每2次critic更新一次 |
| RLT 保留参考策略与渐进接管 | [RLT论文](https://arxiv.org/html/2604.23073v1)；本项目已跑通的 `fsdp_rlt_ac_policy_worker.py` | 复用现有采集、模仿与延迟接管实现；30k不是纯Q-only，期间小actor也按原比例学习 |
| DSRL UTD20→5 | [DSRL作者调参建议](https://github.com/ajwagen/dsrl#tips-for-hyperparameter-tuning)明确UTD与噪声幅度是迁移关键；[高UTD价值发散研究](https://arxiv.org/html/2403.05996v2)解释反复bootstrap可能放大错误 | 结合本机Q偏高、噪声贴边证据，下调每条新增经验对应的Q更新次数；5是本轮有依据的试验值，不是论文证明的本任务最优值 |
| 保留DSRL Q-only2000 | [ResFiT训练代码](https://github.com/amazon-far/residual-offpolicy-rl/blob/main/resfit/rl_finetuning/scripts/train_residual_td3.py)有critic-only预热；本机此前已实现并验收 | 本轮不再加码，避免在有限初始经验上持续放大Q偏差；2000不是照抄作者最优值 |
| 暂不改变噪声相关性 | 本项目5组/10组零训练评估已损害当前π0.5基座；[DSRL π0附录](https://arxiv.org/html/2506.15799v2#A3.SS5)的重复噪声属于另一底座/任务设置 | 保留1600维独立噪声、±2、FP32小网络与优化器；冻结大π0.5仍BF16 |
| 后续候选：OFN | [Dissecting Deep RL with High Update Ratios](https://arxiv.org/html/2403.05996v2)对Q输出前特征做单位范数约束 | 本轮未接入；如果恢复旧配方/低UTD后Q仍漂移，再考虑；不能将现有LayerNorm误说成缺失 |
| 后续候选：Q尺度归一化 | [TD3+BC作者代码](https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py)按平均绝对Q缩放策略项 | 本轮未接入；用于避免Q变大淹没BC，不能冒充已验证收益 |
| 新检索相关线索：DAWN | [作者仓库](https://github.com/Guozheng-Ma/DAWN)：先用基座收集数据，再联合更新；critic LayerNorm | 支持保持基座数据锚定，但它是动作残差RL，不直接证明噪声空间DSRL有效；本机RLT已具备预采集与LayerNorm，不重复加同名组件 |

DSRL加速沿用[首批实现与来源](../dsrl_fp32_accel/README.md)：指标延后回传、fused Adam、向量化Q；RT2另开Q编译，Dojo另开foreach目标更新。第二批未采用候选另见 `tools/dsrl_accel_v2/THIRD_PARTY.md`。RLT此前短测无稳定加速收益，继续保留其关闭配置。本轮不新增加速公式。学习变化与部署路径变化分别留在 `config-diff.json`；底层训练源码不变的哈希见 `source-reuse.json`。

预算：DSRL每路16×200=3200局；RLT每路16×313=5008局，对齐旧成功实验约5000局。RT2物理时限200、Dojo700；Dojo继续γ=.999，RT2恢复旧成功γ=.99。RLT每轮更新上限按16环境/C10/物理时限扩为RT2 3200、Dojo11200，保留旧配置最大新数据更新量的2倍余量，避免硬上限把UTD5截低。上限不是每轮固定更新数。

所有方法从新小网络/新回放开始，避免继承已偏移的Q；RLT复用已校验的冻结Stage1 token。旧实验和候选权重保留。前后任务不同，恢复旧配方不等于保证复现20/20。
