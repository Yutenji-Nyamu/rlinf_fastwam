# 11月8日：RLT恢复旧成功配方，DSRL降低重复更新

本轮按用户授权实际实施并重启双环境四路实验。4/5为RT2/Dojo Clean DSRL，6/7为Dojo/RT2 Clean RLT；保持16环境并行。旧训练产物保留，新小网络和回放从头开始，避免继承已偏移的Q。RLT冻结基座和已验收的Stage1 token复用。

## 为什么这样改

前轮诊断见 `C:/Users/86136/Documents/seek/outputs/training_diagnosis_20261107/README.md`。RT2 RLT策略Q已约101，而真实回报上限1；Dojo RLT主要表现为接管后采集成功消失和偏离参考动作。恢复旧click_bell成功配方是一次整体回归，不能把不同任务的20/20当成本任务保证。

DSRL此前FP32和Q-only2000确已生效。其真实奖励为−1/0、关闭熵bootstrap，策略噪声Q却为正，而且噪声大量靠近±2。这里更像“小策略找到评分器的空子”，单纯多练Q未必有效。先把UTD20降到5，减少有限经验上重复bootstrap放大偏差；保留Q-only2000、独立噪声和FP32，便于解释这次变化。

完整逐项来源、论文与作者代码及未采用方案见 [SOURCES](../../tools/training_retry_20261108/SOURCES.md)。DSRL作者明确建议按环境调UTD；5是本轮试验值，尚无本任务最优证据。OFN、TD3+BC式Q尺度归一化列为后续候选，未混入本轮；DAWN支持基座数据预采集，但其动作残差设定不等于噪声空间DSRL。

## 实际配方与预算

| 项目 | DSRL：4/5 | RLT：6/7 |
| --- | --- | --- |
| 采集并行 | 16环境 | 16环境 |
| 开始学习的数据门槛 | 2000条chunk转移 | 2000→20000条chunk转移 |
| 接管前训练 | 保留Q-only2000；随后Q/actor联合 | 5000→30000次critic更新，actor每两次critic更新一次；不是纯Q-only |
| 动作执行长度 | 沿用原配置 | RT2 20→10；Dojo 50→10，VLA仍预测H50 |
| batch / microbatch | 256 / 256 | 256 / 256→512 / 256，梯度累积两次 |
| UTD | 20→5 | 保留5 |
| 模仿约束渐变 | 无新增项 | 强约束5000→20000次更新；渐变10000→50000 |
| 折扣 | 沿用 | RT2 .999→.99；Dojo长时限保留.999 |
| 正式局数 | 16×200=3200 | 16×313=5008，对齐旧成功实验约5000局 |

RLT每轮更新上限按16环境/C10重新计算：RT2 3200、Dojo11200，都是“最大新增数据×UTD5”的两倍。旧8环境/200步/C10对应800次新增更新、上限1600；本轮保留这个余量以偿还预采集后的更新欠账。上限不等于每轮固定更新数。episode预算按采集回合计，不把chunk数冒充局数；Dojo物理时限700、RT2 200均保留。

DSRL沿用已验证的加速项；RLT此前SAC加速短测无收益，继续关闭。没有新增推理近似或修改奖励、终止判定、回放语义。

C10会提高重新观测和推理的频率：同样物理步数下，RT2约2倍、Dojo约5倍。chunk转移和相应更新也会增加；这是为恢复更及时的闭环控制作出的学习配方调整，不能算作提速，也不能只用episode数声称计算预算相同。

## 实现与验证记录

- 复用前轮运行的两个源码导出，底层SAC/RLT训练实现不变，关键文件哈希见 [source-reuse.json](evidence/recipe-retry-20261108/source-reuse.json)。新代码只负责配方构建、针对性CPU检查和复用既有交接入口。
- 四路生产配置验证通过；两个RLT的C10/B512实际前后向形状与梯度有限性通过。实际调度器验证19999条不更新、跨过20000按上限偿还warmup预算、预热完成后100条新经验对应500次更新。
- CPU验证脚本最初没有向Ray工作进程传PYTHONPATH，修正后又遇到同进程Cluster单例复用两个卡位标签；改成每卡一个CPU验证进程后四路通过。未重置共享Ray或使用GPU进行这些检查。
- 服务器Ruff与语法检查通过。DSRL实现/形状未变，直接核对新正式采集；RLT各做一次8局基座参考评估/解码一致性检查、一次两轮短测，再从新小网络进入正式训练。短测不续进正式回放。
- `prepare.py`、`validate.py`、`launch.py`、`observe.py`在 `tools/training_retry_20261108/`。本轮部署是一次性实例；`prepare`/`launch`有已执行拒绝检查，不要对当前部署重复运行。
- 执行命令：服务器训练venv运行 `BASE/package/launch.py`；四路具体训练命令、phase超时、输出和卡绑定见各自 [plan.json](evidence/recipe-retry-20261108)。正常结束/启动或短测失败均交还已注册RLT接续；没有新增按表现提前淘汰的启发式门槛。

## 路由与依赖

- 部署：`/data/chenyiteng/deployment-20261108/dsrl-rlt-retry/{rt2-dsrl-clean-g4,dojo-dsrl-clean-g5,dojo-rlt-clean-g6,rt2-rlt-clean-g7}`。
- 源码：`/data/chenyiteng/projects/rlinf-shenzhen/exports/dsrl-rlt-retry-20261108/{dsrl,rlt}`。
- 结果：`/data/chenyiteng/results/dsrl-rlt-retry-20261108/<同名>/formal/formal/`；短测为 `smoke/smoke/`。
- 主路由仍是 `/data/chenyiteng/deployment-20261105/dsrl-fp32-accel/active-runs.json`，已更新四路；FQL join现等待本轮新4/7，5/6独立结束接旧RLT。
- 00:16注册完成。只停止精确匹配UID/PID/start的旧任务，GPU4先从旧RLT交接，随后5/6/7替换；旧RLT调度器仅重启CPU调度进程。未停止共享Ray、其他用户任务或修改0–3卡任务。
- RLT的Stage1 token仍在11月6日旧运行目录，路径与哈希见 [ACTIVE_DEPENDENCIES.json](evidence/recipe-retry-20261108/ACTIVE_DEPENDENCIES.json)。这些旧路径仍是当前依赖，不能因旧日期清理。既有protected-eval候选继续保护。

现场结果以最后一次验收快照及下方更新为准；工程通过不代表已经获得学习提升。

## 最新验收

截至北京时间 **2026-11-08 00:46**，四路owner均存活，全部处于正式流程；新实配与实际TensorBoard resolved配置的关键项一致。旧四个前驱均已退出；调度总owner与FQL join存活，FQL只等待新4/7。

| GPU | 方法 | 状态 | 本轮验证 |
| --- | --- | --- | --- |
| 4 | DSRL | 正式流程 | 正式流程已启动，采集/初评状态见标量快照；最近评估0.4167（step 0） |
| 5 | DSRL | 正式流程 | 正式流程已启动，采集/初评状态见标量快照 |
| 6 | RLT | 正式流程 | 参考7/8、解码差0.0；短测两轮，累计48次critic更新；DCP/小策略/训练状态/回放保存完整 |
| 7 | RLT | 正式流程 | 参考2/8、解码差0.0；短测两轮，累计48次critic更新；DCP/小策略/训练状态/回放保存完整 |

两路RLT都仅做一次两轮短测，随后新启正式；短测不是学习效果对照。参考8局属于冻结基座、样本较小，不把7/8或2/8当作新RLT成绩。正式流程包含初评、预采集、预热与联合学习；当前不能声称四路均已开始参数更新或改善成功率。

最新完整短测恢复点均为CP2；滚动保留机制已移除CP1的大分片，只留下历史保存标记和小元数据。验收按CP2的实际DCP分片、模型文件、训练状态及回放核对；观察工具也将该列表明确命名为保存标记，避免把旧标记误报成仍可完整恢复的检查点。

当前Data可用约419.2GiB。GPU显存、精确进程、正式/短测标量与完整保存元数据见 [最终验收快照](evidence/recipe-retry-20261108/acceptance-summary.json)。模型、回放、RPC密钥及训练环境凭据未下载或加入Git。

后续按真实episode数及学习阶段比较固定评估、采集成功率、Q尺度/真实回报、噪声贴边和参考动作误差。新配方尚无学习收益结论，不依据预热期未训练小策略的低分判断本轮失败。
