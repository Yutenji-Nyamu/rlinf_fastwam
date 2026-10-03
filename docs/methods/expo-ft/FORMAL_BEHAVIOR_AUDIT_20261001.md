# EXPO 正式零成功：方法与行为核查

现场截止2026-10-01 22:59:37（北京时间）。本轮只读；未改正式源码、参数、阶段、预算，未启动新GPU评估，未停训或切换RLT。论文为EXPO-FT 2605.25477v2，官方源码固定023cf9cfcb09dab962b6e806fea47d2954b2b9bb；OpenPI固定46407a41183b037313a383ff679683f2773b766d，两份本地官方clone的HEAD和clean状态已核。

**结论：没有发现漏warmup、漏demo、实际未更新或核心TD/编辑损失符号错误。零成功发生在学习前，最需要区分的是起点策略、随机编辑和未经训练的Q选优造成的行为差异。另确认一个沿用官方默认的熵目标尺度冲突，不能据此直接认定当前零成功根因。**

## 实际阶段与现场证据

1. 原生Sidney pi0.5基座新建learner；未使用smoke更新后权重。先做原模型单候选的固定20请求种子评估，5/20成功。
2. 成功clean50 demo预先入Q与FM数据池；前10在线回合/2,000实际动作只采集，不学习、不积欠补训。10回合均失败，且全部完成后才出现首次learner_finished。
3. 第11回合采集完才学习，每40实际动作一call、episode末执行。16回合/3,200实际动作，其中warmup 2,000、post-warmup 1,200，对应30calls；实际Q600、base FM30、editor30、temperature30。
4. 每call真实B64；基座抽样参数均改变，冻结prefix梯度数0，记录的learner指标均有限。30call时FM loss0.008624、critic loss0.004324，不能由loss下降声称成功率提升。
5. CPU mmap只读检查16份在线payload：奖励总和/最大值全部0，terminated全部0，每回合最后一步truncated=1；动作均有限，CUDA未初始化。原生环境success同时返回reward=1与termination=1，driver成功/超时优先级正确。零成功不是只遗漏日志中的正奖励。
6. 六份正在执行的driver/core/backend/replay/cadence/eval源码与输入固定SHA和本地port字节均相同。owner/driver身份活，source_hashes_match=true。

## 逐项方法对照

| 项目 | 论文/固定作者代码 | 当前实际行为 | 判断 |
| --- | --- | --- | --- |
| 基座起点 | 主流程先评零样本；不足时先任务SFT至约40%以上，附录使用任务LoRA SFT | 原Sidney RoboTwin基座初评25%，未另做当前任务SFT | 重要实验起点差异，不是方法不能运行的硬阈值；论文弱/无预训练消融可学但更慢 |
| 在线人工纠正 | 主实验早期HIL多、后期下降；无HIL消融也可学习，但需更多数据 | 无HIL，50成功demo在池中 | 已明确的自主学习variant，不能直接期待论文主实验的早期速度 |
| 采集warmup | 官方train_pi_robo.py:391，至少10已完成episode且在线步数满足batch门槛，不补欠更新 | 前10回合仅采集，首次更新在第11回合末 | 已执行；warmup指不学习，不等于只用原模型动作 |
| Q预训练/LR warmup | 固定入口未要求先离线预训练Q；OpenPI所选base LR schedule warmup_steps=0 | Q/editor随机初始化；base LR无额外warmup | 没有漏掉该固定实现所要求的阶段 |
| Rollout候选 | 8base+8edit，target Q随机2取min，硬argmax；入口warmup也调用sample_actions | 同机制；初始10回合200chunk中155次选edited，占77.5% | 与官方机制一致，但早期随机Q/编辑可能使本基座行为变差；未证明因果 |
| TD下一动作 | 同样base/edit候选选优，再独立随机2Q取min；无熵bonus | 同机制、独立Q子集；gamma^10和真实C10折扣奖励 | 未发现漏选优/符号错误 |
| Q/更新 | 10Q，B64，Q20；随后base/edit/temp各1，约40环境动作一个call | 同计数，四卡分batch | 核心预算一致；4卡不改变更新定义 |
| Q视觉/状态 | 作者ResNetV2 GN4、joint cameras、spatial flatten→512；状态64；MLP256×3 | 对齐结构的PyTorch实现、3相机、14D状态 | 非逐比特复现；MLP Linear初始器/LN默认数值等仍有框架差异，未发现根本结构遗漏 |
| Q/Editor损失 | editor用replay执行chunk作条件，共享视觉stop-gradient；在线10Q均值；tanh与scale log-Jacobian | 同路径，编辑梯度不进入VLA/共享视觉 | 核心梯度路径一致 |
| Replay | demo先入池，Q用所有回合，FM成功回合；固定完整C窗，timeout有效窗屏蔽 | demo+在线按真实物理步重叠C10窗；timeout窗不入eligible；FM真实H50成功窗 | 0成功在线没有堵住学习：目前FM批全部来自demo。最后demo成功reward是来源标注，非parquet实录 |
| LR/target | 小组件Adam3e-4、gamma .99、tauQ .005、alpha初始1；选用作者OpenPI base LR2.5e-5 | 同主要参数；base用已有expert AdamW/clip1 | base参数组/优化器是已选迁移variant，不把论文笼统LR表等同实际OpenPI LR |
| Base target | 论文/代码维护tau_base .001 target副本，固定原版backup采样live base | 未实现未被backup使用的target base副本，backup采样live base | 差异已记录；未改变固定原版backup行为 |
| RoboTwin合同 | 作者单臂7D Cartesian、H16/C4或8、两相机、任务0.05/0.2 edit；pick屏蔽旋转编辑 | 双臂14D关节、H50/C10、200动作、三相机、edit .2作用14D | 继承已有基座；修改范围/反馈间隔影响探索。不能照搬Cartesian旋转下标，不能由scale数字相同断言物理扰动相同 |
| 参数组/增强 | 论文LoRA、图像增强；原base图像冻结 | 用户选择expert+projection、无LoRA、无随机图像增强、视觉语言冻结 | 明确选择，非漏实现；不在本次审计中擅自恢复LoRA/增强 |
| 评估 | 论文30回合随机初态；工作区沿Control评估协议 | 原模型N1固定20请求seed；每25在线episode评完整EXPO | 起点25%与训练0/16的种子和策略不同，不能直接计算训练导致的下降；需要同种子策略拆分 |

## 已证实的熵目标配置风险

当前edit有140维，每维delta∈[-0.2,0.2]；logprob已经含动作scale Jacobian。其真实条件微分熵最高为均匀分布的

`H_max = 140 * ln(0.4) = -128.2807`。

当前target_entropy=-140/2=-70，比理论上限更高，因此平均意义下无法达到。temperature loss为`alpha*(entropy-target)`，会持续向增大alpha的方向施压；这会提高探索/熵项权重。

官方默认adjust_target_entropy=False也有同一冲突，属于继承默认与有界尺度组合的风险，不是本次PyTorch移植特有。官方True分支把target改为`-140/2+140*ln(0.2)=-295.3213`，与缩放前的目标保持相同坐标口径。该数值是坐标修正，不是已验证的RoboTwin最优target。

30calls现场entropy=-129.520、alpha=1.0090，alpha仅增加约0.9%，尚不足以解释0/16。scale修正常数本身不改变editor参数梯度，主要通过温度后续变化影响行为。独立审查确认此推导；当前没有参数修改。

## 最小定位方案与结论边界

优先在相同任务、请求/实际场景种子、动作上限、推理与成功统计下，比较原始N1基座、N8仅Q选优（无edit）、完整N8+8 EXPO；使用原起点及当前checkpoint可以进一步区分初始探索和学习带来的变化。需要保留视频、选中候选、编辑幅度与Q排序。当前未执行此新GPU对照。

先拆清选优/编辑的影响，再决定任务SFT、编辑幅度/有效维度或熵尺度调整；这些都会改变实验协议，需在后续具体scope中锁定，不把原正式中途改配后的数据拼成同一条曲线。没有“只要补某个warmup就保证成功”的证据。

主源：[EXPO-FT §4.2](https://arxiv.org/html/2605.25477v2#S4.SS2)、[消融B.2](https://arxiv.org/html/2605.25477v2#A2.SS2)、[参数D](https://arxiv.org/html/2605.25477v2#A4)、[官方入口](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/train_pi_robo.py#L391)、[官方熵目标](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L286)、[官方编辑损失](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L694)、[base训练配置](https://github.com/pd-perry/openpi/blob/46407a41183b037313a383ff679683f2773b766d/src/openpi/training/config.py#L875)。

现场轻量摘要：[audit-summary.json](evidence/20261001/formal-behavior-summary.json)。原始输入/源码/阶段、环境与回放CPU证据均保存在同E盘目录的sz2子目录；凭据仅在SSH进程内，无凭据落盘。
