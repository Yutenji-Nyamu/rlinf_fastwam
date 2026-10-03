# EXPO-FT 正式实验：讨论与建议

2026-10-01。本轮只读核验论文、固定版本作者源码、已通过的移植与深圳2现场；未启动正式训练、未借卡或修改运行源码。本页是正式方案建议，不能当作已派发配置。已通过 smoke 的合同与数值仍以实施记录为准。

## 人已确定：action expert，不用 LoRA

用户明确保留本工作区已跑通的训练方式：冻结 VLA 视觉/语言 prefix，训练 action expert 及原生 action/time projections；不增加 LoRA。Q、edit 和温度仍按方法训练。这是 action-expert-only EXPO-FT RoboTwin variant，不称作者 LoRA 参数组复现。

继承已跑通 Control 的 π0.5 权重/匹配 norm、三个 camera、双臂 canonical 14D（模型 pad32）、H50、执行 C10、ODE10、每回合最多200真实动作。方法部分优先采用论文/作者代码；DROID 的 H16、C4/8、两相机和 Cartesian action mask 不直接替换 RoboTwin 基座。

## 卡与并行：每组先一张 H100，同步一环境

- fresh/resume 已在深圳2物理4单卡跑通，PyTorch allocated 峰值均14.98 GiB；当时 critic B4、FM B1、候选 micro4、一个环境。该数字不是完整设备峰值，也不证明正式 B64 直接放得下。
- 正式建议有效 B64，通过 critic 观测 micro4、FM micro1起步的梯度累积实现；候选仍8 base＋8 edit。每个全局更新只做一次 optimizer/clip/Polyak，20次critic更新各用新的全局批次。微批只分计算，不改变学习预算。
- 当前候选 micro4仅切候选轴；B64会形成一次256份base推理。必须同时限制观测与候选轴，统一一次全局更新的Q子采样，不能只改 batch 参数。
- 先保持已验证的一环境同步流程；2–4环境是后续吞吐候选，不是算法必要条件，当前 driver 的逐环境终止/reset/replay尚未实现。N8候选不等于8环境。
- 两卡需要额外设备分配/权重同步实现，当前未就绪。论文实验用2×H200，不代表最低要求两卡；作者说明少量GPU倾向同步。EXPO每次TD要运行π0.5下一动作候选，同时更新约6.934亿expert/projection参数，比冻结VLA的RLT Stage2更重，不能承诺同速。正式B64显存、更新耗时和prefix缓存收益需实际计时。

来源：[论文同步/异步讨论](https://arxiv.org/html/2605.25477v2#S4.SS3)、[实验设备](https://arxiv.org/html/2605.25477v2#S5.SS1)、本目录 `IMPLEMENTATION_LEDGER_20261001.md` 和新只读回执 `formal-discussion-sz2-refresh.out`。2026-10-01 15:31北京时间现场：EXPO专用source HEAD=c969b851e36bf5fe4d2b4696f17863370f8ae5ff且clean，旧guardian=RESTORED，物理4–7原四RLT精确driver均存活；0–3无compute进程只是快照，不代表可自动占用。

## 方法预算：建议贴近论文，不能延长 smoke 就当正式

| 项目 | 正式建议 | 依据与限定 |
|---|---|---|
| 候选/小组件 | 8 base＋8 edit，scale .2，10Q，target/selection独立随机2-min，edit用online10Q均值 | 当前核心接线已通过；沿固定原版learner |
| 每次更新调用 | 20次critic，各有效B64；随后各1次FM、edit、temperature | UTD20不是每个真实动作20次VLA更新 |
| 采集/学习比例 | warmup后每累计40真实动作一次call，episode末执行，余数持久保存 | 论文约40动作/call；200动作约5calls。这是迁移后的计数规则；作者pick脚本另是每episode3calls |
| Warmup | 前10个在线episode不学习，在线实际动作数也须≥64；第11个episode才可首次更新 | 来自固定代码入口；warmup不积欠补训call，不把demo数算在线warmup |
| Base optimizer | AdamW 2.5e-5、β(.9,.95)、eps1e-8、wd1e-10、clip1；FP32 master/BF16 compute | 当前已跑通，作者所选OpenPI配置亦为恒定2.5e-5；参数组按人指定，不换LoRA |
| Q/edit/temp | Adam3e-4，γ .99、τQ .005、α初始1 | 已采用原版；不能把3e-4套给完整action expert |
| Demo与online | 初始成功demo加入Q回放；Q均匀采demo＋所有online；FM均匀采demo＋成功online的真实H50窗口 | 作者offline_ratio=0仍先插入demo，非不用demo；无固定50/50。可继承目标任务既有clean50，记录和作者示例10条的差异 |

来源：[论文优化](https://arxiv.org/html/2605.25477v2#A4.SS2)、[更新顺序](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L888)、[warmup](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/train_pi_robo.py#L391)、[demo入池与成功采样](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/data/batch_processor.py#L29)、[base LR](https://github.com/pd-perry/openpi/blob/46407a41183b037313a383ff679683f2773b766d/src/openpi/training/config.py#L875)。

## 原来那句术语，实际是三类选择

### 1. TD：怎样从已执行动作学未来回报

`C=10`是计划执行10步；`K`是这次实际执行的步数。一般K=C。假如第6步已经成功，之后的bootstrap（未来价值项）为0，因此γ^6和γ^10都乘0，终止样本的差别并不在折扣。当前smoke把不足10步的真实动作补零后仍以valid=1送Q/editor，完整候选选优时却没有这个假零尾，存在输入分布差异。

推荐正式版按物理动作保存，允许任意真实起点构造同episode完整C10重叠窗口，而非只从0、10、20等决策边界取窗口。例如第25步成功，可用第16–25步的完整窗口保留成功奖励；不足10步、跨episode的尾巴不虚构动作，也不送Q/editor。若成功早于第10步，严格完整窗协议确实没有该episode的Q窗口，应如实记录取舍，不伪造数据。

完整终止窗：累计实际reward、bootstrap=0、loss-valid=1，能训练Q/editor。timeout和真正终止分开记录；作者会屏蔽触及timeout的训练窗口，正式建议跟随该有效窗口协议，真实final observation仍保存，禁止跨reset。原生0/1 success reward和200动作环境上限保留；不另加DSRL负奖励。FM只用真实完整H50动作，不能把预测尾当监督。critic/editor须使用一致的有效样本；作者只在critic乘valid，不能照抄一个字段就声称其他loss也处理了边界。

来源：[作者物理步窗口与边界](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/data/replay_buffer.py#L438)、[fixed-C TD](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L804)。现smoke variable-K数学合同仍有效，正式建议是改变回放/有效窗口协议，不追溯否认已完成闭环。

### 2. Q视觉：怎样看三张相机图，而不是把VLA换掉

| 方案 | 实际差别 | 建议 |
|---|---|---|
| 保留smoke版本 | 三图各过同一个Torch ResNet50/GN32，空间平均后拼特征；已经跑通 | 可保留为已验证移植variant，不能称只把作者Flax换成PyTorch |
| 更贴近作者 | 相机图在通道上拼接后联合编码，作者basic-block/preactivation、GN4、保留空间flatten→512；RoboTwin保留三相机，改成9通道 | 正式主配置推荐作者结构的PyTorch等价移植；只改新Q视觉支路，不动VLA基座 |

作者实际用的是一个联合编码器，不是每相机各一个独立塔。骨干 basic/bottleneck、归一化分组、早/晚融合、空间flatten/平均均是实质差异。Flax只是实现框架，迁移不要求换JAX/Flax；也不推荐为照搬DROID两相机而删掉RoboTwin右腕图。该正式配置尚未实施和验收。

来源：[camera联合输入](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/batch_utils.py#L12)、[共享pixel encoder与投影](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/networks/pixel_multiplexer.py#L45)、[实际encoder](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/networks/encoders.py)。

### 3. 图像增强：给学习中的图像小幅外观扰动

当前可确认Q/editor支路未使用作者增强；FM走原生train=True，不能泛称所有训练无增强。正式建议Q/editor先用保持宽高比的resize+pad，再对齐95%裁切、±5度旋转、±0.1颜色扰动。分别给camera/current/next采随机参数；同一个增强后的next observation用于next VLA候选和target Q，避免一边raw一边aug。不额外加flip/大旋转，不改14D关节动作或环境；FM原生预处理需核实，避免重复增强。

来源：[论文增强](https://arxiv.org/html/2605.25477v2#A4.SS1)、[作者augmentation](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/utils/augmentation.py)。

## 正式前真正剩下的事

1. **补必要训练规模**：B64梯度累积、二维候选微批、持久动作/call计数和warmup；缓存须包含完整prompt/state token/transform身份，不能复用旧action expert suffix。
2. **补正式数据分布**：demo真实transition进Q，demo/online同样按物理步构窗；统一成功H50随机采样。smoke的256chunk容量、最近128成功窗口、总取最后一窗/固定demo起点均不能延长使用。RGB帧去重、压缩或按episode落盘，不能简单扩RAM。
3. **验收上述正式选择**：完整终止/早终止/timeout窗口，Q/editor有效样本一致，图像增强与候选一致，作者结构Q encoder；正式B64耗时/内存和新进程恢复。沿用已有候选/Q/温度核心，不重新发明SAC协议。
4. **锁定实验定义**：首个task、起始checkpoint及基座初评、总真实动作预算、Control及固定评估种子。建议首组延续已通过流程的pick_diverse_bottles，初始预算可取论文上沿20,000真实动作（含warmup、不含demo和评估）；这是建议，尚未派发。RLT的3000采样轮不等于EXPO的3000动作，比较应同时报告真实动作、episode、call和训练时长。未额外SFT，也不为凑论文约40%初始成功率擅自换权重。
5. **新一次资源合同**：正式借卡前重新刷新并与现役窗口协调，用新run-id/精确卡和停止恢复回执；不重跑已RESTORED的旧guardian。正式停止、失败、结束均应归还原RLT。

以上是后续实施清单；本轮不运行训练。完整smoke通过证明方法闭环与恢复，不证明长期成功率上升；两次在线0/2也不代表正式基座评估为0。
