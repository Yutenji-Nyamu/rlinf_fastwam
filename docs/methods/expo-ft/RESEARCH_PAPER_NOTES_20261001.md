# EXPO / EXPO-FT / Real-Time EXPO-FT：原论文核查

核查日期：2026-10-01。范围为原论文、附录与作者项目页；本文未启动训练或操作深圳服务器。论文结论与迁移判断分开写。代码可运行性、当前仓库默认值和本工作区的精确接入位置另见本专题的代码/迁移审计。

PDF定位（以下为PDF文件的1-based页码）：EXPO-FT v2共20页，§3 p3，§4.1–4.2 p4–5，§4.3 p5–6，实验/主表p6–9，消融B.2 p15–16，任务C p16–18，D.1 p19、D.2 p19–20、D.3 p20。Real-Time v1共17页，§IV-B p5–6，实验p7–9，训练附录VII-E p14–16，其中base online fine-tuning p15–16。原EXPO v3算法在p4–5、消融在p7–8。HTML与PDF正文/附录均已读取；PDF缓存是读取证据，不是实验数据。

## 1. 三篇工作分别是什么

| 工作 | 已核最新版本 | 增量与范围 | 主源 |
|---|---|---|---|
| EXPO | arXiv 2507.07986v3，2026-04-30；论文 PDF 标识 ICLR 2026 | 一般 expressive policy 的 off-policy RL：base imitation、Gaussian action edits、rollout/TD 均用 on-the-fly 选优；主要为单步控制 | [论文](https://arxiv.org/html/2507.07986v3)、[项目页](https://pd-perry.github.io/expo-site/) |
| EXPO-FT | arXiv 2605.25477v2，2026-08-17 | 在 EXPO 上接 π0.5、执行 action chunks、真人干预、真机 actor/learner 系统 | [论文](https://arxiv.org/html/2605.25477v2)、[项目页](https://pd-perry.github.io/expo-ft/) |
| Real-Time EXPO-FT | arXiv 2609.18207v1，2026-09-16 | 延迟感知的异步候选生成、当前观测下快速 edit/Q、RTC 训练、noise-Q backup 过滤 | [论文](https://arxiv.org/html/2609.18207v1)、[项目页](https://pd-perry.github.io/real-time-expo-ft/) |

这里只讨论 Perry Dong 等作者的 Expressive Policy Optimization；同名的 LLM EXPO 不是本方法。Real-Time 是独立扩展，不能把它的 H=16、N=32、trainable vision tower、backup filter 等细节倒写为原版 EXPO-FT 的默认设计。

## 2. 原版 EXPO-FT 的核心机制

证据位置：[§3、§4.2、§4.3](https://arxiv.org/html/2605.25477v2#S4)、[附录 D](https://arxiv.org/html/2605.25477v2#A4)。

1. VLA 从 RGB、proprio 和指令生成 action chunk；预测长 H，实际执行 C≤H。
2. 小型 tanh Gaussian edit policy 输入当前观测特征及 base/参考 action chunk，输出有界修正。它通过 Q 的 action 梯度和熵目标学习，Q 梯度不穿过 VLA denoising chain。
3. Q 评估当前观测与整个将执行的 C 步动作，rollout 在原始和修改后的候选中做硬 argmax。TD 的下一动作也由同类候选选优；只在 rollout 选优、backup 改回普通单样本，不是完整 EXPO。
4. base VLA 仍更新，使用其原 supervised/flow-matching objective；“不让 Q 梯度穿过 VLA”不等于“冻结整个 VLA”。附录 D.1 指 task-specific LoRA SFT checkpoint 与匹配 normalization，RL 时 image encoder 冻结，critic encoder 继续学习。原版论文没有完整列出每一组在线 trainable 参数，需结合发布代码。
5. replay 记录真正执行的动作；HIL 可覆盖 chunk 中部分步，保留其他步，并把覆盖后的 chunk 写回 replay。

论文的 edit loss 为 `E[alpha log pi_edit(delta | obs, a) - Q(obs, a + delta)]`。actor update 的 `a` 来自 replay（公式写法）；rollout 的候选来自当前 base。移植时应检查这两个路径，不要默认所有 actor update 都重新运行 base VLA。

## 3. 输入、模型与预算：论文给了什么

| 项 | EXPO-FT v2 的明确报告 | 位置 |
|---|---|---|
| 图像 / 状态 | side + wrist RGB 224×224；末端位置、朝向等 proprio | §4.1、§5.1 |
| 动作 / 频率 | 真机末端 Cartesian velocity + gripper velocity，10 Hz | §5.1 |
| 候选 | 8 个随机 base chunks，各一个 edit，共 16；hard top-Q | D.1 |
| 执行 C | 一般 8；灯插入与花插入为 4；绝对预测 H 未在正文/附录明确给数值 | B.2、D.2 Table 4 |
| edit scale | 0.05 / 0.1 / 0.2；任务表主要用 0.05 或 0.2 | D.1、Table 4 |
| Q | REDQ-style 10 网络，target 随机选 2 取 min，LayerNorm | D.1 |
| 视觉 | critic 独立 ResNet-50，edit 复用 critic 特征；图像512维、proprio64维；MLP 256×3 | D.1、B.2 |
| 优化 | B64、UTD20、γ=.99、τQ=.005、τπ=.001、初始 α=1；小组件 Adam 3e-4 | D.2 Table 3 |
| 更新节奏 | 支持 step/episode/batch；实验任务每 episode 1–6 次更新调用；文中另述约每40物理步一次更新 | §4.3、D.2 Table 4 |
| 设备 / 训练 | 两张 H200，8k–20k env steps | §5.1、D.2 |

口径检查：Eq.(5) 用符号 γ，附录说 chunk discount 一致，移植应明确实际是 physical-step γ^C 还是重新定义的 chunk γ；UTD20与每 episode 更新调用数不能直接相乘后称为每物理步20次 VLA 更新。D.2称 τπ=.001 比 τQ=.005 “faster” 与数值不一致，不能据该形容词推导目标更新速度。base optimizer 的精确默认与 trainable filters 需代码审计，论文表不能替代实配。

并行代码审计回传的当前官方原 `EXPOLearner` 事实：`expo_ft.py` 实际target用 `γ**replan_steps`（L804）；每update call先UTD次critic，再1次成功episode BC、1次edit/temp（L888–938）。配套OpenPI EXPO-FT分支任务config报告H16（L875–881），默认C8；示例每episode调用3次，至少完成10episode才开始更新。当前main已合并Real-Time体系，这些代码默认应带commit/分支引用，不能自动等同于论文所有历史实验。精确来源和固定版本由代码审计记录。

## 4. 数据、干预、reward 与证据边界

证据：[§4.1–4.3、§5、附录 B/C](https://arxiv.org/html/2605.25477v2#S5)。

- 若 zero-shot 不够，先收少量演示并 SFT 到约40%及以上，再在线 RL；8项实验演示量约10–40条，各任务单独定义 reward/reset/randomization。
- 主实验为 sparse binary success；规则分类器报告识别准确率超过95%，正式30回合评估由人观察成功。部分 reset 为人手完成。
- 8个真机任务最后均报告30/30；平均 online robot interaction data 为19.1分钟。该时间不是总 wall-clock：不自动包含演示收集、SFT、重置等待、VLA/critic训练、环境搭建。
- DSRL/HIL-SERL 对比只覆盖 Egg Flip、Cube Pick、Pool Shot、Flower Insert 四项；HG-DAgger/SFT 覆盖八项。不能称所有基线在全部八项均做了同样实验。
- 30/30是有限评估样本下的成功率；论文未给 RoboTwin 上的EXPO-FT结果，也不保证迁移后同样样本效率。

消融（附录 B.2，Cube Pick / Egg Flip）：无HIL与半HIL最终仍能到30/30，但需要更多在线步；C4/8优于C16；冻结base在同预算内未到30/30；共享critic视觉特征比独立/来自VLA更快；不预训练也可学习但更慢。原EXPO §5.5另显示 action edits 与TD中的OTF选优都重要，不能只实现一个 residual actor 就算EXPO。

## 5. Real-Time 扩展应单独讨论

证据：[§IV-B / VII-E](https://arxiv.org/html/2609.18207v1#S4)。

- VLA在旧观测上提前生成，当前执行期间将已承诺的d步动作作为clean prefix inpaint；到执行时取候选的 `[d:d+C]` 窗口，用最新观测做edit与Q。
- base以 prefix-conditioned flow-matching BC更新，loss仅作用于postfix；在线按实际延迟而非重新随机d。成功演示/在线成功episodes用于base更新。
- real-world配置H16、C8、N32，32base+32edited；10 Euler denoising。backup用noise-Q给32个raw noises打分，选1个解码、生成1个edit，再由action Q二选一。rollout仍64候选，该filter只降backup成本。
- 原版冻结base视觉，Real-Time附录明确在线视觉塔/投影等可训练；base AdamW 2.5e-5，小组件Adam3e-4；每update call 20 critic steps后各1次base/edit/temperature，而不是VLA也20次。
- 真机30Hz，4个动态任务，无在线HIL；平均42%→97%，10分钟在线数据上限。Kinetix10任务，4seeds，先1M offline transitions再100k online steps，4-step inference delay。两套实验分别有其预算，不能混成RoboTwin配方。
- privileged critic states用于动态真机任务，例如ball position/velocity；这些不是原版EXPO-FT多视角RGB+proprio的无条件默认输入。

如果当前 RoboTwin 的环境 step 等待推理后才继续，任务不包含真实部署延迟，先实现原版EXPO-FT即可；引入Real-Time意味着新增队列、旧/新观测对应、prefix mask和延迟实验，范围明显更大。

## 6. 对 RLinf + RoboTwin + π0.5 的迁移判断

以下是机制推导，具体复用文件由迁移审计确认：

| 与已有路线 | 可以复用的概念 | EXPO-FT必须另外具备 |
|---|---|---|
| DSRL | off-policy replay、SAC式小actor、Q/target、VLA推理与norm接口 | action-space edit条件输入base action、真实action chunk Q、原/edited多候选选优、OTF TD、base online FM训练；不能只替换latent-noise维度 |
| 当前 RLT / RL Token | Stage1 的 encoder 输出 z_rl，Stage2 小 actor 直接输出完整 C×14 动作chunk；可复用交互/评估与部分 off-policy 骨架 | EXPO 保留 base 候选，经小 actor 加 delta 后与原候选比较；当前 RLT 的小 actor 不是输出 RL token，不能按 token 输出层简单替换；critic条件字段与OTF TD路径需重做 |
| RoboTwin | 原生success reward、演示与自动reset | 摄像头/proprio/动作norm统一；环境真实执行动作与C-step rewards/mask写回；chunk内done截断正确 |

推荐范围是独立EXPO方法实现并复用已跑通的RLinf基础设施，而非把官方真机JAX客户端整体换成生产RLT入口。第一版可无HIL（论文支持，但样本预算需重新验证）；先继承当前control的动作上限、H/C、模型与资源，不为“照论文”暗改预算。少量代码并不等于只加一个loss：最低完整闭环含base FM、edit SAC、chunk TD、rollout/backup候选选优、可恢复replay与权重。

## 7. 下一次追问的定位入口

- “方法 / 为什么有效”：本笔记§2 + EXPO §4、§5.5。
- “30/30 / 19分钟是否可靠”：本笔记§4 + EXPO-FT §5 / B / C。
- “本仓库怎么跑 / 默认参数”：以本专题代码审计及commit为准；本笔记不声称运行验收。
- “要不要Real-Time”：本笔记§5；先明确是否研究真实推理延迟，再单独立项。
