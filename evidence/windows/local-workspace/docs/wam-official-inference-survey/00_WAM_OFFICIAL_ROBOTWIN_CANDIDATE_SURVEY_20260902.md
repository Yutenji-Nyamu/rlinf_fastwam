# 下一条官方 RoboTwin WAM 推理：候选、控制合同与推荐

日期：2026-09-02

## 1. 结论

如果目标是“在深圳机上再跑通一条原作者提供的 RoboTwin 闭环推理”，首选 HUSTVL 的
**Faster-WAM**（`hustvl/FasterWAM`，arXiv `2608.04404`）；如果更看重与 Fast-WAM 的机制差异，
首选 **AHA-WAM**；如果更看重轻量和后续高并发，选 **LiLa-WAM**。

注意：2026-08 同时存在另一篇同名
[Faster-WAM: Do World Action Models Need Deep Action Modules?](https://arxiv.org/abs/2608.02365)。后者来自另一团队，
研究浅 ActionDiT；本调研推荐的不是它。后续所有 packet 必须同时写 repo 与 arXiv ID，不能只写模型名。

推荐顺序：

1. **Faster-WAM**：官方 RoboTwin 代码、权重和统计量齐；qpos 路线；与当前已跑通的 Fast-WAM 最近，复现风险最低。
2. **AHA-WAM**：官方 RoboTwin manager 和 HF 权重齐；14D action；异步视频规划器与动作执行器带来更大的研究差异。
3. **LiLa-WAM**：0.5B、动作输出 qpos14；但观测状态是 endpose16，后续接 RLinf 仍需扩展 observation 合同。
4. **Kairos-4B / Motus**：影响力和发布成熟度高，但运行栈更重；Motus 在 AutoDL 已经有官方推理成功记录，深圳再跑的新增信息较少。
5. **FlowWAM**：外部仍是 qpos，但内部用 optical flow 表示动作；研究新颖，后续 DVAC/RL 对齐更复杂。
6. **LaWAM**：官方 standalone 值得跑，但 EEF16 会引入 IK/MPLib 规划路径，不是下一条 RLinf 主线的最低风险选择。

## 2. 候选矩阵

“规模”按模型公开口径或主要 backbone 记；它不等于闭环峰值显存。论文成功率是作者协议下的结果，不当作深圳复现结果。

| 模型 | 时间 / 成熟度 | 规模 | 官方原生 RoboTwin 闭环 | 外部动作合同 | 主要价值与限制 |
|---|---|---:|---|---|---|
| [Faster-WAM](https://github.com/hustvl/FasterWAM) | 2026-08；代码、权重、stats、eval 齐；当前约53 stars/3 forks/4 commits，影响仍早期 | Wan2.2-TI2V-5B 系 | 有 | qpos14 | 另一团队基于FastWAM代码提出的后续，不是原作者续作；最容易复用现有三相机、RoboTwin 和模型资产 |
| [AHA-WAM](https://github.com/serene-sivy/AHA-WAM) | 2026-06；代码、HF 权重、manager 齐 | Wan2.2-5B 系 + action expert | 有 | 14D action / qpos 数据 | 异步 slow video planner + fast action executor；作者报 RoboTwin 平均 92.80%；后续 RL 要处理状态化 async context/cache |
| [LiLa-WAM](https://github.com/teee000/LiLa-WAM) | 2026-08；代码、权重、bridge 齐 | 0.5B，总可训练 0.2B | 有 | 输出 qpos14；输入 endpose16 | 单 24GB 可训练，作者报 clean50 90.48%；轻，但不能把“qpos 动作”误写成“14D qpos 观测” |
| [Kairos](https://github.com/kairos-agi/kairos) | 2026-06/07；生态和提交历史较成熟 | 4B | 有 | qpos14 | 高影响、官方 RoboTwin 权重；HTTP/server 与多 GPU 运行栈较厚 |
| [Motus](https://github.com/thu-ml/Motus) | 2025-12 / CVPR 2026；发布成熟 | 约 8B 总体 | 有 | qpos14 | 影响力强，三视角；预编码文本时仍大于 24GB；AutoDL 已有成功推理与旧 RLinf smoke 记录 |
| [FlowWAM](https://github.com/YixiangChen515/FlowWAM) | 2026-07；官方推理入口完整但仓库较新 | Wan2.2-5B 系 | 有 | 最终 qpos；内部 optical flow | 模态设计差异大；后续把 denoise 信号映射到物理 future action 比直接 Action-DiT 更难 |
| [LaWAM](https://github.com/RLinf/LaWAM) | 2026-06；RLinf 组织、权重和评估齐 | 总体约 2.3B；LaWM 230M | 有 | absolute EEF16 | 轻量 latent world module、作者结果强；但外部控制进入 IK/MPLib，不是当前 qpos14 主线 |
| [X-WAM](https://github.com/sharinka0715/X-WAM) | 2026-04/06；代码权重齐但仓库早期 | Wan2.2-5B 系 | 有 | EEF 路线 | 4D video/depth/action 联合建模；同样有 EEF planner 风险，且运行较重 |
| [Light-WAM](https://github.com/L1ziang/Light-WAM) | 2026-06；极轻但 release 较早期 | 2B，总可训练约 0.44B | 有 | Fast-WAM/qpos 数据路线 | 适合资源与高并发研究；官方结果明显弱于前列候选 |
| [StarWAM](https://github.com/shaohua-pan/StarWAM) | 2026-07；通用研究 codebase | Wan2.2-5B 系 | 有 | qpos14 | 与 Fast-WAM 接近；当前更像模块化 codebase，而非成熟独立方法 release |

以下不适合作为本轮“官方 RoboTwin 闭环”首选：DreamWAM 当前主要发布 LIBERO；DreamZero 原始 release 不是 RoboTwin；GigaWorld-Policy 原仓侧重 open-loop，RoboTwin 支持来自另一套运行面；OpenDW 的 RoboTwin 材料更偏 action-conditioned 生成，而非清晰的闭环成功复现。

## 3. qpos、EEF 与内部表示必须分开

RoboTwin 官方支持：

- `qpos`：`left_arm_joints + left_gripper + right_arm_joints + right_gripper`。双臂 ALOHA 通常是 14D。
- `ee`：每只手 `xyz + quaternion + gripper`，双臂固定 16D。
- `delta_ee`：相对末端位姿增量，是第三种合同，不能与 absolute EEF 混用。

qpos 的好处不是“算法更高级”，而是外部 policy 输出可以直接进入关节控制路径；EEF target 还要经过 IK、碰撞检查和路径规划。后者会新增几类失败：目标不可达、self/table collision、planner timeout，以及并发环境中长尾延迟。

还有两个常见误区：

1. **观测空间不等于动作空间**。LiLa-WAM 输入 endpose16，但输出是 qpos14。
2. **内部动作表示不等于机器人控制接口**。FlowWAM 内部生成 optical flow，最后仍解码为 qpos；因此它不会自动引入 EEF planner，但 DVAC 的语义对齐会更复杂。

RoboTwin 一手定义：[Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html)、[Deploy Your Policy](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)。

## 4. LaWAM 旧问题到底是什么

旧记录不能概括成“EEF 放不进 RLinf”。实际有两层：

1. **确定的工程缺陷**：RoboTwin 已产生 `endpose_states[B,16]`，旧 RLinf 的 observation 白名单却把它丢掉，LaWAM 在第一次构造输入时缺字段。把该字段贯穿数据链后，此问题消失。
2. **真实的控制/规划长尾**：absolute EEF target 会进入 RoboTwin IK/MPLib。旧 `lift_pot` / 并发实验出现 self/table collision、`Invalid start state` 和单环境超过 120 秒；这不是模型加载或 FSDP 问题。

反证也存在：同一旧路径在 `adjust_bottle` 上完成过 B=1 官方语义推理并自然成功，PPO Global Step 1 也完成过。因此：

- EEF 不是原则上不支持；
- 但它比 qpos 多了一层会失败、会长尾的 planner；
- official standalone 和接入 current RLinf 是两个独立问题，应先跑前者。

LaWAM 官方当前已提供独立 policy server、RoboTwin worker、checkpoint 和 EEF 数据合同：[官方仓库与 RoboTwin 入口](https://github.com/RLinf/LaWAM#robotwin-inference)。

## 5. 建议的下一轮 official-only 顺序

### 首选：Faster-WAM

先做 official standalone，不接 RLinf：

1. 锁定官方 repo commit、RoboTwin revision、checkpoint revision 和 stats。
2. 复用已存在的 Wan/Fast-WAM 大模型资产时，逐个用 hash/文件名确认，不重新下载重复权重。
3. 先跑 `move_stapler_pad` 1 个固定 seed、1 个 episode；它比已接近饱和的 `adjust_bottle` 更有信息。
4. 成功后扩到 16 个固定 episodes，记录成功/失败、动作数、query 数、显存、时延和视频。
5. 只有 official oracle 稳定后，才讨论 current RLinf adapter。

选择它的理由不是“论文最新”这一项，而是四点同时成立：官方闭环完整、qpos 合同兼容、模型资产接近现有 Fast-WAM、任务和录像能直接横向比较。

### 第二选择：AHA-WAM

若希望下一模型与 Fast-WAM 有更明显的机制差异，AHA-WAM 更合适。它仍保留 14D/qpos 数据路线，但引入异步 video planner 与 action executor。Standalone 很清楚；以后接 RLinf 时，必须把 async context/cache 当作可重放状态，而不能只保存 action tensor。

### 第三选择：LiLa-WAM

若研究问题变成“小模型能否提高并发、采样效率”，LiLa-WAM 最合适。它避免 EEF planner，但 current RLinf adapter 仍需显式接入 endpose16 observation 和 VTT/task-condition，不能声称是纯 YAML 迁移。

## 6. 当前不建议做的事

- 不先把新模型塞进 RLinf，再用 RLinf 故障判断官方模型是否可跑。
- 不把 EEF 16D 在 policy 侧“硬换算”为 qpos14；IK/规划失败是控制合同的一部分，不能隐藏。
- 不把论文成功率、GitHub star 或单条官方视频当作本机复现。
- 不同时更换模型、控制空间、RoboTwin revision和RL算法；先得到单模型 official oracle。
