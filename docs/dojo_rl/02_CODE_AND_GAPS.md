# 代码复用与必须补的接口

## 上游现状（2026-11-03核查）

RLinf PR1606 → 1736 → 1738已关闭，转由 [PR1754](https://github.com/RLinf/RLinf/pull/1754) 承接。当前head为 `4ea60c4e800ac47f67fd838e6f08b7239e2e275f`，open、未合并。RPent的RoboDojo接入PR96已经合并，但agent工具接入不等于RL训练。

PR1754提供安装依赖、环境类、14维动作排列转换、布局seed、chunk有效步mask、π0.5权重转换及评估配置。其文档明确写“RL training is not supported yet”，公开验证也没有真实Isaac rollout/评估验收。它是有用的工程起点，不是可直接宣称已跑通的Dojo PPO/GRPO方案。

## 我们复用什么

| 部分 | 已有依据 | 增量 |
|---|---|---|
| 仿真、资源和场景 | 已跑通的Dojo固定版本、原生多环境、三相机、H100兼容配置 | 保持版本；包装reset/step/chunk和训练信息 |
| 外层恢复 | `tools/dojo_chunk_eval/run_task.py`、`parallel_queue.py`、`recover_queue.py`，既有U视频wrapper | 复用超时、设备错误、精确清场；训练增加坏组丢弃语义 |
| 模型 | 官方Dojo `RoboDojo-sim-arx_x5-joint-*/59999`，原生XPolicyLab/OpenPI配置 | 同一权重转PyTorch，不更换成RoboTwin底座 |
| RL | 已跑通Clean GRPO的Flow-SDE、分组优势、FSDP更新和续训 | 更换环境/Dojo观测与动作适配；保留方法主体 |
| 评估 | 既有Dojo布局、任务指令、成功判据和步数上限 | 固定验证集、轻量成绩；日常关闭信号和录像 |

## 四个容易“看似跑通、实际训练错”的点

### 1. 动作顺序、变换和权重转换

Dojo原生顺序为 `[左臂6,右臂6,左夹爪1,右夹爪1]`；官方π0.5接口是ALOHA排列 `[左臂6,左夹爪1,右臂6,右夹爪1]`。状态进入模型和动作返回环境都要逆向对应。模型内部32维、真实14维；三相机、归一化、关节增量/还原及夹爪适配共同构成接口，不能只验证shape。

保留Dojo自己的norm_stats和OpenPI data transforms。上游评估以 `state_action_order: aloha`、`adapt_to_pi: false` 接入；这两项必须与我们原生JAX端对照，不能同时再做一次重复转换。部署只改变实现后端，应在同观察同噪声比较动作，再看少量真实执行。

### 2. 上游VectorEnv不等于我们需要的并发

PR所引用runtime `6d76a0c...` 中，`NativeEnv`写死 `args.num_envs=1`；`VectorEnv`创建多个独立Isaac子进程。`step`逐个调用并等待返回，未同时派发。大规模套用既浪费显存，也可能串行仿真。

应复用我们已验收的原生25环境承载方式，增加批量动作入口和每slot状态，而不是直接把这个桥的 `n_envs` 改为24。上游环境类的观测/动作格式、指标和生命周期可借鉴，runtime的扩容方式不照搬。

### 3. GRPO组内必须是同初态

runtime的 `NativeEnv.reset(seed)` 在 `random=True` 时调用 `_random_reset(env)`；后者自行随机选择模板和新seed，没有使用传入seed。于是RLinf外层即使给8个slot传同一seed，仍可能生成8个不同场景。

接入须让布局模板、物体初态、物理reset种子可显式重放，组内相同、策略噪声不同。普通失败和不稳定布局跳过分别记录；一组内换布局时整组一致重置。训练布局与验证布局分开；当前上游示例只有官方Eval_Layout列表，不能把同一列表既训练又当独立验证。不把 `random=True` 自动等同于“训练种子已解决”。

### 4. 奖励、动作截断与状态保存

Dojo提供完整成功判据和部分完成Score。首版建议仅完整成功给一次1，失败0，Score仅作诊断，无需训练奖励模型。上游 `use_custom_reward` 使用terminated构造奖励，必须确认terminated只代表成功而不是任意结束。

上游wrapper把chunk累计奖励/终止放最后一格，同时返回实际执行步mask。若第17步就成功，训练需要核对奖励没有因后33格无效而丢掉、也没有把未执行动作当行为数据。当前只能确定存在这个需要对齐的接口组合，未断言已经产生错误梯度。

对PPO还要区分成功终止和步数耗尽，并正确保存final observation，避免重置后的新场景参与上一回合价值目标。GRPO虽不训练价值网络，同样需要正确成败和组归属。

## 原则

首次接入只做：原生Dojo环境 → 同权重PyTorch π0.5 → 稀疏奖励Clean GRPO。保留已有原生物理和恢复代码；不附带引入planner、SAM3、VLM奖励、U加权、自动选成功种子或新的控制时限。无需修改正在运行的其他实验。
