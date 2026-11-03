# RoboDojo RL公开代码：能借什么

2026-11-03继续检索。范围：RLinf仓库内全部RoboDojo命中PR、RPent及runtime、官方RoboDojo/XPolicyLab、FluxVLA和个人fork。状态依据原仓库页面；读取了PR1754环境/桥接源码、FluxVLA PR93补丁、RPent PR319补丁。未启动新仿真或训练。

## “上游”具体是谁

前文指 **RLinf仓库的Dojo接入贡献者**，不是Dojo官方已经发布的RL训练方案。最初 [PR1606](https://github.com/RLinf/RLinf/pull/1606) 作者为 `littleZ05`，经1736、1738重整，当前 [PR1754](https://github.com/RLinf/RLinf/pull/1754) 作者为 `qurakchin`。当前页面仍open；其检查只覆盖配置、转换、静态/CPU项目，没有真实Isaac端到端RL验收。

选择扔瓶子的直接工程依据是他们已为它写了环境配置、布局列表和π0.5评估入口，能少做一部分首次接线。任务优先级的主要本地依据仍是历史52%基础、700动作上限。**这不构成扔瓶子优于叠碗的算法结论**。叠碗保留，其历史48%及我们实际成功视频/原生并发经验同样有用。

## 分类结果

| 项目 | 已提供什么 | 本次可借 | 不直接照搬 |
|---|---|---|---|
| [RLinf PR1754](https://github.com/RLinf/RLinf/pull/1754) | 环境注册、reset/chunk接口、14维排列、JAX转换说明、评估YAML | 注册/接口契约、动作排列、依赖版本线索 | 未验收的RL训练；逐slot独立Isaac、同步等待的runtime扩容方式 |
| [RLinf/RoboDojo runtime](https://github.com/RLinf/RoboDojo/blob/6d76a0c4a49a7a47bf5ee3924de2ccd7ecb8a8b7/robodojo_runtime/bridge.py) | 原生控制转换、结束状态保持、实际执行数量 | 逐slot结果字段、退出生命周期、关节/夹爪执行映射 | 随机reset不使用传入seed；每个slot启动一个模拟器；每动作强制final_check的语义需与原生评估区分 |
| [RPent](https://github.com/RLinf/RPent/pull/96) | agent调用Dojo的工具和π0.5入口 | 安装/服务边界、错误清场经验 | agent规划器、SAM3与动作原语不是π0.5 GRPO更新 |
| [RPent PR319](https://github.com/RLinf/RPent/pull/319) | VPP2/动作原语的探索记忆和后续评估，依赖307/310 | 训练/测试布局分离、明确原生成功与异常的记录方式 | 58/120结果来自Astra+VPP2+记忆的选定子集，不能当π0.5 RL结果 |
| [FluxVLA PR93](https://github.com/FluxVLA/FluxVLA/pull/93)，已合并b7042d2 | Dojo的π0/π0.5/GR00T/SmolVLA监督训练配置、数据变换和ZMQ评估 | 三相机、14维顺序、相对关节/绝对夹爪、量化归一化核对；episode绑定的策略服务、并行启动 | 其训练是SFT，且有自己的模型权重/统计；不替换我们选定的官方Dojo checkpoint和norm_stats |
| [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/blob/10ab2651a0b7cd5b8a948cb13f18ccdf818c6de4/policy/Pi_05/model.py) + 官方Dojo | 我们已跑通的原权重、变换、原生多环境、布局/任务实现 | 模型与环境语义的第一依据 | `get_action_batch`内部逐环境调用，不能按名字认定已经模型合批 |
| [embodied-first-exam](https://github.com/mll-lab-nu/embodied-first-exam/blob/main/tasks/robodojo/README.md) | Dojo冻结场景、agent评测封装 | 固定布局、运行隔离和结果约定的辅助参考 | 不是VLA参数更新器 |
| [Doo-mon/RLinf](https://github.com/Doo-mon/RLinf)、[Doo-mon/RoboDojo](https://github.com/Doo-mon/RoboDojo) | 个人fork存在 | 作为后续实现线索 | 未找到其公开Dojo GRPO训练配置与学习曲线，不能算又一份RL成功实现 |

本轮没有找到**可核实已完成Dojo π0.5 GRPO学习闭环并公布提升曲线**的现成实现。此结论仅限检索到并核对的公开内容，不是证明网上不存在其他代码。搜索中PR1760实际是Search-R1 PPO；Java Robocode同名“robo-dojo”也不相关，均排除。GitHub API读取部分请求遇到限流，已通过原PR页面、固定版本raw源码及补丁继续核查。

## 新发现的实用模型细节

[FluxVLA π0.5配置](https://github.com/FluxVLA/FluxVLA/blob/b7042d2/configs/pi05/pi05_paligemma_robodojo_full_finetune.py)与现场已跑通的Dojo OpenPI `LeRobotAlohaDataConfig`相互印证：12个臂关节预测相对于当前状态的变化，2个夹爪保持绝对值；使用所选权重配套的q01/q99统计，`adapt_to_pi=False`。这与RoboTwin Sidney底座的绝对关节接口有区别。模型内动作 → 反归一化 → 关节增量还原 → 原生排列 → 夹爪物理行程，必须只转换一次。

因此最短路径是：**官方Dojo权重和原生仿真 + 我们已跑通的RLinf Clean更新链 + 一个薄的批量环境适配器**。不整体迁到新的agent框架，也不为接入重做SFT。
