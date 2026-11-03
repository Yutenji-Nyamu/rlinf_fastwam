# 来源与复用边界

核查日期：2026-11-03。公开源码已经阅读；本轮未实际运行新GPU链。

| 来源 | 固定版本/入口 | 用途 |
|---|---|---|
| Dojo官方 | [仓库](https://github.com/RoboDojo-Benchmark/RoboDojo)、[论文](https://arxiv.org/abs/2607.04434) | 任务、布局、原生成功与Score；仓库面向评估，不提供现成RL训练器 |
| 已跑通Dojo任务代码 | [726e9aa](https://github.com/RoboDojo-Benchmark/RoboDojo/tree/726e9aabfaa642203722eb126f5eaf0f37f3e1ad) | 保持既有仿真版本；stack_bowls/put_bottles/build_tower/stack_blocks上限800/700/1050/550 |
| 已跑通π0.5适配 | [XPolicyLab model.py](https://github.com/XPolicyLab/XPolicyLab/blob/10ab2651a0b7cd5b8a948cb13f18ccdf818c6de4/policy/Pi_05/model.py)、[deploy.yml](https://github.com/XPolicyLab/XPolicyLab/blob/10ab2651a0b7cd5b8a948cb13f18ccdf818c6de4/policy/Pi_05/deploy.yml) | JAX权重、相机、状态动作打包、逐环境推理基线 |
| RLinf新Dojo提案 | [PR1754](https://github.com/RLinf/RLinf/pull/1754)，head `4ea60c4e800ac47f67fd838e6f08b7239e2e275f` | open未合并；取代1606/1736/1738；提供评估接入、转换、动作排列和mask |
| 新wrapper | [robodojo_env.py](https://github.com/qurakchin/RLinf/blob/4ea60c4e800ac47f67fd838e6f08b7239e2e275f/rlinf/envs/sim/robodojo/robodojo_env.py) | `state_action_permutations`、reset、step、chunk_step、组seed |
| runtime | [bridge.py](https://github.com/RLinf/RoboDojo/blob/6d76a0c4a49a7a47bf5ee3924de2ccd7ecb8a8b7/robodojo_runtime/bridge.py) | 实际多进程/顺序step、随机reset不使用传入seed、terminal latch；明确迁移边界 |
| RPent | [PR96](https://github.com/RLinf/RPent/pull/96)、[文档](https://rpent.readthedocs.io/en/latest/rst_source/simulators/robodojo.html) | 已合并agent集成和转换线索；不作为RL学习成功证据 |
| πRL | [论文](https://arxiv.org/abs/2510.25889)、[RLinf模型页](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html) | π0.5 Flow-SDE、PPO/GRPO可训练概率、其他benchmark效果 |
| EXPO | [官方代码](https://github.com/pd-perry/expo-ft) | 回放、Q、Editor和策略学习的后续方案；不作为首个环境接入的必需项 |

## 本地历史入口

- 工作区旧 `docs/robodojo-openwam/REVIEW_20261003.md`、`N25_FORMAL_20261004.md`：原生评估、25环境与资源实测。旧 `RLINF_RPENT_REUSE.md` 的9月PR状态已过时，本专题替代其当前支持判断。
- 仓库 `docs/dojo_u_video/04_EXECUTION.md`、`05_TASK_BUDGETS.md`、`13_PARALLEL_AUDIT_20261025.md`、`14_PARALLEL_EXECUTION_20261025.md`：信号、录像、原生并发和六任务并发。
- `tools/dojo_chunk_eval/policy_entry.py`：两种合批尝试失败后回退逐环境的真实实现，不把未通过分支当正式可用优化。
- `docs/experiments/entropy-replay-20261102/grpo-config-grpo_clean.yaml`：当前Clean的G8、Flow-SDE0.5、lr5e-6、global512/micro32等参数来源。Dojo预算与并发需单独定，不能连200动作限制一起拷贝。
- `docs/experiments/pi05-followups-20261019/expo-closeout.json`：turn_switch EXPO初始40%、末次85%、峰95%、末五均69%；任务/分母与Dojo不同。

历史CSV仅整理实验条件与成绩；服务器容量、账号、进程与网络资料保留在本地维护记录，不进入本专题公开载荷。
