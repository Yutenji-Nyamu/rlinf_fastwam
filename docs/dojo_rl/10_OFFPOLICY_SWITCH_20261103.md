# Dojo单卡路线：GPU6 DSRL，GPU7 RLT或回退

用户授权：撤掉Dojo双卡GRPO；6卡DSRL，7卡尝试Dojo RLT，Isaac仍失败则回Robotwin RLT。4/5药瓶Clean保持。准备期间原RLT继续运行，材料齐全才接管；失败自动释放到既有RLT，不重启共享Ray、不碰别人。

## 复用边界

- DSRL核心来自已跑通的 `dsrl-pi0-single-u-n-20261009`，固定提交 `ff64209f8ee668b35837ae6469509b7f0257b89d`。复用SAC、10头Q、紧凑回放、Gaussian warmup、目标Q、严格续训；不开U加权。
- Dojo接口来自本轮已验收GRPO：原生多环境、相机/关节/夹爪排列、delta动作和归一化、独立RPC、逐动作成功与mask、布局过滤、波次换新进程。**不重写仿真循环。**
- 模型是已转换并比较过JAX/PyTorch的Dojo原π0.5，N10/H50/C50/700动作；DSRL冻结全部VLA，只训练噪声控制器与评分器。
- 调度沿用精确PID身份及 `priority_pair_handoff`，单卡预约/释放，RLT最新完整点续训。GPU7只做一次新的独立渲染验证，设备故障则结束Dojo尝试。

## 方法依据与预算

官方DSRL：[代码](https://github.com/nakamotoo/dsrl_pi0)、[RLinf说明](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html)。参数以我们已验收的Clean DSRL实配为准。

| 项目 | 新配置 | 依据 |
|---|---|---|
| 噪声策略/评分器学习率 | 1e-4 / 3e-4 | 历史Clean DSRL |
| batch、Q、UTD | 256、10头均值、每条新转移20次更新 | 历史Clean DSRL |
| 回放/预热 | 25000 / 500条转移 | 历史Clean DSRL |
| 噪声维度 | 32，沿50动作重复 | 既有DSRL设计，不等同原策略独立50×32噪声 |
| 模型与动作接口 | Dojo π0.5、10步、H50/C50、原生700上限 | 原Dojo默认及已验收转换 |
| 仿真并行/每轮新采样 | **8环境、8局** | 用户要求回到历史小批采样；32只留作接入短测，未进正式 |
| 正式样本预算 | 800训练回合：8×100轮 | 总episode预算保持；每轮更新更及时，不把扩大每轮采样当纯工程提速 |
| 中评 | 起点一次；每12轮一次，加第100轮末评，共10次，每次32局 | 8环境×4批，固定32个验证布局轮换；不进入训练回放 |
| 保存 | 每4轮及最终完整保存，滚动最新 | 每32个episode一次，与原保存间隔一致；历史格式仍包含冻结VLA副本，全写Data |
| 显存换速度 | global=micro=256，模型常驻GPU | 复用历史DSRL合批和常驻设置；不扩大global batch、UTD或采样预算来填显存 |

历史DSRL按动作块使用“成功0、否则−1”的到目标成本，超时继续bootstrap；不是GRPO的一次成功+1。保留这个方法定义并单独记录环境成功率。模型复用、仿真上限改变与算法参数继承分别留档，不宣称与旧Robotwin成绩直接可比。

## 验收与限制

短测只跑100动作、UTD1、预热32条，用于真实转移→更新→完整保存→续训，不用于成绩。32环境接入检查后，追加8环境常驻配置的一轮恢复检查。正式回到700动作、UTD20、预热500。模型尺度与场景适配不意味着已验证RL提升。

GPU7历史两次Xid109是Isaac设备上下文超时，普通CUDA/RLT仍可用；没有证据确定硬件故障。旧 `dojo-rlt-per-card` 实际是“Dojo采集结束接Robotwin RLT”的调度器，并非已训练Dojo RLT。若新渲染通过，仍需Dojo自己的RLT阶段一准备，不能拿Robotwin阶段一直接冒用。

## 运行入口与输出

- 训练源码：服务器 `.../worktrees/dojo-dsrl-20261103`，由历史DSRL提交导出，再运行本分支 `tools/dojo_offpolicy/integrate.py <source> <overlay>`。
- 当前部署：`/data/chenyiteng/deployment-20261103/dojo-offpolicy-g6-r3-n8`；r1/r2保留前序短测和修复记录。三份YAML见本轮轻量产物目录。
- CPU所有者：`python tools/dojo_offpolicy/owner.py <deployment>`；内部按 `training_entry.py --config-path <deployment>/config --config-name smoke → smoke_resume → formal` 顺序运行。
- 模型、仿真、训练均绑定物理6卡。旧版Isaac残留在4卡的纯图形7 MiB上下文已单列记录，无4卡计算任务；不能称完全零占用。
- 产物：`/data/chenyiteng/results/dojo-offpolicy/dojo-offpolicy-g6-r1`，正式点在 `formal/formal/checkpoints`，日志和曲线保留。更好正式评估的噪声控制器另存保护目录；完整续训依靠最新完整点。
- 短测每段最多1小时；原生启动10分钟、RPC步进15分钟、reset20分钟。正式仿真错误最多3次从完整点重启，丢弃未完成波次；每次退出先释放回RLT。其他错误直接留RLT，不无限循环。
