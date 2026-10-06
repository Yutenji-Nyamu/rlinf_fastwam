# RLT 4 Q-U / Q-Norm 当前上下文

用户在2026-10-06授权实现、服务器检查、smoke、推送及正式训练，并明确本次优先于原RLT候补。范围为深圳2物理GPU6/7；EXPO占4/5并保持。凭据仅在连接进程内。

## 历史与本次切口

历史π0.5 `adjust_bottle` Clean4：源码4600080，N4/U5、800轮、B512/micro256、C10×14、teacher H50/ODE10、10k回放预采集/15k初始化更新，固定20场每25轮评估和保存，任务专属Stage1 CP2000。历史DVCA只加权actor的reference-BC；早期共同窗口优于Clean，晚期平台优势未成立。参考`../ugrow/CONTEXT_20261006.md`和`../../server-admin/PI05_RLT_PAIR_AND_FASTWAM_SWITCH_20260919.md`。

本次从同一Clean配置与完整Stage1重新建Stage2，不从旧训练断点或smoke继续。只新增教师信号、回放字段及actor Q项chunk权重；BC、critic TD/target、reward、采样概率、Q/BC课程与其他预算保持。U生产代码复用已验证bfbc9c88，其BC方法在本次关闭。

## 已定方法

对有效回放chunk的C10信号先算`g=mean_h(log(signal_h+1e-12))`，全局B512所有rows含失败一起minmax，再`exp(s/2.5)`并归一至均值1。权重detach，shape为[B,1]，全批算后切micro256。常数信号归1，非有限/负值报错。仅将actor的`mean(Q1)`替换为`mean(w*Q1)`。

Q方法参数：τ2.5沿历史chunk映射，α1持续启用；不引入DVCA的成功掩码、逐动作BC权重、dropout或R500退火。原Clean Q/BC系数课程保持。持续Q加权是本次新方法设定，不声称已有收益。

- GPU6：U，完整ODE10/5、同输入同初噪声，前14维总体std/(RMS+1e-8)后平均坐标；额外5次动作专家求值。
- GPU7：Norm，主ODE10末5轮×动作专家最深3层，residual更新后/final norm前先L2再平均；无额外网络前向。

信号来自冻结教师，是上下文训练优先度，不是小student当前动作的不确定性。详细合同见`SIGNALS_AND_LOSS.md`，操作状态见`EXECUTION.md`。

## 当前阶段

2026-10-07启动前刷新：EXPO owner18670/start410221234、driver3503441/start410425820，计算/图形限4/5；6/7及0–3无进程。数据盘余6.1TiB、root余32GiB。两个新方法及独立namespace/运行目录已准备，37项服务器CPU检查、优先级释放检查、四份配置与渲染边界检查通过。原Stage1精确文件正在从SZ1直传SZ2；无本次GPU工作。

本地独立分支`codex/rlt-q-signals-20261006`，工作树`worktrees/rlt-q-signals-20261006`；原dirty保留。后续：权重完整性核验→发布源码/冻结配置→仅交接EXPO的CPU资源监视器→独立短smoke→fresh正式800→首轮核验。原RLT低优先级候补：EXPO和本次两组释放后才可恢复，原四卡回归配置还需补验渲染绑定，勿重放旧owner。
