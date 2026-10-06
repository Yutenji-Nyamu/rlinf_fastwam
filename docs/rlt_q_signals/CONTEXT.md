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

2026-10-07 01:00：两组smoke均通过（CP2完整、4次更新、160条回放、非均匀Q权重和非零梯度），CP2另经CPU回放重载/信号重算/合同校验通过。各自fresh正式800已启动并核首轮80条真实回放及信号；仍处原10k预采集阶段，正式update0。Q-U6 driver50987/start410961868，Q-Norm7 driver50956/start410961692。EXPO driver3503441/start410425820保持4/5，CPU资源协调器为3008973/start410890278，旧owner18670退休。数据盘余6.1TiB、root余32GiB，0–3无上下文；原完整Stage1已精确复制并回读核SHA。

本地独立分支`codex/rlt-q-signals-20261006`，工作树`worktrees/rlt-q-signals-20261006`；原dirty保留。运行源码`b1d2d8d553fa057435e0169846ec99c3a2c48c28`已发布；同分支后续提交只补文档和轻量回执，运行checkout保持该SHA。后续观察10k初池→15k初始化更新→在线U5及R25固定评估，不修改预算。原RLT低优先级候补：EXPO和本次两组释放后才可恢复，原四卡回归配置还需补验渲染绑定，勿重放旧owner。
