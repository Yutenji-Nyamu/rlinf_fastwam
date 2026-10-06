# Norm信号合同

沿用本地既定定义：读取π0.5 action expert每层残差更新后的hidden，末层在final normalization之前。固定动作位置h，对ODE10的最后5次前向、最深3层各求hidden维L2，再平均15个标量，输出detached FP32[B,H]。不使用最终14维动作幅度，不先平均hidden。

已有前向hook为实现参考，必须验证实际模型是否调用这些层，以及开关信号时主动作、RNG、cache保持。Norm不新增ODE5比较前向。新的信号元数据必须区分U，回放/校准/断点拒绝混用。

BC继承过去5轮log统计、bounded_linear[0,5]、首轮单位权重、成功入池冻结、原成功长度过滤及逐H FM误差加权。DSRL继承当前U方案的历史chunk信号：有效动作上mean(log signal)，global batch内min-max后exp/mean、温度2.5，用于actor loss；critic保持原算法。任务、SFT、种子、动作/环境预算、优化器、评估保持。

来源：C:/Users/86136/Documents/seek/outputs/dvca_signal_study_2026-10-03/24_NORM_INSTANCE.md；现有参考hook位于worktrees/rlt-q-signals-20261006/rlinf/algorithms/rlt/norm_signal.py。本轮属于该信号的训练适配，不等于原论文已验证的机器人方法。
