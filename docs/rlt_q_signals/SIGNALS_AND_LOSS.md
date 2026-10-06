# 信号与loss合同

原讨论来源：`C:/Users/86136/Documents/seek/outputs/ugrow_training_readiness_2026-10-06/02_RLT.md`、`06_REPLAY_U_AND_LOSS_WEIGHTING.md`；信号来源`outputs/dvca_signal_study_2026-10-03/23_UGROW_INSTANCE.md`、`24_NORM_INSTANCE.md`。这些定义已做推理采集；本次验证训练接线。

例如抽到A/B两个chunk，原Q项为`-(QA+QB)/2`。若信号给出1.2/0.8权重，新项为`-(1.2*QA+0.8*QB)/2`；BC仍取原均值。执行C10以外H50后40位置不参与权重。

U保存为`teacher_ugrow_u`；Norm保存为`teacher_norm`，均浮点[B,H]。curr_obs optional白名单必须完整通到真实transition与compact replay；next_obs保持原核心字段。更新只读取已保存信号，不重算VLA。信号定义和Q方法配置进入resume合同，避免混读旧回放。

Norm复用原推理observer的hook读点，训练版只保留15组标量所需累加，不保存所有hidden。每轮每层恰好命中一次；捕获不修改output或RNG，退出移除hook。U旁路保持原主输出和随机状态。两者仅采集教师时计算。

必要检查：关闭方法/单位权重还原loss和梯度；C10范围、[B,1]广播、全批和micro梯度一致；失败样本参与权重；常数与非有限边界；Norm先L2后平均及hook完整性；真实采集→replay→非均匀权重→有限更新→完整保存/恢复。

只验证实现正确和可运行；收益仍由正式固定评估回答。
