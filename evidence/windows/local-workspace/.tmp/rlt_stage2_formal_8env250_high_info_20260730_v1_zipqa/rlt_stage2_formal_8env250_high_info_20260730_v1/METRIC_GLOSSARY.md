# RLT Stage 2 优化指标简表

最终 actor 目标是：

$$
L_{\text{actor}}
=w_{BC}L_{BC}-w_QQ_0(s,\pi(s)).
$$

cycle 250 使用 `w_BC=2.5`、`w_Q=0.45`：

$$
0.03200-0.11319=-0.08119.
$$

所以 actor loss 变成负数是 Q 收益项超过 BC 惩罚项，不是异常。

| 图中指标 | 精确含义 | 怎么读 |
|---|---|---|
| `actor loss` | `weighted BC - weighted Q` | 可为负；必须结合两项和成功率判断 |
| `weighted BC` | `w_BC × BC loss` | actor贴近π0 reference的正惩罚项 |
| `weighted Q` | `w_Q × Q0(policy)` | 图上画正数，进入loss时取负号，推动student提高Q |
| `BC loss` | student chunk相对BC target的MSE | 本项目无人介入，因此target就是π0 reference |
| `critic loss` | 两个在线Q相对同一TD target的总体MSE | 低且finite说明拟合稳定；不等于Q绝对准确 |
| `Q0(policy)` | critic 0对当前student action的均值 | 实际actor目标使用这一条 |
| `Q1(policy)` | critic 1对当前student action的均值 | 与Q0的差主要用于双Q一致性诊断 |
| `Q(data)` | 两个Q对replay实际动作的总体均值 | 不是TD target，也不是twin-Q最小值 |
| actor/critic grad norm | 对应更新阶段裁剪前的全局梯度范数 | 超过红线10才触发缩放；长期贴线才危险 |

`reference_dropout=0.5` 只遮掉一半actor输入中的reference，BC target仍保留。
critic target使用10步折扣reward与target twin-Q最小值；success termination不bootstrap，
纯time-limit truncation仍bootstrap。

阶段背景：

- P1，cycle 1–135：reference收集，无优化；
- P2，136–154：reference继续控制，SAC已更新；
- P3，155–191：student接管，BC/Q权重ramp；
- P4，192–250：student稳定在线，权重固定。

三条实色竖线是阶段边界；cycle 250黑色虚线只是最终endpoint。actor图灰色横线是loss=0，
梯度图红色横虚线才是clip 10。

本次最终值：

- actor/critic loss：`-0.08119 / 0.001286`；
- actor/critic grad：`2.821 / 0.170`，均未触发clip 10；
- Q0/Q1/Q(data)：`0.2515 / 0.2383 / 0.2123`；
- 双Q差约0.0132；policy Q比replay-action Q高约12%–18%，属于轻度乐观，需要观察，
  但没有失控证据。
