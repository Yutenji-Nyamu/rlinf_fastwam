> 2026-10-26实际源码核正：本页早期“指标只统计真实执行维度”不是CP/OneDP当前loss的实况；两者实际平均内部32维，直接蒸馏只用真实14维。训练均H50；C10仅为新评估设置。见[51范围核对](51_TRAINING_HORIZON_SOURCES_20261026.md)、[52诊断](52_ONEDP_DMD_AUDIT_20261026.md)。

# π0.5 代码切口与适配草案

核心公式已实现；实配见05。这里保留推导，旧的3000步建议已被正式500轮CP/20轮OneDP取代；成功率以实际训练评估为准。

## 继承已有实现

源码基线：`C:/Users/86136/Documents/rl/worktrees/pi05-policy-budget-eval-20261017`，本轮读到的分支为 `codex/sz3-dense-rlt-recovery-20261021`。实施时从已验收提交切独立分支。

老师：`/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab`。统一使用原始 Sidney π0.5；近期 GRPO 峰值不混入第一轮老师，以免把老师变化当作蒸馏收益。

| 代码 | 复用点 |
|---|---|
| [openpi_action_model.py](../../rlinf/models/embodiment/openpi/openpi_action_model.py:379) | `sft_forward`：已有动作监督；`nft_forward`（768）接受显式噪声动作/时间；`get_velocity`（1372）输出速度；`_build_prefix_cache`（1380）处理图像/语言 |
| [Sidney 数据配置](../../rlinf/models/embodiment/openpi/dataconfig/__init__.py:455) | `pi05_sidney_robotwin`：绝对动作、mean/std、无额外 delta 转换；不要直接照搬默认 `pi05_aloha_robotwin` 的 delta 配置 |
| [SFT 配置](../../examples/sft/config/robotwin_sft_openpi_pi05.yaml) | 训练/FSDP/数据接口的模板；路径与数据变换仍须替换成我们的实配 |
| [配对评估入口](../../tools/pi05_policy_eval/run_batch.py) | 原生环境、相同实际种子、逐 query 固定噪声、完整录像、真实成功判据 |
| [旧评估协议](../../tools/pi05_policy_eval/protocol.py:33) | 目前固定 B32/200 动作，N∈{5,10,15,20,30}。新蒸馏评估另设协议支持 N1/400，不改既有接管实验 |
| [U/四噪声采集](../../tools/pi05_u_video/run_batch.py:131) | 已支持逐任务动作上限、原始 observation NPZ、动作和视频映射，可用作数据接口参考 |

底层采样已接受 `num_steps>=1`；N1 的主要入口限制在旧评估协议。训练调用可求梯度的速度接口，不能穿过 `@torch.no_grad()` 的公开推理包装。

所有方法保留三相机、14 维物理动作（网络内部补到 32）、H50/C50、原 tokenizer/归一化。动作与时间 mask 继承同一基座；指标只统计真实执行维度与有效动作。避免重复归一化、把 delta 和绝对关节位置混用、把填充尾段当真动作。

## CP-flow：保留 CTM 核心

我们时间方向是 $t=1$ 为噪声，$t=0$ 为动作。用动作数据 $a$ 与独立高斯噪声 $\epsilon$ 构造：

$$
x_t=(1-t)a+t\epsilon,\qquad v^*(x_t,t)=\epsilon-a.
$$

拟增加一个零初始化的“目标时间 $s$”条件，沿用动作专家；定义跨阶段输出：

$$
G_\theta(x_t,t,s,o)=x_t-(t-s)v_\theta(x_t,t,s,o).
$$

这样 $s=t$ 时输出原输入，$s=0$ 时输出动作估计；新条件初始化为零时保留原网络的速度输出。部署 N1 只调用 $G_\theta(z,1,0,o)$。

训练沿用官方 CTM 路径：取 $t>u\ge s\ge0$；冻结老师从 $t$ 走到 $u$ 得到 $x_u$。比较两条路径：

$$
P=G_{\rm EMA}(G_\theta(x_t,t,s,o),s,0,o),
$$

$$
Q=\operatorname{stopgrad}\!\left[G_{\rm EMA}(G_{\rm EMA}(x_u,u,s,o),s,0,o)\right].
$$

距离复用官方`Huber_Loss(delta=-1)`原式：delta按动作块维度自动设定，**不是普通MSE**。官方先算逐元素平方误差，再算`sqrt(mse**2+delta**2)-delta`；我们使用等价的稳定写法避免小误差相减丢失精度，不擅自替换成PyTorch的另一个Huber定义。数据监督使用已有flow-matching loss；它对应原CP的动作数据约束，但时间加权与EDM DSM不同，记录为适配项。

关键实现点：EMA 参数冻结，但 **P 中外层 EMA 对其输入的梯度必须保留**，否则学生收不到梯度；Q 才整条 detach。老师局部推进沿用 π0.5 的 Euler 方向，不能搬 EDM 的 sigma=80 等尺度。局部推进步长、采样时间网格在首轮配置中明确记录。

仅端点的一致性模型可以作为简化备选，但那比原 CP 的 CTM 少了一层机制；第一版计划保留上面 CTM 结构，不另开一个模糊命名的 baseline。

## OneDP-S-flow：保留分布匹配目标

老师 $v_T$ 冻结，学生动作生成器 $G_\theta$ 和辅助速度网络 $v_\phi$ 由同一老师初始化。学生拟用原 N1 形式：

$$
A=G_\theta(z,o)=z-v_\theta(z,1,o).
$$

训练取独立噪声，给学生输出加噪：

$$
x_t=(1-t)A+t\epsilon,\qquad 0<t<1.
$$

在这一明确的线性高斯路径下，速度对应的噪声预测和 score（指向更可能动作的方向）为：

$$
\widehat\epsilon=x_t+(1-t)v(x_t,t,o),\qquad s(x_t,t,o)=-\widehat\epsilon/t.
$$

辅助网络用 `stopgrad(A)` 的加噪样本训练。若保留 OneDP 的单位噪声预测损失，其流写法为：

$$
L_\phi=\|\widehat\epsilon_\phi-\epsilon\|^2
=(1-t)^2\|v_\phi-(\epsilon-\operatorname{stopgrad}(A))\|^2.
$$

更新学生时，老师/辅助预测只提供固定方向，梯度经过学生输出：

$$
\nabla_\theta L=\mathbb{E}\left[\frac{w(t)}{t}(\widehat\epsilon_T-\widehat\epsilon_\phi)\nabla_\theta x_t\right].
$$

沿用论文 $w(t)=\sigma(t)^2=t^2$，可用内积代理损失实现此梯度。它等价于对 $A$ 使用方向 $t(1-t)^2(v_T-v_\phi)$，**不是直接最小化两个速度的 MSE**。代理损失值不是可报告的真实 KL 数值。

先更新辅助网络，再用其当前值更新学生；两个优化器、显式 detach。拟在内部时间区间采样（例如 [0.02,0.95]，属于 flow 适配起点），避免端点除零/零信息；不要把论文 DDPM 的索引 65、95 当成此处时间。

这部分可参考 FastGen 的预测类型转换和交替优化，但不引入其 DMD2 的 GAN、额外 guidance、梯度归一化等不同目标。

## 起步参数与保存

| 项目 | 首轮建议与依据 |
|---|---|
| 可训练部分 | 两种方法都先冻结视觉语言前缀，训练动作专家；OneDP 辅助专家也训练。这是本项目资源/基座选择；原 OneDP 训练了视觉编码器 |
| batch | 全局 64 作为已有 SFT 模板起点；显存不足先降 microbatch、累积到 64，实际容量由服务器 smoke 决定 |
| CP 学习率 | π0.5 SFT 模板的 2.5e-5 作为待测起点；不照搬官方 U-Net 的 1e-4 |
| CP 两项权重 | 1:1 可作为官方组合对应的起点，但需记录两项实际量级；换成 FM 后不是数值等价 |
| OneDP 学习率 | 学生 1e-6、辅助 2e-5，Adam β=(0,0.999)，来自最终论文；在 π0.5 上仍需小规模验证 |
| 正式训练 | 一个种子；CP按官方配置500轮，OneDP按正式论文20轮；具体更新数按真实数据窗口数计算，见05 |

显存包含学生/辅助网络的梯度、优化器状态和激活，不能按现有推理显存估算训练余量。若视觉语言前缀冻结且各副本完全相同，可共享一次观测编码；先保证与独立前向一致，不新建整轨迹反复加载/校验缓存。

保存学生、CP EMA 或 OneDP 辅助网络、全部优化器/调度器、RNG、数据采样进度、配置、老师路径/版本。保留完整续训能力；推理导出另外生成，不能拿它替代训练 checkpoint。

必要 smoke：原 N10 行为不变；N1 真正一次动作专家调用；CP 上述梯度路径存在；OneDP 老师不更新、两优化器只更新各自参数；短保存/恢复能继续；随后真实闭环评估。项目测试均在获授权的服务器卡上执行。
