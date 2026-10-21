# 来源、代码和复用判断

核对日期：2026-10-21。这里只把读到的论文/源代码作为依据；第三方实现未在本轮运行。

## 官方主依据

| 来源 | 已核实内容 | 对我们的用途 |
|---|---|---|
| [Consistency Policy，RSS 2024](https://arxiv.org/html/2405.07503v2)；[官方代码](https://github.com/Aaditya-Prasad/consistency-policy) | 基于 EDM 老师，使用 CTM 跨阶段映射和示范动作监督；支持一步/少步 | 保留训练结构，将 EDM 参数化换成 π0.5 的流参数化 |
| [One-Step Diffusion Policy，ICML 2025](https://proceedings.mlr.press/v267/wang25ba.html)；[NVIDIA 项目页](https://research.nvidia.com/labs/cosmos-lab/onedp/) | 随机版三网络，确定版可去掉辅助网络；老师冻结；通过加噪动作的分布差异训练 | 采用 OneDP-S 核心目标与交替优化 |
| [RoboTwin 官方动作上限](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/env_cfg/task_config/_eval_step_limit.yml) | 50 个任务中，24 个上限为 400；其他为 500～1700 等 | 第一阶段选 400 任务；50 任务阶段各用原任务上限 |
| [OpenPI π0/π0.5 源码](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py) | 线性混合噪声与动作；网络预测速度，积分生成动作 | 校对时间方向与训练目标；执行仍用我们已验证的实现 |

OneDP 的正式出版信息以 PMLR 为准，不把旧 OpenReview 投稿页的 ICLR 2025 写成最终录用。到本轮为止，从官方项目页、PMLR 和定向检索尚未找到可确认的公开官方 OneDP 实现。

## 其他团队与个人实现

| 实现 | 阅读深度 / 定位 | 可复用内容与边界 |
|---|---|---|
| [CleanDiffuserTeam/CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) | 已读 `consistency_model.py` 的 teacher 初始化、加噪、一步 teacher 推进、EMA 和损失 | CM 组件结构很清楚；它是通用一致性模型库，不是 RSS CP 的完整 CTM 复现。此次读的是 main 快照，项目另有 lightning 分支 |
| [NVlabs/FastGen](https://github.com/NVlabs/FastGen) | 已读 DMD2 的学生/辅助网络交替更新、预测类型转换、Flux 配置 | 参考大型流模型的 `flow↔x0↔eps`、梯度隔离和 checkpoint 组织；不把 DMD2 的 GAN、归一化或更新比率默认混进 OneDP |
| [ManiCM-fast/ManiCM](https://github.com/ManiCM-fast/ManiCM) | README、仓库树、DP3 策略入口 | 点云动作策略的一致性蒸馏和部署组织可参考；与我们 RGB+语言 π0.5 的输入不同，不整套替换 |
| [aminamazlin/onestep-diffusionpolicy](https://github.com/aminamazlin/onestep-diffusionpolicy) | 已读 `onedp.py`、训练器、DDPM/EDM scheduler、数据读取 | 可核对 OneDP-S/D 分支和交替训练。个人复现，无本轮运行证据；其训练器冻结视觉编码器，而原论文训练编码器，不能称完全复现 |
| [alexander-soare/consistency_policy](https://github.com/alexander-soare/consistency_policy) | 已读 README、CM 和 sampler | 可参考 NFE/成功率/耗时曲线及少步对照。它是独立个人实验，不是上述 RSS 论文官方仓库；target/EMA 选择也不同 |
| [OFP](https://arxiv.org/html/2603.12480v1) | 已读 π0.5 四个 RoboTwin 任务实验与训练说明 | 提供“在 π0.5 流模型上做一步策略”已有研究依据；其 warm start、自引导等不加入本轮两个 baseline |

个人 OneDP 的具体限制：README 仍提示配置待补；训练器显式冻结视觉编码器；保存函数主要保存学生和辅助网络，不能替代我们的完整续训保存；EDM 原始输出与噪声/score 的转换须重新核对。主实现应依据论文目标和本地接口，不照抄这个训练器。

## 已读代码锚点

- CP：`eed0c465d3c1aa6dcf22f10e784df07dec06beac`。[compute_loss](https://github.com/Aaditya-Prasad/consistency-policy/blob/eed0c465d3c1aa6dcf22f10e784df07dec06beac/consistency_policy/student/ctm_policy.py#L404)；[CTM 映射](https://github.com/Aaditya-Prasad/consistency-policy/blob/eed0c465d3c1aa6dcf22f10e784df07dec06beac/consistency_policy/diffusion.py#L549)；[CTM+DSM 配置](https://github.com/Aaditya-Prasad/consistency-policy/blob/eed0c465d3c1aa6dcf22f10e784df07dec06beac/configs/ctmp_square.yaml#L82)。
- 个人 OneDP：`37002bebdec5cfe9df515c824f33e0a851fce677`，[模型与损失](https://github.com/aminamazlin/onestep-diffusionpolicy/blob/37002bebdec5cfe9df515c824f33e0a851fce677/onedp/models/onedp.py)。
- 个人 CM：`e0d3bcbe87c943c35b7d3612b3d3bde7c5ae8e63`。
- CleanDiffuser：`05f17fc9dbeae7c19a5e264632c9ae9aaac5994e`，[一致性模型实现](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/05f17fc9dbeae7c19a5e264632c9ae9aaac5994e/cleandiffuser/diffusion/consistency_model.py#L264)。
- ManiCM：`f45a61932bf79c9225fbc4906ce8c41b57d213ee`。
- FastGen：`f7d845629b00c75527fd04d8ce66392c363b1fbb`，[flow 转换](https://github.com/NVlabs/FastGen/blob/f7d845629b00c75527fd04d8ce66392c363b1fbb/fastgen/networks/noise_schedule.py#L626)；[交替训练](https://github.com/NVlabs/FastGen/blob/f7d845629b00c75527fd04d8ce66392c363b1fbb/fastgen/methods/distribution_matching/dmd2.py#L202)。

OneDP 已读 ICML 最终 PDF 的 §2、Algorithm 1、附录 D/E，而非只依据摘要。最终论文的学生/辅助学习率为 1e-6/2e-5，Adam β=(0,0.999)，DDPM 蒸馏时刻范围 [2,95]、生成输入时刻 65；这些离散时刻不能当成 π0.5 的 [0,1] 时间直接搬。

正式实现保留搬运代码的原许可/署名；没有确认可复用许可的个人代码仅作为结构参考。
