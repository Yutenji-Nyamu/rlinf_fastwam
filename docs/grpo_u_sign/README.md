# GRPO U按成败方向加权

2026-10-19。用户确认沿用现有U，只改变加权方向；本轮实现、完整一轮smoke、公开源码与轻量证据，不启动长期训练。

## 两版

| 配置 | 正优势（成功） | 负优势（失败） |
|---|---|---|
| `sign-signed-overlay.json` | 高U权重大 | 低U权重大 |
| `sign-positive-overlay.json` | 高U权重大 | 原权重1 |

共同设置 `direction=advantage_sign`；两版 `scope` 分别为 `both`、`positive`。U仍是同一输入、同一初始噪声下完整ODE10/ODE5动作的差异；训练主采样仍用原Flow-SDE0.5。没有用概率熵替换U，没有改主动作、采样RNG、优势、PPO裁剪或优化器。

## 加权与边界

沿用两层：块内50动作，以及同场景组中的动作块。先对log(U+eps)做原有MinMax，再映射 `exp(z/τ)`；负优势映射 `exp(-z/τ)`。块内均值归一为1；块间按实际loss贡献归一为1。正、负优势分别归一化，避免反转失败方向时整体缩放成功部分；因此两版成功权重完全相同。最后仍为两层因子相乘，不再额外归一。

`τ_local/τ_chunk`、alpha、chunk dropout、alpha退火均保留。τ越小，分配越集中。默认 `direction=increasing` 与旧算法及旧恢复契约保持一致；新方向进入恢复契约，防止错用旧状态。

这里的成功/失败按GRPO优势符号决定。二值任务混合组中分别对应成功/失败；全对、全错及无效组仍受原过滤，零优势仍为零。本次没有加入RL-ZVP的全同结果组替代优势。

参考 [EAPO §3](https://arxiv.org/html/2609.33781v1)：按优势符号决定熵权重方向，并保持每条回答平均权重为1。本实现采用该方向想法，保留自己的U及两层分组归一化；不等同于论文的token熵或逐回答算法。

## 源码与验收入口

- 基线提交 `19a66c1a9`；仅在 `dvac_two_level.py` 增加可选方向和正负分池，actor透传并保存契约。
- `tests/unit_tests/test_grpo_u_sign.py` 验证方向、同成功权重、各符号loss质量守恒、零优势、mask、τ/alpha/dropout和实际actor恢复。
- `tools/grpo_un/prepare_sign_smoke.py` 复制已有一轮smoke准备器，继承10月14日pillbottle实配；`driver.py/common.py` 复用原清场逻辑。
- [执行记录](EXECUTION.md)保留命令、资源、停止条件与结果。
