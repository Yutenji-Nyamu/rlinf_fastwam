# 100-step正式训练：Global Step 39现场分析与梯度强度讨论

最后刷新：2026-08-21 16:31（Asia/Shanghai）  
现场状态：formal driver仍在运行；已完整写出Global Step 39/100。本文件是运行中快照，不改变训练。

## 1. 先给结论

截至g39，DVAC训练**正常、稳定、确实进入了反向传播，但相对历史GRPO的效果不明显**：没有持续领先，
也不能从当前历史对照断言它导致了下降。g1仍是`w=1`的warmup，两次run在DVAC生效前就相差
`-6.25 pp`，说明后续约1–2个百分点的间隔包含自然run差异。

训练rollout success对照：

| 窗口 | DVAC | 历史GRPO | 差值 |
|---|---:|---:|---:|
| g1–10 | 79.80% | 84.92% | -5.12 pp |
| g11–20 | 84.06% | 84.57% | -0.51 pp |
| g21–30 | 88.28% | 88.09% | +0.20 pp |
| g31–39 | 90.45% | 92.80% | -2.34 pp |
| g30–39 | 90.55% | 92.30% | -1.76 pp |
| g1–39 | 85.53% | 87.46% | -1.93 pp |
| g39 | 89.84% | 91.02% | -1.17 pp |

这里的success是on-policy训练rollout，不是held-out fixed-ID评估。历史GRPO后段均值仍从g40–59的
92.09%升到g80–100的95.98%，所以g39不是终点结论。

## 2. 各张图怎样读

### 2.1 `BASELINE_COMPARISON.png`

- **Rollout success**：两条单步曲线频繁交叉；看10-step mean仍没有持续正间隔。
- **Approx KL**：g1–39均值`0.0422 vs 0.0462`。当前方法没有表现为整体更激进的policy移动。
- **PPO clip fraction**：均值`0.1439 vs 0.1506`，整体接近；g39的`0.241 vs 0.193`只是单步偏高。
- **Pre-clip gradient norm**：`34.44 vs 31.60`。两者每一步都远高于`clip_grad=1`，所以它不等于
  optimizer最终步长更大。
- **Step time**：全39步平均每步多约11秒；g20–39反而快约1.65秒，没有持续可见的DVAC耗时税。

### 2.2 `DVAC_FORMAL_STEP39_METHOD_OVERVIEW.png`

- `log V_L3`从g1约`-4.413`升到g39约`-4.088`，对应几何均值约`1.38×`；recent-5参考线会滞后追随。
- 有效query的权重p05/median/p95跨apply步骤平均约`0.881/1.016/1.197`；边界比`1.2/0.8=1.5×`。
- 上界`1.2`平均命中`5.53%`，下界`0.8`仅`0.224%`。当前实际行为更像“少量高V明显增权”，
  而不是大量低V被压低。
- 后25个future action比前25个平均高`0.0382`；负advantage query比正advantage平均高`0.0278`。
  因此当前权重主要捕获了chunk尾部趋势和loss-mask/polarity结构，不等同于已定位任务关键动作。

## 3. 梯度到底怎样被改

旧GRPO把50个future action的log-prob相加后做一次joint chunk ratio和PPO clip。新挂点保持这个forward
数值不变，只把反向中的每个future-action贡献改为：

\[
g_q=c_q\sum_h w_{qh}\sum_d\nabla_\theta\ell_{qhd}.
\]

其中`c_q`含GRPO advantage、ratio和loss聚合。advantage仍决定强化或抑制；DVAC只决定chunk内部哪个
`h`更用力。若整个chunk已经落到PPO无梯度的clipped plateau，50个`h`仍一起为零。

FSDP最后对全模型梯度做：

\[
g_{used}=g\min(1,1/\lVert g\rVert_2).
\]

当前所有pre-clip norm都大于21.8，所以共同乘一个常数大多会被全局clip抵消；不同`h`的相对权重仍会
改变向量和的方向。当前并没有显式per-query mean-one；all-query均值接近1是rolling z-score的统计结果，
不是权重归一化。

## 4. 当前强度是否太小

它相对hard 80/20确实很温和。对g2–39共14,551个有效query做离线重映射：

| 映射 | p05 / median / p95 | weight ESS | 等长正交方向proxy | 后25−前25 |
|---|---:|---:|---:|---:|
| 当前连续`a=.1` | .881 / 1.016 / 1.200 | .992 | 5.1° | +.039 |
| 连续`a=.2` | .763 / 1.031 / 1.400 | .970 | 9.9° | +.077 |
| 连续`a=.3` | .644 / 1.047 / 1.600 | .938 | 14.4° | +.116 |
| hard top-20%，`{0,5}` | 80%为0 | .200 | 63.4° | +.589 |

ESS这里只衡量权重集中度，不是实测policy-gradient ESS；方向角也只是比较强弱的proxy。它们说明：
当前`a=.1`更像链路验证和小扰动版；`a=.2`能大约翻倍相对方向压力，仍远弱于hard mask。只向上增权
并不更有效，因为公共放大会被global clip消掉；若要增强方向变化，下压低V位置通常更直接。

## 5. 相关工作实际用了多大幅度

- **Beyond 80/20 / Fork Tokens**：仅保留top-20%高熵token，低80%梯度为0；loss除以入选token数，
  相对full-token mean可理解为有效权重约`{0,5}`。论文与官方代码：
  [论文](https://openreview.net/forum?id=yfcpdY4gMP)、
  [mask实现](https://github.com/Shenzhi-Wang/Beyond-the-80-20-Rule-RLVR/blob/main/verl/trainer/ppo/core_algos.py#L49-L81)。
- **STEER**：连续抑制预测entropy change过大的token。论文主设置`lambda_min=0.7`，稳定区间约
  `0.5–0.8`；linear消融使用`[0.7,1.2]`，binary top-20%反而更差。
  [论文](https://aclanthology.org/2026.acl-long.1436/)、
  [官方代码](https://github.com/zz-haooo/STEER)。
- **A3PO**：正rollout的低概率20%和负rollout的高概率20%在早期`×2`，再随训练衰减到1；探索性
  实验也比较了`0.2/0.5/2/5×`。其重点是`outcome polarity × confidence`，不是统一放大高不确定位置。
  [论文](https://aclanthology.org/2026.acl-long.134/)、
  [官方代码](https://github.com/RUCAIBox/Sample-Polarity-of-RLVR)。
- **HTMR / LESS / ResT**进一步说明：低熵位置里可能有必须保留的任务控制字段、稳定好套路和稳定坏套路；
  不能把一维uncertainty直接等同于credit方向。

因此当前`[0.8,1.2]`与STEER温和连续方案同量级，不是异常小；它只是远弱于Fork的hard选择和A3PO的
选中位置`×2`。文献也没有统一规则说“不确定越高就一定越增权”。

## 6. 当前建议

1. 正在运行的100-step保持不变，完成后看整条训练曲线和同fixed-ID评估。
2. 如果下一次只选一个更强但仍干净的候选，优先`a=.2`：同一公式、权重约`[0.6,1.4]`，而不是直接
   跳到80%归零。
3. 下一版比单纯继续加大倍率更有信息量的改进是记录少量`weighted vs off`真实梯度夹角，并把
   `future-h`位置趋势、advantage polarity与DVAC residual分开观察。

## 7. 证据入口

- `evidence/formal_live_step39_20260821/analysis/BASELINE_COMPARISON.png`
- `evidence/formal_live_step39_20260821/analysis/DVAC_FORMAL_STEP39_METHOD_OVERVIEW.png`
- `evidence/formal_live_step39_20260821/analysis/DVAC_WEIGHT_COUNTERFACTUAL.png`
- `evidence/formal_live_step39_20260821/analysis/GRADIENT_MECHANICS_AND_COUNTERFACTUAL.md`
- `evidence/formal_live_step39_20260821/analysis/STEP39_EFFECT_SUMMARY.json`
- `evidence/formal_live_step39_20260821/analysis/DVAC_WEIGHT_COUNTERFACTUAL_SUMMARY.json`
