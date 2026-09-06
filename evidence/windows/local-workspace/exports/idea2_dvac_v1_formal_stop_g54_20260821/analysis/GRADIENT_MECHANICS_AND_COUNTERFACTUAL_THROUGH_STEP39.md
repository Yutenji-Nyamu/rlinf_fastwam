# DVAC-GRPO梯度语义与Step 39离线反事实

数据范围：Global Step 2–39的两rank apply shard；`14,551`个loss-mask有效query、`727,550`个
future-action位置。Global Step 39单步有242个有效query。分析只重放已记录的`V_L3`和历史统计，
没有重跑actor forward/backward，也没有改正在运行的训练。

## 1. 当前代码实际怎样改梯度

权威source是`145fa810...`：

- `rlinf/algorithms/dvac_train_weighting.py::straight_through_scale_logprobs`实现
  `stopgrad(l)+w*(l-stopgrad(l))`；
- `rlinf/workers/actor/fsdp_actor_worker.py::train_micro_batch`在`[B,H,D]`新log-prob进入原loss前调用它；
- `rlinf/algorithms/utils.py::preprocess_loss_inputs`随后仍按`chunk_level`对`H,D`全部求和；
- `rlinf/algorithms/losses.py::compute_ppo_actor_loss`仍做一次chunk ratio和一次chunk PPO clip；
- `rlinf/hybrid_engines/fsdp/strategy/fsdp.py::clip_grad_norm_`最后计算跨FSDP rank的整体L2范数，并以同一个
  系数把全部参数梯度缩到`clip_grad=1`。

对一个未进入PPO clipped plateau的query，可把当前policy-gradient写成：

\[
g_q=c_q\sum_{h=0}^{49} w_{qh}\sum_d\nabla_\theta \ell_{qhd},
\]

其中`c_q`包含GRPO advantage、ratio和loss聚合。由此可直接得到：

1. forward数值完全不变，所以在同一参数快照上，chunk ratio、PPO clip分支和loss显示值不变；
2. backward时每个future action的likelihood贡献乘`w[h]`；
3. advantage仍决定强化或抑制，高V只决定同一方向上哪里更用力；
4. 一个chunk若进入无policy-gradient的clipped plateau，50个`h`仍一起为零；当前挂点不会单独解锁某个`h`。

## 2. Global grad clip是不是“最后归一化，所以只改方向”

当前权重**没有**做per-query mean-one归一化。Step 2–39全部位置的weight均值是`1.0222`，每个query的
均值p05/p95为`0.9304/1.1184`；Step 39均值是`1.0262`。所以它既有chunk内`h`重分配，也有小幅
query-to-query总音量变化。

global grad clip发生在所有query/microbatch梯度相加之后：

\[
g_{final}=g\min(1,1/\lVert g\rVert_2).
\]

本run报告的pre-clip actor norm是几十，远高于1，因此实际更新的整体L2尺度主要被压回1。这个条件下：

- 若所有weight共同乘同一个正数，乘法会被global clip完全抵消；
- 非均匀weight会改变不同`g_{qh}`的向量和，因此方向变化仍保留；
- global clip不是per-query weight归一化，不能消除query间或`h`间的相对差异；
- 若未来某次pre-clip norm不超过1，weight对整体幅度的影响也会保留。

所以“在当前run里主要改方向”是对的；但“weight已经被归一成均值1”是不对的。

另一个直接推论是：只增权、不下压往往没有看起来那么强。它增加的公共尺度很大一部分会被global clip
消掉；真正留下的是高V相对低V的差。若目标是更明显地改变方向，下压低V通常比把所有低V留在1、只把
高V抬高更有效。

## 3. 反事实幅度

这里的`a`对应`w=1+a*clip(z,-2,2)`。ESS fraction定义为
`(sum w)^2/(N*sum w^2)`：1代表接近均匀，0.2代表等权只保留20%。它只是weight集中度，不是实测
policy-gradient ESS。`angle proxy`进一步假设每个`h`的梯度等长且彼此正交，因此只用于比较相对强弱，
不等同于真实梯度夹角。

| 映射 | 范围/关键分位 | mean | ESS fraction | angle proxy | 后25−前25 |
|---|---:|---:|---:|---:|---:|
| 当前 symmetric `a=.1` | p05/median/p95=`.881/1.016/1.200` | 1.022 | .992 | 5.1° | +.039 |
| symmetric `a=.2` | `.763/1.031/1.400` | 1.044 | .970 | 9.9° | +.077 |
| symmetric `a=.3` | `.644/1.047/1.600` | 1.067 | .938 | 14.4° | +.116 |
| only-up `a=.3` | `1.000/1.047/1.600` | 1.146 | .974 | 9.3° | +.073 |
| per-query mean-one symmetric `a=.3` | `.658/1.006/1.333` | 1.000 | .961 | 11.4° | +.105 |
| bounded hard top20：top=`1.2`、其余=`.95` | `.95/.95/1.2` | 1.000 | .990 | 5.7° | +.029 |
| hard top20 raw：`{0,1}` | 80%为0 | .200 | .200 | 63.4° | +.118 |
| hard top20 mean-one：`{0,5}` | 80%为0 | 1.000 | .200 | 63.4° | +.589 |

当前映射确实很温和：保留了约`99.2%`的weight ESS，proxy离uniform只有约`5°`。把`a`提高到`.2/.3`
会把相对对比约翻倍/三倍，但依然远弱于top-20% hard mask。当前所有apply数据中只有`0.22%`位置触及
0.8下界，而`5.54%`触及1.2上界；多数所谓“低V位置”仍保留了接近完整的gradient credit。

“hard top20”本身不保证强：若仍限制为top=`1.2`、其余=`.95`，ESS仍是`.990`，几乎与当前相同。
Beyond-80/20式强选择来自**80%直接变0**，不是来自top-20这个排序动作本身。

## 4. `{0,1}`与`{0,5}`在当前global clip下是什么关系

若两种方案选择完全相同的20%位置，则`{0,5}`的pre-clip gradient恰好是`{0,1}`的5倍。只要global
clip确实触发，二者缩放后方向与L2范数相同；区别只剩数值精度、其他未同比例缩放的loss项，以及某次
没有触发global clip的情况。当前entropy/KL coefficient为0，因此首要语义就是相同方向。

不过它仍不完全等同Beyond-80/20的token-level PPO：我们的straight-through hard mask若放在当前挂点，
forward chunk ratio仍由全部50个action算出，全部位置仍共同决定PPO clip gate；只是gate放行后，80%的
backward likelihood贡献为0。

## 5. 从这份数据能支持的选择

- 当前`.1`可合理称为“链路验证和小扰动版”，不能称为接近80/20强度；
- 若以后需要一个仍连续、但能更明显改变方向的单一候选，`.2`对应`[.6,1.4]`、ESS约`.970`，比直接
  hard mask温和很多；`.3`对应`[.4,1.6]`、ESS约`.938`；
- 只增权方案不是更强的替代：`a=.3`虽然mean升到1.146，但global clip会消掉大部分公共放大，其weight
  concentration反而弱于symmetric `.3`；
- naive `w/mean_h(w)`确实让每个query均值为1，但会破坏原边界：`.1`已扩到`.682..1.340`，`.3`扩到
  `.266..2.352`。若以后既要mean-one又要硬边界，需要单独设计有界映射，不能简单“先clip再除均值”；
- 现在看不出效果明显，不足以单独证明`.1`太小；但离线幅度明确显示：若信号有用，当前机制对梯度方向
  施加的压力确实很轻。最终性能判断仍要结合完成后的fixed-ID eval。

对应产物：`DVAC_WEIGHT_COUNTERFACTUAL.png`、`DVAC_WEIGHT_COUNTERFACTUAL.csv`、
`DVAC_WEIGHT_COUNTERFACTUAL_BY_H.csv`和`DVAC_WEIGHT_COUNTERFACTUAL_SUMMARY.json`。
