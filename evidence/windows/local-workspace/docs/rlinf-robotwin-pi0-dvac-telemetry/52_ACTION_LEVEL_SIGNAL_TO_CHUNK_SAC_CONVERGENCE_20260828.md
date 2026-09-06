# Action-level signal 怎样进入 chunk-SAC：本轮收敛

日期：2026-08-28  
范围：方法讨论，不改代码、不停止当前 Pure 训练。

## 1. 现在已经理解的目标

想要的是一个能够插进现有 RLT/SAC 的小方法：

```text
现有 actor 输出 C10 action chunk
        +
外部信号 u(q,h)，每个 future action 一个数
        ↓
chunk 内十个 action 获得不同训练力度
```

同时尽量保留：

- 现有 replay、scalar twin-Q、TD target、`-Q + BC` actor 骨架；
- 信号的 action-time 粒度，而不是最后压成一个 query/chunk 标量；
- 信号只决定“哪里多学”，训练中原有的 reward/Q/outcome 决定“往哪里学”。

不想要的是：为了使用 DVAC 重做整套 per-h critic；把 PRM 当最终方法；或者因为 teacher 高 DVAC 就直接断言 teacher target 更值得模仿。

## 2. clean RLT 里现有的四个训练接口

```text
replay: (s, executed C10, reward[10], s')
                       │
                       ├─ reward[10]折扣求和 ─→ scalar TD target ─→ twin-Q critic
                       │
student(s) ─→ Aθ [10,14]
                       ├─ -Q(s,Aθ)                  # RL方向
π0 reference [10,14] ─┴─ MSE(Aθ,Aref)              # teacher锚
```

代码中 primitive reward 先变成

$$
R_q=\sum_{h=0}^{9}\gamma^h r_{q,h},
$$

再形成 scalar critic target；actor 则是

$$
L_{actor}=-\lambda_Q Q(s,A_\theta)+\lambda_{BC}L_{BC}.
$$

对应本地源码副本为 `fsdp_rlt_ac_policy_worker.py:132-140,386-392,417-500`。

因此外部信号可进入四处：

1. reward/replay：改变 critic 学到“什么行为有价值”；
2. actor 的可分解监督项：改变 C10 内哪些位置被更强拟合；
3. actor exploration/entropy：改变 C10 内哪些位置探索更多；
4. critic 结构：训练 prefix/per-h value，直接建立细粒度 outcome credit。

## 3. PRM 到底怎样进入 PPO、SAC 和 actor loss

### 3.1 DenseReward：两个实验都只改 reward

仿真中，π0 每次执行 `C=5`，DenseReward 给一个 `r_model∈[0,1]`，只放到该 chunk 最后一个 primitive step：

$$
r_t=r_t^{sim}+\frac{C}{T_{max}}r_t^{model}.
$$

RLinf PPO 的 ratio、clip、actor/value loss都不改；新 reward 经过 return/advantage 后影响策略。

真机中冻结 π0，DSRL-SAC 原非终止奖励是 `-1`，成功终点是 `0`；DenseReward 改成：

$$
r_t=-1+r_t^{model},\qquad r_T=r_T^{model}.
$$

仍不改 SAC actor、critic 或 latent steering 结构；只改写 replay reward，随后 TD target、Q 和 actor 间接受影响。

来源：[DenseReward](https://arxiv.org/html/2607.13033v1)。

### 3.2 Robometer 与 LLM-as-a-Verifier：rollout 后重标 replay

- Robometer 异步给轨迹每个时间点预测 progress，并在 rollout 已进入 replay 后重标 reward；success head 还负责终止。SAC 公式不改。[Robometer](https://arxiv.org/html/2603.02115)
- LLM-as-a-Verifier 对 rollout prefix 得到 `ρ_t`，使用

$$
r'_t=r_t^{env}+\lambda\rho_t,
$$

再把重标后的 transition 放进 replay，critic 学新 return。[LLM-as-a-Verifier](https://llm-as-a-verifier.com/)

类比到 clean RLT：若有十个 primitive progress，就改 `reward[0:10]`；若每个 C10 只有一个分数，就像 DenseReward 一样放到最后一格。随后仍会折扣求和成一个 chunk return，因此它细化的是**时间奖励来源**，不是直接产生十个 actor 梯度权重。

### 3.3 SARM、GVL、TOPReward：不经过 SAC critic，直接加权监督

- SARM 用相隔一个 chunk 的 progress difference，经过 running mean/std 映射成 `[0,1]`，乘整条 demonstration 的 BC loss；它是 sample/chunk weight，不是 future-h weight。[SARM](https://arxiv.org/html/2509.25358)
- GVL 用逐 transition 的 value difference：

$$
\exp\{\tau(v_{k+1}-v_k)\}\,[-\log\pi(a_k|o_k)],
$$

也就是 progress 差同时给出好坏方向和模仿强度。[GVL](https://arxiv.org/html/2411.04549)
- TOPReward 用相邻 prefix score increment 形成正权重：

$$
\Delta_t=\min\{\exp[\beta(r_t-r_{t-1})],\delta_{max}\},
$$

再乘 flow-matching MSE；下降时降权，而不是把 MSE 变成负数。[TOPReward](https://arxiv.org/html/2602.19313)

这三篇教的是“有方向的质量信号可以直接决定哪个监督 target 值得学”，不是 SAC reward relabel。

## 4. 为什么 PRM 切口不能原样套到 DVAC

PRM/progress 的语义是有方向的：

```text
progress上升  → 更接近任务完成 → 正向训练信号
progress下降  → 退步             → 降权或负向reward
```

DVAC 的语义是：π0 在最后几次去噪中，对某个 future action 的 endpoint 改口多大。它只有“稳定/不稳定”，没有自动给出“更好/更坏”。

所以不能直接写成 `reward += λ·DVAC`；那会让 critic 学成“越不确定，回报越高”。同样，“高 DVAC → 更强复制 π0 reference”也只有在额外假设“高 DVAC 是重要且 teacher 最终答案可信”时才成立。

SAC replay reward 也不等于一般意义的“进度标签”；它是任务效用。PRM 恰好能把视觉进度翻译成任务效用，DVAC 目前还不能。

## 5. 23篇工作的真正挂点

| 类别 | 工作 | 信号进入哪里 | 对本项目的意义 |
|---|---|---|---|
| reward | [DenseReward 2026](https://arxiv.org/html/2607.13033v1) | chunk progress→PPO/DSRL reward | 方向信号可保持原RL算法 |
| reward | [Robometer RSS 2026](https://arxiv.org/html/2603.02115) | per-frame progress→replay reward | 异步重标，不改SAC公式 |
| reward | [LLM-as-Verifier 2026](https://llm-as-a-verifier.com/) | prefix progress→replay reward | clean RLT可直接类比reward[10] |
| reward | [TimeRewarder 2025](https://arxiv.org/abs/2509.26627) | frame temporal distance→step proxy reward | progress difference给出逐步方向 |
| reward | [Robo-Dopamine CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Tan_General_Process_Reward_Modeling_for_Robotic_Reinforcement_Learning_CVPR_2026_paper.pdf) | progress potential→policy-invariant shaping | 避免直接累加绝对progress造成停留偏好 |
| reward | [RoboReward 2026](https://arxiv.org/abs/2601.00675) | episode progress score→DSRL reward | 仍是chunk/episode utility，不是per-h actor credit |
| actor监督 | [SARM 2025](https://arxiv.org/html/2509.25358) | progress delta→BC sample weight | target质量决定模仿强度 |
| actor监督 | [GVL ICLR 2025](https://arxiv.org/html/2411.04549) | value difference→AWR likelihood | transition级、有方向的weight |
| actor监督 | [TOPReward 2026](https://arxiv.org/html/2602.19313) | prefix increment→flow MSE weight | 正权重、下降只降权 |
| actor监督 | [DAM-VLA 2026](https://arxiv.org/abs/2603.00926) | phase/gripper/horizon prior→per-h diffusion loss | 真正chunk内逐h监督先例 |
| actor监督 | [Set-Supervised DP RSS 2026](https://arxiv.org/abs/2606.01865) | 错误chunk+纠正chunk→desired-action set | 先给正确方向，再训练chunk |
| actor数据门 | [AC3 AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38937) | actor只从成功轨迹更新；critic使用全部经验 | “谁值得给actor学”可与critic数据分开 |
| SAC探索 | [ACE ICML 2024 Oral](https://proceedings.mlr.press/v235/ji24b.html) | action维对reward的因果影响→逐维entropy | scalar Q不动，外部逐动作重要性控制探索 |
| SAC探索 | [CIP ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/baa41b7368408670c6a14e06a04420d9-Abstract-Conference.html) | action-reward因果矩阵→empowerment辅助项 | 重要性进入actor探索，不改Q credit |
| flow-RL | [DACER NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6174c67b136621f3f2e4a6b1d3286f6b-Abstract-Conference.html) | diffusion entropy→temperature/noise | uncertainty更自然地对应探索 |
| flow-RL | [DIME ICML 2025](https://proceedings.mlr.press/v267/celik25a.html) | diffusion entropy→MaxEnt actor | 仍是分布探索，不是teacher BC |
| Q→actor | [DIPO](https://arxiv.org/abs/2305.13122) | `∇_aQ`先改action target，再做diffusion regression | Q给方向，监督项负责学习 |
| Q→actor | [Q-Score Matching ICML 2024](https://proceedings.mlr.press/v235/psenka24a.html) | policy score对齐`∇_aQ` | scalar Q仍可给action-shaped方向 |
| Q→actor | [FQL ICML 2025](https://proceedings.mlr.press/v267/park25f.html) | `-Q(student)+flow-teacher distillation` | 与RLT骨架最接近 |
| Q→actor | [GFP ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33f131b806a93d376cf0ce1a456464bb-Abstract-Conference.html) | target/student的Q差→flow BC weight | Q先判断target值不值得学 |
| chunk critic | [Q-Chunking NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/50348e8f9aef984abe0ea1ec2a326f78-Abstract-Conference.html) | full chunk→scalar Q+n-step TD | 说明scalar chunk-Q是合理主流前提 |
| prefix critic | [Decoupled Q-Chunking ICLR 2026](https://iclr.cc/virtual/2026/poster/10008665) / [CQN-AS NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/9eecce65f1f93d7a3e7ae677c31942ab-Paper-Conference.pdf) | partial chunk或每序列位置有独立Q | 真正per-h value需要结构升级 |

## 6. 本轮真正收敛出的两个候选

### 候选A：高 DVAC 释放 teacher BC，让原 Q 更能改这个位置

保留 clean RLT 的 `-Q`，只把 reference-BC 改成：

$$
L_{BC}=\frac1{10}\sum_h c_h^{trust}\|A_{\theta,h}-A_{ref,h}\|^2,
$$

其中 `DVAC高 → c_h^{trust}低`，并在 C10 内 mean-one。

通俗解释：teacher 自己在该位置更不稳定，就少用 teacher 锚住 student；现有 Q 梯度自然拥有更大的相对话语权。它不是“少学这个 action”，而是“少跟 teacher 学、更多交给 RL 学”。

优点：只改可分解 BC；reward、critic、TD、`-Q`、target 都不变。它比当前 Pure 的方向更符合“不确定性→置信度”的文献主线。

### 候选B：Q 先产生改进动作，DVAC 再决定哪些 h 多学

先用 scalar Q 的 action gradient 构造带方向的 target：

$$
A^+=\operatorname{sg}\left(A_\theta+\eta\nabla_AQ(s,A_\theta)\right),
$$

再增加逐 h 辅助项：

$$
L_{aux}=\frac1{10}\sum_h w_h^{DVAC}\|A_{\theta,h}-A_h^+\|^2.
$$

这里 Q 回答“往哪学”，DVAC 回答“哪里多投入”。这是把 DIPO/Q-Score Matching 的 Q 方向与 chunk 内 weighted actor loss 结合起来；仍保留原 scalar critic 和原 `-Q` 路径。

它比候选A多一次 `∇_AQ` target 构造，但最贴近最初的高层语义：**外部 action-level 重要性只分配训练量，RL 本身给出方向。**

### 最统一的合并形态：DVAC负责teacher-vs-RL路由

两个候选还可以共用同一个逐 h gate：

$$
L_h=c_h^{trust}\|A_{\theta,h}-A_{ref,h}\|^2
 +(1-c_h^{trust})\|A_{\theta,h}-A_h^+\|^2,
$$

其中 `DVAC高 → c_h^{trust}低`：

- 稳定位置更多保留 π0 reference prior；
- 不稳定位置减少复制 teacher，更多学习 Q 指出的改进方向；
- reward/Q 决定方向，DVAC只决定两种学习来源的相对份额。

没有一篇论文原样给出这条公式；它是 FQL 的“Q + teacher distillation”骨架、GFP/ACTIVE 的“质量或置信度控制模仿”、DIPO/Q-Score Matching 的“Q提供action方向”，以及逐 h actor-loss工作的最小组合。若只选一个后续主线，这个组合最贴近当前目标。

## 7. 不再作为新候选的旧路线

旧 RLT-DVAC 第一版把

$$
\nabla_AQ\rightarrow W\nabla_AQ
$$

直接逐 h 缩放。最近讨论中再次出现的 straight-through `A_tilde=sg(A)+W(A-sg(A))` 与它计算完全相同；不是新的挂点，只是换了解释。

低层计算能运行，但它用外部 DVAC 直接旋转 scalar-Q 的联合 action 上坡方向；DVAC没有给出旋转方向正确的证据，因此不再列为首选。

## 8. 当前 Pure 训练对方法讨论的提示

20:01 CST共同 Step 268：

- `s0p5`：5步/10步=`82.5/87.5%`，Step250 fixed20=`19/20`，weight ESS=`.912`；
- `s2p0`：5步/10步=`80.0/86.25%`，Step250 fixed20=`15/20`，weight ESS=`.589`；
- 两条健康运行，OOM/OOM-kill=0；预计分别还约`14:57/15:33`。

目前较强的高-DVAC reference-BC 没有显示优势。这不是最终效果结论，但足以说明下一轮方法设计不应只是继续扩大同一方向。

现场材料：[Step268证据](evidence/rlt_dvac_pure_dual_live_g266_20260828/README.md)。
