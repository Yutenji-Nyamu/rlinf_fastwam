# RLT 中 teacher-DVAC 的语义与替代切口

日期：2026-08-28  
状态：方法讨论；当前 `s0p5/s2p0` 两条 formal 继续运行，没有修改或停止。

## 1. 先给结论

你的疑问成立。当前 Pure 方法做的是：

```text
成功 episode
  + frozen π0 在某个 future h 的 DVAC 较高
  -> student 在这个 h 更强地模仿同一个 π0 reference action
```

它只有一种正向解释：**高 DVAC 表示困难、精细或值得投入更多学习量的位置**。它不能解释成“teacher 越
不确定，teacher 给出的单点动作越可信”。如果把 DVAC 理解为 teacher 的置信度，BC 方向应当相反：
高 DVAC 少复制，低 DVAC 多复制。

更清楚的分工是：

```text
success / Q / advantage  -> 这个 target 值不值得学
teacher DVAC             -> teacher 在这个 h 的单点 target 有多稳定
BC                        -> student 多大程度跟 teacher
Q                         -> student 朝更高价值动作改进
```

## 2. 当前 Pure 到底改了哪里

原 RLT student actor loss 可通俗写成：

$$
L_{actor}=-\lambda_Q Q(s,a^{student})
+\lambda_{BC}\operatorname{mean}_{h,d}
\left(a^{student}_{h,d}-a^{\pi_0}_{h,d}\right)^2.
$$

- `-Q`：让 student 的整段 C10 朝 critic 判断更好的方向移动。
- `BC`：让 student 不要离 frozen π0 reference 太远。
- Pure：`-Q` 不动，只在成功 episode 把十个 BC 位置从均匀改成 mean-one DVAC 权重。

因此 success 只能说明“这条访问轨迹最终成功”；它不能单独说明 π0 在某个高 DVAC 的 $h$ 上给出的那个
reference 点更可靠。当前实验实际检验的是“成功轨迹的困难位置是否值得更强 reference-BC”，而不是
“不确定 teacher 是否更可信”。

这也解释了为什么把 GRPO 版原样迁到 RLT 会变味：GRPO 中 DVAC 只是重新分配同一个 π0 policy-gradient
的 credit，reward/advantage仍决定强化或抑制；RLT Pure 则把 DVAC 变成 student 朝 frozen teacher 单点
靠近的 MSE 强度，因而同时引入了“这个 teacher target 是否值得相信”的问题。

## 3. DVAC 可能同时包含的三种含义

1. **预测不稳定**：最后几次 clean endpoint 仍在改口。这时更自然的做法是降低单点 teacher-BC。
2. **精细或接触阶段**：DVAC 论文观察到 contact-rich / precision-sensitive 阶段较高。这时它可能是
   attention 信号，但“阶段重要”还不等于“这个 reference 点更值得复制”。
3. **多种合理动作**：抓取或接触可能存在多个可行动作 mode。此时单点 MSE 无论增权或降权都不完整，
   更合适的是多样本或集合 target。

原 [DVAC](https://arxiv.org/html/2606.03847) 对高方差 future action 的处理是提前停止并重新规划；
它没有把高 DVAC 用作更强模仿该单点动作的依据。

## 4. 六条更清楚的结合路线

### 4.1 Confidence-DVAC BC：最简洁

保持原 π0 target、原 Q、原 replay 和 schedule，只反转 DVAC 到 BC 的方向：

```text
低 DVAC / teacher 稳定  -> BC 较强
高 DVAC / teacher 不稳  -> BC 较弱，给 Q 更多改进空间
```

仍可在每条 C10 内 mean-one，只重分配十格的 BC，不改变平均 Q:BC 比例。这一版最直接回答“DVAC 能否
作为 teacher confidence”。

### 4.2 Quality × confidence：价值与置信度各管一层

令整条 target 的质量门为：

$$
u_q=\sigma\!\left(\frac{Q(s,a^{ref})-Q(s,a^{student})}{T}\right),
$$

再令 $c_{q,h}$ 随 DVAC 增大而减小：

$$
L_{BC}=\operatorname{mean}_{q,h}\left[u_qc_{q,h}e_{q,h}\right].
$$

意思是：reference 比 student 更有价值时才多模仿；决定模仿以后，再少信 teacher 不稳定的位置。
[GFP（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33f131b806a93d376cf0ce1a456464bb-Abstract-Conference.html)
最接近前一层；DVAC补充它没有提供的 C10 内部粒度。

### 4.3 高 DVAC 学成功执行动作，而不是不稳定 reference

如果某个 episode 已成功，高 DVAC 可解释成“这里困难，值得多学”；但 target 改为真正成功执行过的
action，会比继续追同一个不稳定 π0 reference 更顺：

```text
成功 outcome  -> executed action 是有结果支持的 target
高 DVAC       -> 在困难 h 上加强这份成功经验
```

这与此前 success-executed 版本的语义相近；它的问题不是逻辑不通，而是当时同时改了 target 与 DVAC
权重，难以单看 DVAC 的贡献。

### 4.4 多样本 / 集合 teacher：不把多解压成一个点

同一 state 从 π0 采多个 action chunk。高 DVAC 位置让 student 学候选分布、最接近的有效候选，或一个
可接受动作集合，而不是把单个随机 reference 的 MSE 乘大。
[Set-Supervised Diffusion Policy（RSS 2026）](https://arxiv.org/abs/2606.01865) 的核心启发就是：
当监督本身允许多解时，学习目标应从单点变成集合。

### 4.5 DVAC 用于采集，而不是直接改 BC

高 DVAC 状态可以：更频繁重规划、提高 replay 采样概率、额外采一条 π0 候选、请求专家或优先收集新数据。
[Diff-DAgger（ICRA 2025）](https://diffdagger.github.io/)、
[FIPER（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0b7cb3b8cc44e652761245537027db44-Abstract-Conference.html)
和 [UQ-VLA（2026 预印本）](https://arxiv.org/abs/2606.18043) 都更接近“高不确定时获取信息、告警或主动选数据”。

### 4.6 先让 student 可模仿，再谈权重

teacher 可能利用 student 没有的表达能力或产生 student 难以拟合的动作。
[Student-Informed Teacher Training（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a8223b0ad64007423ffb308b0dd92298-Abstract-Conference.html)
通过让 teacher 考虑 student 的可模仿性来缓解这种不对称。对 RLT 的启发是：可增加
`student-reference error`或局部 Jacobian/容量信号；高 DVAC 且 student 长期学不会时，不应只是继续放大 MSE。

## 5. 昨晚相关工作放回同一张图

| 工作 | 它用什么判断“值得学” | 它怎样处理不确定或多解 | 给 RLT 的灵感 |
|---|---|---|---|
| [FQL, ICML 2025](https://proceedings.mlr.press/v267/park25f.html) | critic Q 改进 one-step student | flow teacher MSE 作为支持域锚 | 保留 `-Q + teacher MSE` 骨架，DVAC应改 MSE 语义而非联合 Q 梯度 |
| [GFP, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33f131b806a93d376cf0ce1a456464bb-Abstract-Conference.html) | 高价值 dataset action | 用 Q 权重决定 BC 学多少 | 最适合做 query 级 quality gate |
| [VACO, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/17ab5964d90f38b174a068646c958789-Abstract-Conference.html) | 元网络学习哪条 BC 数据能提高价值 | 不直接用 uncertainty | 说明 BC weight 可以由训练收益学习，而非固定手写 |
| [ACTIVE, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c06f788963f0ce069f5b2dbf83fe7822-Abstract-Conference.html) | 保守 $Q-V$ advantage | V ensemble控制价值可靠性，另调总 BC 强度 | quality、confidence、BC总强度应是不同旋钮 |
| [QVPO, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6111371a868af8dcfba0f96ad9e25ae3-Abstract-Conference.html) | 候选动作 Q | 保留 diffusion 多模态和探索 | 若直接训练生成 actor，应由 Q 选择高质量候选 |
| [AC3, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38937) | 成功轨迹 | actor只从成功轨迹更新，critic用全部经验 | success适合选择actor学习数据，但不自动决定 teacher 置信度 |

共同主线是：**价值、成功、advantage负责增权；不确定性常负责降信任、收集信息或保留多解。** 当前 Pure
选择了另一种可检验的解释——高 DVAC 是困难位置，所以增权——因此当前两条 run 仍有信息量，但结果应按
这个假设解读。

## 6. 为什么叫 `[0,2]` 与 `[0,5]`

正式参数名更准确：

- GPU0：`s0p5`，即 `strength=0.5`；
- GPU1：`s2p0`，即 `strength=2.0`。

简称来自旧的未居中映射。若 $z\in[-2,2]$：

$$
w=1+0.5z\in[0,2],
$$

而 `strength=2` 时 $1+2z\in[-3,5]$，做非负截断后名义为 `[0,5]`。

Pure 的真实公式多了 C10 内居中和 mean-one：

$$
r_h=z_h-\operatorname{mean}_j z_j,
\quad
\widetilde w_h=\max(0,1+s r_h),
\quad
w_h=\frac{\widetilde w_h}{\operatorname{mean}_j\widetilde w_j}.
$$

所以 `[0,2]/[0,5]`只是讨论简称，不是最终硬边界；正式结果应写`s0p5/s2p0`。Step146实测
p05/mean/p95分别为`.460/1/1.523`与`0/1/2.554`。

## 7. 为什么单卡平均 262 秒，双卡 151.6 秒

一个 RLT cycle 的主要数据流是：

```text
π0 / student policy inference
  -> 8个环境执行C10并收集replay
  -> bootstrap下一状态动作
  -> 按UTD做critic/actor update
  -> 每25 cycle额外做fixed20 eval与checkpoint
```

| 部分 | 历史双卡 | matched-width单卡 | 影响 |
|---|---|---|---|
| rollout拓扑 | 2 rank × 4 env | 1 rank × 8 env | 总env仍是8，但policy推理、bootstrap、数据收发和reset锁域少了一套并行rank |
| 稳态rollout实测 | 约98--108秒 | 约186--194秒 | 占最大头，约慢1.8--1.9倍 |
| optimizer batch | global512；每卡micro128×2轮 | global512；micro256×2轮 | 单卡容量够，但512个样本仍由一张GPU算；较大microbatch提高利用率 |
| fixed20 eval实测 | 约237--256秒 | 约465--491秒 | 两个rollout rank变一个，约慢1.9倍 |
| 分布式通信 | 有跨卡同步 | 无all-reduce | 单卡省掉通信，update段没有严格慢2倍 |

RoboTwin 的8个环境内部仍有线程并行，所以不是把8个环境完全串行。问题在于双卡原来有两份GPU计算、
两份policy/rollout rank和两把独立reset锁；单卡只有一份。更大的microbatch能把空闲显存变成更宽的
矩阵计算，却不能变出第二个GPU或第二个rollout rank。

因此最终是：rollout/eval约慢1.9倍，update因micro256和省通信没有同比翻倍，混合上每25步评估与保存，
全程平均得到`262 / 151.6 = 1.72×`，而不是严格2倍。

## 8. 2026-08-28 10:50 CST 现场

- 两条均完整到`Step149/480`，Step150 rollout进行中，driver alive；
- 日志滚动平均step time约`250.7/250.9秒`；日志ETA约`23:03/23:04`；
- 若当前速度保持，预计约在`2026-08-29 09:55 CST`完成，后续fixed eval与训练阶段耗时会带来小幅漂移；
- GPU0/1现场约`15.8/17.1 GiB`，cgroup RAM约`192.77 GiB`；`oom=0`、`oom_kill=0`；
- 按用户决定，两条继续运行，本次未改配置、未停止进程。
