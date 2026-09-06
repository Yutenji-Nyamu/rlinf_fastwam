# OGPO × π0 × RoboTwin：参数来源与首版取值

最后更新：2026-08-08 13:36+08:00。本文只回答“配置里的算法/模型/数据/资源参数从哪里来、为什么这样
适配”。设计主线仍以 `00_INDEX_AND_IMPLEMENTATION_PLAN.md` 为准；这里不是第二套运行方案。

## 0. 读取规则

- **锁定**：方法或目标系统合同已经明确。
- **首版实现值**：完成官方源码对照后已经写入当前 OGPO YAML；首次 smoke、已完成35k formal与
  当前fresh 90k formal都用单独resolved override包执行，预算均已逐项展示并获批准。
- **资源落定**：不改变算法，已经由 π0 真实显存或 row-size probe 确认的容量/microbatch事实。
- OGPO 数学优先论文和官方固定 commit；π0 所有权/优化器优先 RLinf/OpenPI；RoboTwin 执行
  合同优先现有 adapter；历史专题只提供窄工程先例。

官方参数并非只有一套：通用表、Square-PaliGemma runnable 和 LIBERO 表各有覆盖。下表逐参数
选最相近的来源，不把“某个官方脚本”误写成 VLA 通用默认。

## 1. 模型、动作与时间尺度

| 参数 | 首版结论 | 状态 | 依据与适配 |
|---|---:|---|---|
| `H_model` | 50 | 锁定 | π0 checkpoint/模型头固有；不为 OGPO 改头 |
| `D_model` / `D_env` | 32 / 14 | 锁定 | π0 padded model coordinate / RoboTwin ALOHA action |
| `K` flow steps | 4 | 锁定 | 服务器 RLinf π0 resolved 值；官方 OGPO 小 actor 的 10 步不覆盖 π0 |
| execution `C` | 10 | 锁定 | 用户确认；每 10 waypoint 重看 observation，RLT 是同系统执行先例 |
| replay sequence `h` | `1..10` 真实长度 | 锁定 | terminal 前不足 10 时不伪造固定 horizon |
| episode limit | 200 primitive waypoints | 锁定 | RoboTwin `adjust_bottle` 合同 |
| action for likelihood | full raw `[K+1,50,32]` chain | 锁定 | OGPO whole-chain objective + π0 joint generative policy |
| action for Q | normalized canonical `[10,14]` | 锁定 | 从 raw model action 取执行前缀/active 维，供 critic 学习 |
| action for env | physical execution `[10,14]` | 锁定 | 同一语义动作经 π0 output transform 后送 RoboTwin；不写回 likelihood chain |

`gamma` 的**公式**已经锁定为 primitive 单位：

```text
R_h = sum(i=0..h-1) gamma^i * r[t+i]
y   = R_h + gamma^h * mask * Q_target(s[t+h], a_next)
```

不再有 `gamma_macro` 或开根。官方取值不是只按 episode 长度决定：

| 官方任务 | primitive horizon | chunk `h` | `gamma` | 与本项目的关系 |
|---|---:|---:|---:|---|
| Adroit | 200 | 4 | `.95` | 同为 200 步，但 reward/动作系统不同，不作为 RoboTwin 近邻 |
| Kitchen | 约 280 | 4 | `.99` | compositional reward |
| Square / PaliGemma-Square | 400 | 4 | `.99` | image 配置，但不是 VLA action expert |
| Toolhang / Transport | 1000 / 800 | 8 | `.999` | 长 sparse credit 先例 |
| LIBERO | 1000 | 8 | `.999` | image + language + sparse，方法语义最接近 |

RLinf π0 PPO 的 `.99` 是 H50 chunk-level：chunk 内 reward 先不折扣求和，再在四个 macro 之间乘
`.99`；它的名义 primitive 等价值约为 `.99^(1/50)=.999799`，但本项目不把该开根值写回 OGPO。
RoboTwin 当前成功前 reward 为 0、成功时为 1 并终止，因而早期动作能否收到末端成功 credit 比
“episode 恰好是 200 步”更关键：

| primitive `gamma` | `gamma^10` | `gamma^200` |
|---:|---:|---:|
| `.99` | `.9044` | `.1340` |
| `.999` | `.9900` | `.8186` |

所以当前单一建议是 **primitive `.999`**：它是 OGPO 已发布的 VLA/sparse 取值，也不会像 primitive
`.99` 那样把 200 步末端成功压到 `.134`。公式与单位已经锁定；数值仍标作“首版建议”，直到配置
审阅时由用户确认。

## 2. Q ensemble、TD 与 CA

| 参数 | 首版结论 | 状态 | 依据与适配 |
|---|---:|---|---|
| `num_qs` | 10 | 锁定 | 通用、PaliGemma、LIBERO 官方配置一致 |
| Q trunk | 5×512 + LayerNorm，FP32 | 锁定 | Square-PaliGemma image critic；π0 四块 prefix pooling 是输入适配 |
| `q_agg` for TD | `mean` | 锁定 | released PaliGemma runnable + LIBERO 论文表/配置；CA 不读取这个均值 |
| `q_variance_reduction` | false | 首版建议 | PaliGemma 脚本另采 8 个 next actions 再平均；不是核心公式，3B π0 首版主动裁掉额外 8 倍 target sampling |
| critic Polyak update rate | `tau=0.05` | 首版建议 | 论文表与 image/LIBERO 配置一致；即 `target=0.95 target+0.05 online` |
| critic LR | `3e-4` | 首版建议 | 论文/配置声明值；不复制官方 staged-online reset 把 critic 错设成 `ppo_lr` 的源码错位 |
| critic weight decay | `1e-5` | 首版建议 | OGPO image 配置 |

CA 没有额外 `alpha` 或 learned weight。对同一 state 的候选 `j`、Q head `m`：

```text
A[j,m] = Q[j,m] - mean_over_candidates(Q[:,m])

all A[j,:] > 0  -> A_CA[j] = min_m A[j,m]       # 最弱的正意见
all A[j,:] < 0  -> A_CA[j] = max_m A[j,m]       # 最接近 0 的负意见
otherwise       -> A_CA[j] = 0                  # head 分歧，不更新
```

CA 沿十个 heads 做一致性；TD `q_agg` 沿十个 target heads 形成 bootstrap 标量；group mean 沿同一
state 的候选做 baseline。三者是不同轴。

## 3. 候选链、PPO 与 SDE

| 参数 | 首版结论 | 状态 | 依据与适配 |
|---|---:|---|---|
| candidate group `G` | 8 | 首版实现值 | practitioner/PaliGemma 用 32，但官方 LIBERO 用 8；全量 π0 比官方 flow MLP 重得多，采用已有官方 VLA 任务值 |
| `clip_epsilon` | 0.01 | 首版建议 | OGPO normalized whole-chain score ratio；RLinf PPO 的 0.2 属于另一 loss/ratio 粒度 |
| `normalize_denoising_horizon` | true | 锁定 | 官方代码把 joint chain log-density 除以贡献项数；full K4 chain 为 `K+1=5` |
| `normalize_act_space_dimension` | true | 锁定 | 再除以 flattened `50×32=1600`；得到平均 score，不是论文字面的 joint importance ratio |
| `entropy_coeff` | 0 | 锁定 | 官方通用/image 配置 |
| `ft_flow_steps` | 4（全部 K） | 锁定 | whole-chain；不做 partial-chain ratio |
| SDE schedule | tapered | 锁定 | OGPO marginal-preserving sampler；每个 flow transition 都随机 |
| `sigma_init` | 0.01 | 首版建议 | 论文通用表与 LIBERO；PaliGemma-Square 的 0.05 是另一任务覆盖 |
| score drift correction | true | 锁定 | OGPO ODE-to-SDE correction 的方法组成 |
| independent `eta` | 不存在 | 锁定 | 当前 OGPO sampler 的随机强度就是 `sigma_init`，不新增同义旋钮 |
| Gaussian sample clip | mean ± 3σ | 首版建议 | 继承官方 sampler；不据此衍生替代采样路线 |
| intermediate `[-1,1]` mean clip | false | 首版建议 | OpenPI quantile normalize/unnormalize 本身不 clip；避免改变预训练 π0 的中间 flow path |
| final action bound | 不新增；任何既有 bound 不写回 raw chain | 锁定 | Q 取 normalized C10×14 projection；env 取现有 output-transform/14D execution copy |

时间坐标需要适配而不是换算法：OGPO 源码从 `t_ogpo=0`（noise）走到 1（action），π0 从
`t_pi0=1` 走到 0，因此实现用 `t_ogpo=1-t_pi0`。OGPO tapered
`sigma=sigma_init*sqrt(1-t_ogpo)` 在 π0 坐标下等价于
`sigma=sigma_init*sqrt(t_pi0)`。

RLinf π0 PPO 的 `noise_level=0.5` 不进入这里：它使用
`noise_level*sqrt(t/(1-t))`，默认只随机一个 denoise step、其余走 ODE；schedule、步数和 drift
correction 都与 OGPO 不同。只复用 π0 prefix、velocity、batching 和 raw-chain transport。

两个 normalization 不会删掉某个 denoise step 或某个 action 坐标。若原始 full-chain
log-density 为 `L=log p(x0)+Σ_k log p(x[k+1]|x[k])`，则本项目沿官方可执行代码使用：

```text
normalized_score = L / ((K+1) * (H_model*D_model))
ratio = exp(normalized_score_online - normalized_score_EMA)

K=4,H=50,D=32:
ratio = (p_online(raw_chain) / p_EMA(raw_chain))^(1/8000)
```

直观上，它比较的是“每个随机 chain 因子、每个 action scalar 的平均 log-density 变化”。这样 K 或
动作维数变大时 ratio 不会仅因求和项变多而指数放大。代价是它不再是论文 Eq. 3.2 的严格 joint
importance ratio，而是官方代码采用的 normalized whole-chain score ratio；`.01` clip 与这套可执行
尺度配套。实现时必须先把 `[50,32]` flatten 为 1600 再沿动作轴求和。

## 4. Actor、EMA 与 success BC

| 参数 | 首版结论 | 状态 | 依据与适配 |
|---|---:|---|---|
| trainable scope | frozen VLM + full action expert/projections | 锁定 | RLinf π0 PPO ownership；OGPO full actor update |
| PPO value head / GAE | false / 不存在 | 锁定 | OGPO baseline 来自 group-Q，不是 `V(s)` |
| actor LR | `5.6e-6` | 首版建议 | 全量 π0 action expert 沿用 RLinf RoboTwin PPO；不抄官方小 MLP 的 `4.5e-5` |
| actor AdamW | betas `.9/.95`、eps `1e-8`、wd `.01`、grad clip `1` | 首版建议 | RLinf π0 PPO optimizer 合同 |
| EMA update rate | `0.005` | 首版建议 | 论文 decay `alpha=.995` 与 PaliGemma runnable 等价 |
| success BC | true | 锁定 | 本项目只做 OGPO+CA |
| `bc_coeff` | 1.0 | 首版建议 | practitioner 常用值与 PaliGemma CA runnable |
| success-rate BC cutoff | false | 首版建议 | 官方逐任务可开关；首版不引入 0.45/0.5 经验阈值 |
| `use_success_buffer_q` | false | 锁定 | success BC 与成功样本额外 TD 是两件事；后者不是 OGPO+CA 必需项 |
| `best_of_n` | 1 | 锁定 | 关闭部署时 Q reranking；不影响训练期 candidate group |

EMA/online 的分工不是两套算法：EMA 负责生成链并给出 `old_lp`；online 对**同一链**给出
`current_lp`，只有 online scorer 保留梯度。更新后再把 online 慢速混入 EMA。

## 5. Replay、batch 与更新计数

| 参数 | 首版结论 | 状态 | 依据与适配 |
|---|---:|---|---|
| `offline_ratio` | `0.0`（online fraction=`1.0`） | 锁定 | 已有 SFT checkpoint 只初始化 actor；online/success replay 从空开始 |
| `start_training` | formal v1/v2均10,000 primitive rows；source参照20,000 | v2已部分完成 | 10k已产生足量success rows；PaliGemma image runnable为20k |
| `utd_q` | v1 `.1`；v2 `.05`；source参照1 | v2完成2,703次 | `.05`把bulk burst和update wall减半，不是官方值 |
| `utd_pi` | v1 `.1`；v2 `.05`；source参照1 | v2完成2,703次 | `.05`即每20个learning rows做一次paired update；trajectory仍批量结算 |
| logical Q/actor state batch | 64 | 首版实现值 | 官方 LIBERO；不抄 RLinf PPO 的 512/2048 current-rollout batch |
| candidate microbatch | 32 flat candidates / rank | 资源落定 | B4×G8 slice、B64/G8 update probe 和 8-env end-to-end smoke 均通过；probe allocated/reserved 35.31/39.57 GiB，全流程物理峰值 50.7/51.3 GiB |
| replay capacity | v1 40k；v2 100k；source250k | v2到64,078 rows | 容量本身未满；两次完整save的临时clone/RSS保留先撞240 GiB容器上限 |
| train env concurrency / rollout count | 8 / 1 | 首版资源建议 | 同机 RLT 先例；每批 8 条完整 episode，避免照搬 PPO 32×8 的大 rollout burst |

source `utd_pi=1` 对完整 200-row trajectory 意味着200次actor updates；当前 formal `.1` 对同样
200个learning rows结算20次paired updates。真实两卡一个paired credit约12.75秒，所以当前比率是
明确的完整3B π0计算适配，不能标成“官方 UTD=1”。

真实 replay probe 使用 runtime 同形的 current/next 三相机 raw observation：每 row 原始 tensor
1,383,224 bytes，RSS 约 1,438,720 bytes，序列化约 1,386,370.6 bytes。对应全局 ring：20k 为
26.80 GiB RAM/25.82 GiB sidecar，50k 为 67.00/64.56 GiB，250k 为 334.98/322.79 GiB。当前服务器
RAM 能放 250k，但 50k、100k、150k、200k、250k 若各保存一份逐渐增大的 replay，累计超过 816 GiB
数据盘。本次 formal 选择total35k/capacity40k，并保留50k interval，因而只在final35k保存一次：
35k约46.90 GiB RAM/45.19 GiB replay sidecar；连同smoke测得的checkpoint固定底座，final粗估约
58.3 GiB。该选择只覆盖本次一天预算，250k长期保留策略仍是另一问题。

final实测62,526,543,133 bytes（58.23 GiB）的组成是：FSDP/DCP两片共约10.27 GiB；两个rank
sidecar共约47.96 GiB。后者中replay按probe约45.19 GiB，余量才是Q/target-Q、critic optimizer、
EMA FP32 shadow、success IDs、RNG与计数器。因此“大”主要来自把三相机current/next observation
随replay一起保存以支持精确resume，不是π0单独占58 GiB。model-only导出可以更小，但不能恢复replay/
optimizers。v2实际落盘30,959-row checkpoint约53.01 GiB、60,968-row checkpoint约91.78 GiB，和
容量估算接近；但save后actor RSS分别由约48.6升到89.1 GiB、再由88.8升到168.5 GiB，最终触发
Ray 228 GiB阈值。因此限制不是replay capacity预分配或磁盘，而是`state_dict()`克隆全replay并一次性
`torch.save`后CPU allocator没有把RSS归还给容器。后续完整resume保存必须优先改成流式/分块、避免
整份clone；修复前不能把“磁盘能容纳三份checkpoint”误写成“90k运行资源已经可行”。

`offline_ratio` 是**每个训练 batch 的数据组成**，不是“先 online、再 offline”的时间比例。官方
runner 取 `int(B*offline_ratio)` 条 offline transition，再用 online replay 补满同一个 batch；该 batch
同时供 Q update 和 actor imagined-group state 使用。因此本项目的 `0.0` 表示：

- Q 只对 online replay 的 `(s,a,r,s')` sequence 做 TD；
- actor 的 B 个 candidate 起点也只从 online replay 采；
- success BC 来自 online replay 的成功子集；`use_success_buffer_q=false`，Q 不额外过采成功样本；
- SFT checkpoint 只有 actor 参数，没有 reward/next observation/done，不能当作 critic data。

论文允许有 demonstrations 时先做 actor BC、online 时再可选混入 offline transition；发布的主脚本
虽然先做 actor BC，却统一使用 `offline_ratio=0`，并把 BC-Q、CalQL、Q-only warmup 设为 0。当前
项目已有 SFT actor，因而直接进入同一类“随机 Q + 空 online replay + 纯在线 warmup/TD”合同。

## 6. 串行时间轴与正式训练规模

### 6.1 所有 step 的单位

| 参数/动作 | 计数单位 | 当前设置 | 依据与适配 |
|---|---|---:|---|
| `bc_pi_steps` | actor BC optimizer step | 0 | SFT checkpoint 已完成 actor 初始化，不在 OGPO runtime 重训 |
| `bc_q_steps` | critic optimizer step | 0 | 发布主脚本关闭；不把 demonstration 变成 Q 数据 |
| `calql_steps` | offline critic optimizer step | 0 | 发布主脚本关闭；当前无 mixed/offline critic phase |
| `start_training` | primitive online row | v1/v2均10k；source20k | 已完成v1支持保留10k，不再同时改变warmup |
| `q_warmup_steps` / `utd_warmup` | online rows / Q-only updates per row | 0 / 1（不生效） | 官方可选项，PaliGemma runnable 关闭 |
| `bc_refine_steps` | online actor refine optimizer step | 0 | 官方可选项，当前 success BC 在 joint phase 自然生效 |
| `online_steps` | primitive environment row | v1 35k；v2 90k；source250k | 90k按v1实测拟合约24h，不冒充论文规模 |
| `utd_q` / `utd_pi` | 每新增 primitive row 的 optimizer credits | v1 .1/.1；v2 .05/.05；source1/1 | 单位对齐官方；`.05`是完整π0/bulk调度适配 |
| actor / critic LR schedule | optimizer step | constant / constant | actor 沿 π0 PPO `5.6e-6`；critic沿 OGPO common `3e-4` |
| train metrics / eval / checkpoint | rollout batch / primitive row | v1每批/20k/50k；v2每批/10k/30k | 90k时baseline+10..90k共10次eval，30/60/90k共3份完整checkpoint |
| eval episodes | 完整 episode | 20 | PaliGemma image runnable |

官方不同实验公开规模为：common `1M/start10k`，Square-PaliGemma `2M/start20k`，
Toolhang-PaliGemma `3M/start20k`，论文 LIBERO `250k`。固定 commit 没有 LIBERO runnable；这里的
250k 来自论文 Table 8，不把加载 π0.5-LIBERO encoder 的 Robomimic PaliGemma 脚本误称为 LIBERO。

### 6.2 本项目只保留两个阶段

```text
Stage A — collection
load SFT actor -> baseline eval -> Q random init -> online/success buffers empty
formal collect rows 0..9,999; Q updates=0; actor updates=0

Stage B — steady OGPO+CA
formal rows 10,000..34,999
UTD-Q=.1; UTD-PI=.1; success BC 有成功样本时自然加入
LR、loss、G、batch 不在里程碑切换
```

第二轮不增加新阶段，只更改同一时间轴的预算：fresh SFT actor、random Q、empty replay重新开始；
rows 0..9,999只收集，rows 10,000..89,999用paired UTD-Q/PI`.05`，最终目标4,000 paired updates。
35k checkpoint不作为初始化或offline数据进入第二轮。

首次 one-update smoke 暴露了这个 source-faithful 顺序的冷启动边界：10Q 首次随机初始化后，actor
先于 critic TD update 执行；严格 10-head CA 在本批把候选全部 veto 的现象与
`actor_loss=actor_grad_norm=0` 一致，而 critic 随后得到非零 loss/grad。它不是更改 Stage A/B 的依据，
也不能由 `policy_version=1` 推断 actor 参数已有有效变化。若继续做最小诊断，优先只记录
`ca_nonzero_fraction/ca_abs_mean` 并观察第 2–3 次 update；只有 CA 非零而 actor grad 仍为零，才指向
scorer/backward 实现问题。

source 250k rows对应1,250个满长200-step episode-equivalents和约230k paired credits。当前 formal
35k rows对应175个episode-equivalents；warmup后25k rows×.1严格产生2,500 Q与2,500 actor updates。
环境按最多8×200的整批执行，尾批仿真槽位可超过入replay的rows，但actor ingest会把replay截在exact35k。

论文 Algorithm 2 写 `Q -> actor -> targets`，固定 commit 的 fused `_update` 实际执行
`actor -> actor EMA -> critic -> target critic`。两者都串行，且损失主要读取 update 前的 target
network；本项目按实现优先级采用**发布代码的 actor-first 顺序**，不保留顺序开关。官方 runner
逐 primitive step 交错采集和更新；RLinf 首版仍按已定边界等完整 trajectory 返回后批量写 replay，
再结算相同数量的 credits，因而 UTD 总数相同但更新呈 burst。

formal实测把这个“调度形状”具体化：典型8-env wave产生1,242–1,600条valid rows，UTD`.1`随后
一次连续执行约124–160个paired updates，即约26–34分钟；一个burst均值只写成一个ratio点。
第25 wave写1,107 rows并跑110次，第26 wave只接收quota所需32 rows并跑4次，最终26 waves、
208条train episodes、严格2,500 updates。ratio在同样长度burst下从早期最低`.133`逐步恢复到
final`.942`，所以burst/EMA lag是重要线索，但不能单独证明总UTD过高。

源码对照给出了节奏参照：官方OGPO在warmup后每个primitive `env.step`都会调用一次update，
PaliGemma脚本的UTD1通常就是两次环境步之间1次paired update；官方RLinf π0 PPO虽先收大rollout，
其公开RoboTwin配置一次rollout后也只有4次真正optimizer step（microbatch只是梯度累积切片）。本项目
`.1`的124–160次burst来自“8条完整episode一起返回 + row credits集中结算”，不是OGPO官方节奏。
`.05`把它降到62–80次，但仍未恢复逐primitive交错；因此它是简单对照变量，不叫源码完全对齐。

### 6.3 Source-aligned 250k 与当前 formal 的显式计算账

在 `B=64,G=8` 下，约 230k actor updates 会生成/重评：

```text
230,000 * 64 * 8 = 117,760,000 imagined chains
```

G32 则是 471,040,000 chains。每条 chain 还有 K4 EMA forward、K4 online forward/backward 和
10Q 评分；官方 250k/G8/UTD=1 的近邻 actor 是小 flow 网络，不是完整 π0 action expert。因此
250k 是source-aligned交互规模参照；已完成formal使用35k/10k/UTD.1/cap40k。

真实 2×A800 production probe（B64/G8、flat32/rank、rank0 有 success BC、rank1 无）测得：

| 串行段 | wall time / paired credit |
|---|---:|
| TD next action | 2.00 s |
| actor：8 个 candidate microbatches + success BC + optimizer | 9.85 s |
| critic + optimizer | 0.89–0.90 s |
| 合计 | 约 12.75 s |

因此 230k paired credits 的 **update-only** 线性投影约为 2.93M 秒，即 33.9 wall-days；两张 GPU
合计约 1,629 A800 GPU-hours。该数字不含 model load、simulator rollout、eval、checkpoint 和调度间隙，
也不把未来优化假装成已实现加速。当前 formal 选择2,500 paired credits，即1,280,000 imagined
group chains；最终实测wall为44,622秒（12:23:42）：rollout 12,714秒（28.5%）、paired training
30,733秒（68.9%）、三次eval约833秒（1.9%），其余为初始化/sync和final save。此前19–21小时是
短smoke线性投影，现由完整formal实测替代。

v1 fixed eval为`5%@0 -> 35%@20,081 -> 5%@35k`，因此`.1`不能再标作经验最优。用户已批准并于
2026-08-08 13:27启动fresh第二轮：total90k/warmup10k/paired UTD`.05`、capacity100k、eval10k、
checkpoint30k；目标4,000 updates、204.8万imagined chains、典型62–80
updates/wave。按本轮实测，rollout约9.08h、updates约13.66h、10次eval约0.77h、3次save与固定开销
约0.20h，名义合计23.71h；用户明确24小时只是近似值，因此没有改用85k保守案，也不设置wall timeout。
`.025`学习强度再减半，当前无依据。若要更直接限制actor drift，可讨论Q`.1`/PI`.05`，
但当前worker要求二者相等，属于需实现和验证的新配置。formal v2固定10个eval点；完整resume
checkpoint与轻量actor/EMA/Q快照必须分开设计，不能把“约10次评估”直接等同于永久保留10份replay。

2026-08-08 18:08的v2首个live对照修正了一个先前假设：21,802 rows时调度精确得到590 paired
updates，典型每波67–80次；ratio为`.948@4 -> .123@79 -> .609@518 -> .703@590`。v1按累计
optimizer update对齐时为`.575@566 -> .745@707`；v2随后为`.703@590 -> .777@656`，两轮恢复轨迹
接近；按environment rows看v2反而
更慢，因为`.05`只积累一半updates。因此`.05`已经确认减少wall/burst，却**尚未显示它本身修复ratio**，
不能再把v1的低ratio主要归因于单个124–160-update burst。

v2最终在64,078 rows、2,703 paired updates处因checkpoint后的CPU RSS阶梯增长而异常结束，不是
算法数值或GPU OOM。可见fixed eval为`5,5,15,30,10,30,40%`（0至60,968 rows）；latest ratio`.944`，
actor/BC/critic均有限。故`.05`至少提供了学习信号和较低计算量，但没有完成90k，不能称为收敛实验或
经验最优。是否需要Q/PI分离、改变EMA或更细交错，等补齐ratio quantile/clip fraction、CA和Q排序指标
后决定；恢复训练前先解决保存RSS，参数判断与运行故障不能混为一谈。

连续更新先例的来源边界也已锁定。官方OGPO按primitive `env.step`调用update，当前episode只在done
时flush到replay，所以可连续复用旧replay，但真实动作执行仍夹在updates之间；当前RLinf则是8条完整
episode返回后集中做62–80次纯计算paired update。
[Spinning Up SAC固定实现](https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/sac/sac.py#L318-L322)
默认每50个env steps集中做50次完整Q+actor+target update，SB3 off-policy循环也支持rollout后多次
gradient steps；因此几十次burst不是结构性错误。REDQ/RLPD的UTD20主要是20次critic、一次actor，
不能证明连续80次actor安全。第二轮`.05`准确标记为降低上一轮burst与总训练强度的项目适配，不称作
OGPO官方验证值；本轮fresh对照将检验它。

## 7. 并行轴与双 A800 首版布局

| 并行轴 | 当前单一建议 | 来源与边界 |
|---|---:|---|
| node / GPU | 1 node / 2×A800-80GB | 当前服务器事实 |
| component placement | actor、env、rollout 均在 GPU 0–1 | 服务器两卡 π0 PPO placement 先例 |
| actor / rollout world size | 2 / 2 | RLinf FSDP/HF worker |
| model parallel | FSDP full-shard；TP=0、PP=0 | RLinf π0；`pipeline_stage_num=1` |
| train env concurrency | 8 total（每 rollout rank 4） | 同机 RLT 已跑通的资源先例；比 PPO 32-env 基线减少整 episode 后的 update burst |
| train `rollout_epoch` | 1 | 每批每 env 一条完整 episode；不叠多轮 trajectory |
| eval concurrency | 4 env × 5 waves = 20 episodes | 官方 PaliGemma eval 数 + 两卡 RoboTwin 批量执行 |
| env `group_size` | 1 | imagined candidate `G` 不进 EnvWorker |
| logical replay-state batch `B` | 64 | 论文 LIBERO；Q、actor、success BC 的逻辑 batch |
| imagined candidate `G` | 8 | 论文 LIBERO；每个 state 的整组候选留在同一 rank |
| flat candidate microbatch | 32 / rank，已真实落定 | B4×G8 microbatch 两卡通过；只切计算，不改 B/G 数学 |
| production paired update | 约 12.75 s；35.31/39.57 GiB allocated/reserved | 真实 B64/G8 `target -> actor+BC -> critic` 两卡 probe |
| Q heads | 10 个 FP32 head 同张量 vectorize | 官方 `M=10`；不建 10 个进程 |
| TD next-action samples / QVR | 1 / disabled | B 轴批量；不增加 PaliGemma 可选的 8× target actions |
| Best-of-N | 1 | 关闭部署期额外候选 |
| rollout/train pipeline | false | 同一双卡先 rollout 后训练；eval 也暂停训练 |

2026-08-08运行后决策：上述2×A800、train env8、B64/G8、flat32/rank与10Q vectorize均无资源或
并行异常，formal v2保持不变；只有出现明确吞吐问题才做单变量probe，不因ratio先改并行。

首次 smoke 保持并行轴 `2 GPU / train 8 env / B64 / G8 / flat32/rank`，只把串行预算缩为 80 rows、
1 个 paired update 和一次 4-env C10 eval。实测如下；`nvidia-smi memory.used` 是整卡物理占用，不能
与 probe 的 PyTorch allocated/reserved 混写：

| end-to-end smoke 量 | 实测 |
|---|---:|
| train rollout | 8 env × C10 = 80 primitive rows；54.05 s |
| paired update / sync / eval rollout | 12.62 / 12.49 / 14.86 s |
| GPU0 / GPU1 physical memory peak | 50,701 / 51,311 MiB |
| GPU0 / GPU1 util peak | 100% / 100% |
| cgroup current peak | 60,136,706,048 bytes（约 56.0 GiB） |
| minimum host MemAvailable | 1,003,950,000,000 bytes（约 935 GiB） |
| OOM / OOM-kill delta | 0 / 0 |
| checkpoint disk delta | 14,093,619,200 bytes（约 13.13 GiB） |

因此每卡观察到约 30 GiB 显存余量，但不能直接推出“并行度乘 2”：B 会扩大 prefix/Q/candidate 工作，
G 会增加 imagined-chain microbatch 数，env 数主要影响 simulator/RSS 和 rollout wall time，而且这些阶段
并非同时达到各自峰值。smoke 支持的结论是当前布局可运行，并为下一次单变量、有界并行 probe 提供
上界；调参选择在聊天中基于吞吐和预算讨论。

并行只发生在一个 denoise step 内的 state/candidate/head 轴；K4 的 `x[k+1]` 依赖 `x[k]`，环境中
C10 个 primitive actions 也按时间串行，optimizer updates 则因权重逐步改变而串行。prefix/VLM 对 B
个 states 只算一次并跨 G 复用，candidate-specific action-expert suffix、online backward 和 Q score
不能跨候选复用。

`B=64` 时，G8/G32 分别是 512/2048 chains/update。每 rank flat microbatch=32 时，两卡分别约
需要 8/32 个 candidate microsteps；G8 已足以填满 GPU，G32 的 candidate 主体接近四倍工作量，
总 step time 因 prefix/TD/BC 固定成本会落在 1–4 倍之间，不是只增加一点。首版因此固定建议 G8。

真实两卡单个 B4×G8 microbatch 约 1.41/1.46 秒，same-chain score delta=0，online gradient tensors
为 132/101；该 slice 峰值约 33.19/36.26 GiB。B64 全局每 rank 32 states，串行 8 个 candidate
microsteps；随后测得完整 actor+success-BC+critic paired update 约 12.75 秒、峰值约
35.31/39.57 GiB，故切片大小与最终 update 预算现在都有现场依据。

分布式沿 state 轴切 batch，并让同一 state 的完整 G8 留在一张 rank 上完成 group mean/CA；不把一个
group 拆到两卡后再 all-gather。

## 8. 参数冲突的固定处理

- 论文公式与可执行代码冲突：先区分确认缺陷还是任务覆盖；不会把静态错位悄悄复制。
- PaliGemma-Square 与 LIBERO 不同：encoder/critic 结构优先前者，重图像/语言候选规模优先后者。
- RLinf PPO 与 OGPO 同名参数不同单位：以消费它的数学对象为准，例如 PPO `.2` clip、macro `.99`
  gamma 和 `.5` SDE noise 都不因同名而迁移。
- resource-only 数值不派生第二套算法；实现期一次 probe 后写入唯一配置。

## 9. 官方入口

- [OGPO practitioner guide](https://arxiv.org/html/2605.03065v4#A1.SS1)
- [默认超参 Table 4](https://arxiv.org/html/2605.03065v4#A10.T4)
- [LIBERO 超参 Table 8](https://arxiv.org/html/2605.03065v4#A10.T8)
- [PaliGemma runnable：buffer/UTD](https://github.com/simchowitzlabpublic/OGPO_public/blob/0b3be413cde766a41257c6b19c0c2b06393a557f/scripts/ogpo/square_image_paligemma.sh#L29-L35)
- [PaliGemma runnable：Q/G/BC](https://github.com/simchowitzlabpublic/OGPO_public/blob/0b3be413cde766a41257c6b19c0c2b06393a557f/scripts/ogpo/square_image_paligemma.sh#L46-L65)
- [PaliGemma runnable：critic/Q-VR/SDE](https://github.com/simchowitzlabpublic/OGPO_public/blob/0b3be413cde766a41257c6b19c0c2b06393a557f/scripts/ogpo/square_image_paligemma.sh#L97-L150)
- [tapered-SDE sampler/scorer](https://github.com/simchowitzlabpublic/OGPO_public/blob/0b3be413cde766a41257c6b19c0c2b06393a557f/ogpo/agents/modules/pg_helper.py#L123-L414)
- [CA implementation](https://github.com/simchowitzlabpublic/OGPO_public/blob/0b3be413cde766a41257c6b19c0c2b06393a557f/ogpo/agents/modules/pg_helper.py#L451-L494)
- [official UTD call site](https://github.com/simchowitzlabpublic/OGPO_public/blob/0b3be413cde766a41257c6b19c0c2b06393a557f/ogpo/runners/online_rl_runner.py#L529-L657)
- [official phase entry](https://github.com/simchowitzlabpublic/OGPO_public/blob/0b3be413cde766a41257c6b19c0c2b06393a557f/ogpo/main.py#L369-L559)
- [official actor-first fused update](https://github.com/simchowitzlabpublic/OGPO_public/blob/0b3be413cde766a41257c6b19c0c2b06393a557f/ogpo/agents/ogpo.py#L1533-L1619)
- [official offline/online batch mixer](https://github.com/simchowitzlabpublic/OGPO_public/blob/0b3be413cde766a41257c6b19c0c2b06393a557f/ogpo/runners/online_rl_runner.py#L110-L141)
- RLinf target-system values：本机 `.research-rlinf/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml`
  与 `.research-rlinf/examples/embodiment/config/model/pi0.yaml`；实施以服务器 `6d0db56b` 同路径为准。
