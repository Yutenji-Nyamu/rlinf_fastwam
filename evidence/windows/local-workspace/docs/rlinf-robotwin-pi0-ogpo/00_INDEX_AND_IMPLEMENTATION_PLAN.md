# OGPO × π0 × RoboTwin × RLinf：上下文索引与实施主计划

最后更新：2026-08-08 18:19+08:00。

本文是专题 **OGPO × π0 × RoboTwin × RLinf** 的唯一当前规范性实施计划。它随讨论持续维护设计
决定、适配理由、实施状态与验收口径。2026-08-07 已在服务器独立
`codex/ogpo-pi0-robotwin` worktree 完成主体实现和定向测试，commit `5d5c84e3`
已推送到同名 `personal` 分支；经用户批准的首次真实 RoboTwin smoke 与一日预算 formal 均已完成。
formal 精确达到35,000 rows和2,500 paired updates，26 waves/208条train episodes，exit 0；fixed eval
为5%→35%→5%。用户随后批准fresh第二轮90k/10k/paired UTD`.05`、10次eval和3次完整checkpoint；
该轮已于2026-08-08 13:27启动并通过baseline/首个train rollout健康边界。动态事实、失败和逐条命令见本文件第 10 节与实施账本，不能由旧聊天或
历史专题补写成“当前”。

详细逐来源职责见 [`01_REFERENCE_MATRIX.md`](01_REFERENCE_MATRIX.md)，基线与目标调用/数据流见
[`02_CALL_AND_DATA_FLOW.md`](02_CALL_AND_DATA_FLOW.md)，参数来源与首版取值见
[`03_PARAMETER_PROVENANCE.md`](03_PARAMETER_PROVENANCE.md)，查阅、裁剪和后续实施过程见
[`evidence/CONTEXT_INVENTORY_LOG.md`](evidence/CONTEXT_INVENTORY_LOG.md)。它们是条件读取附录和
账本，不是并行实施计划。

## 0. 一屏结论

“官方实现优先级最高”是对的，但不能把所有材料压成一条总排名。需要同时维护三条
互相正交的 P0 真值：

1. **OGPO 论文 v4 + 官方代码**：定义 OGPO 的目标、critic、replay、denoising-MDP PPO、
   SDE likelihood、target/EMA 和稳定化项，是**算法真值**。
2. **RLinf 官方 π0 + PPO + RoboTwin**：定义目标工程中的 runner、EnvWorker、π0 policy
   wrapper、rollout/replay payload、FSDP、同步、DCP 与配置入口，是**集成骨架真值**。
3. **OpenPI + RoboTwin 2.0 官方实现**：分别定义 π0 flow/action/prefix/normalization 与
   RoboTwin observation/action/reward/success/reset，是**模型和环境接口真值**。

DSRL、RLT、QAM、Fast-WAM PPO/GRPO 和历史流水账都放在第二层，但用途不同：

- DSRL/RLT/QAM 只提供已经验证过的窄接口、resume、replay 与 action-coordinate；
- QAM 与 Fast-WAM PPO/GRPO 的历史结果不参与 OGPO 方法或 C 的因果判断，出现相同具体问题时
  再按需查阅；
- 操作文档和日志对路径、资源、launcher、checkpoint、故障定位很有用，但对算法语义权威低，
  对动态状态没有权威。

## 1. 官方 OGPO 已确认的核心

当前 source lock：

- 论文：[arXiv:2605.03065v4](https://arxiv.org/html/2605.03065v4)，2026-06-26；
- 项目页：[OGPO project page](https://simchowitzlabpublic.github.io/ogpo-site/)，标注 ICML 2026 accepted；
- 官方代码：[`simchowitzlabpublic/OGPO_public@0b3be413`](https://github.com/simchowitzlabpublic/OGPO_public/commit/0b3be413cde766a41257c6b19c0c2b06393a557f)，MIT。

算法骨架是：

```text
真实环境 transition
  -> 长期 replay
  -> off-policy Q ensemble 的 TD 更新

replay 中一个 state
  -> 当前/参考生成策略并行采多条 denoising trajectories
  -> 终点 action chunk 由 target-Q ensemble 评分
  -> 同 state 的 group mean(Q) 作 baseline
  -> 对整条 denoising chain 做 clipped PPO / likelihood-ratio 更新
```

必须保留的语义边界：

- expensive environment data 用 off-policy critic 反复利用；cheap denoising trajectories 可以
  从 replay state 重新并行采样；
- actor 是 zeroth-order policy-gradient 更新：**不对 critic 求 `dQ/da`，不把 Q 梯度穿过
  生成链，也不做 BPTT**；
- group baseline 是 mean-Q；论文明确省略 GRPO 的方差归一化，所以 OGPO 不是把现有 GRPO
  配置改个名字；
- flow ODE 的每步概率原本奇异，官方做逐步噪声注入并用 stochastic-interpolant correction
  保持 ODE/SDE 边缘分布对齐；
- action chunk 在 environment MDP 中作为一个 action，critic 用 chunk 内有效回报和
  `gamma^h` bootstrap；不能直接继承另一个专题的 `N`、reward 或 done 投影；
- OGPO+ 增加 success-buffer BC；OGPO+CA 再增加 critic-ensemble sign-consensus conservative
  advantage。本项目只实现完整的 OGPO+CA 运行路径，不暴露 vanilla/OGPO+ 运行变体。

官方代码的可执行入口主要是：

- `ogpo/agents/ogpo.py::OGPOAgent`：actor/critic loss、group-Q advantage、整链 ratio；
- `ogpo/runners/online_rl_runner.py`：online replay 与交替更新；
- `ogpo/configs/algos/{common,ogpo}.yaml`：默认合同；
- `ogpo/main.py` 与 `scripts/ogpo/*.sh`：训练入口与逐环境覆盖。

重要限制：官方仓库是 JAX/Flax 的独立 flow-matching policy，公开环境不是 RoboTwin，且没有
直接训练 OpenPI π0/π0.5。图像路线可从 π0/π0.5 checkpoint 读取**冻结 PaliGemma encoder**，
actor 仍是 OGPO 自己的 flow policy；论文 LIBERO 实验使用的又是 frozen MUSE + IMPALA。
因此它是算法 oracle，不是可直接复制进 RLinf 的 π0 backend。

## 2. 目标系统的 P0 上下文

### 2.1 RLinf 官方 π0 PPO RoboTwin

- 官方说明：[RL with RoboTwin Benchmark](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html)。
- 2026-08-05 在线参考 pin：[`RLinf@36baa750`](https://github.com/RLinf/RLinf/commit/36baa75031ff863449322b548be49bd5620698ef)；这是参考锁点，不代表已经选定服务器实施基线。
- 2026-08-07 服务器可执行基线：`/root/autodl-tmp/RLinf@6d0db56bf26f...`，本地
  `origin/main` 也指向该 commit；它比在线参考 pin 旧，但与现有 A800 配置、共享 `.venv`、SFT
  checkpoint 及三个已验证专题的共同父线一致，因此首版从它建独立 worktree，再把新上游只作
  source-level collision review，不在实现前先升级整个 RLinf。
- 本机只读官方镜像：`.research-rlinf@c5ca51cc21c007a41d287159f9e1b14e0200000e`。
- 首要配置：
  `.research-rlinf/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml`。
- 首要接口：
  `rlinf/models/embodiment/openpi/openpi_action_model.py`、
  `rlinf/workers/env/env_worker.py`、`rlinf/workers/rollout/hf/huggingface_worker.py`、
  `rlinf/runners/embodied_runner.py`、`rlinf/algorithms/{advantages,losses}.py`。

它最值得复用的是已经存在的 π0 denoising/logprob 底层函数、trajectory transport、
FSDP/sync/DCP 和 RoboTwin 环境链。PPO 的 GAE、value head、current-rollout minibatch 和现有
actor-critic loss 不进入 OGPO；它也不能决定 OGPO 的 Q target、replay state sampling、
group-Q advantage、success buffer 或 critic/actor 更新顺序。

### 2.2 OpenPI 与 RoboTwin

- [`Physical Intelligence/openpi@15a9616a`](https://github.com/Physical-Intelligence/openpi/commit/15a9616a00943ada6c20a0f158e3adb39df2ccac)：π0 flow
  模型、checkpoint、action horizon/dim、prefix、输入输出 transform 与 norm stats 真值。
- [RoboTwin 2.0 project](https://robotwin-platform.github.io/) / [`main@13c3c47f`](https://github.com/RoboTwin-Platform/RoboTwin/commit/13c3c47ff4312dd62484bcd51be034af55c062d1)：
  cameras、14D state/action、task、success、episode/reset、数据与仿真真值。
- RLinf 官方指南实际要求 RoboTwin `RLinf_support`；2026-08-05 该集成分支参考 pin 为
  [`0008ae68`](https://github.com/RoboTwin-Platform/RoboTwin/tree/0008ae6800df9f75fc8de7098bacb01735fd8fd2)。不能用 RoboTwin main 的新状态替代这个兼容性合同。

RLinf 的 adapter 是目标可执行路径，但任何 π0 数学或 RoboTwin 环境争议仍回到这两个上游。
2026-08-07 只读刷新已核验服务器 SFT checkpoint、task config、resolved π0 config 与 norm SHA；
它们是本次带时间的实施快照，真正创建 worktree 或运行前仍按 live-truth 规则再核对一次。

## 3. 第二层参考怎样用

DSRL 提供 target/replay/resume 形态，RLT 提供 `H50/C10/D14` 和 canonical action，QAM 提供
π0 prefix、10Q 与 replay 工程。Fast-WAM 只在遇到同类 logprob、同步或 Ray 问题时查阅。
具体文件和禁止迁移的算法部分统一由 [`01_REFERENCE_MATRIX.md`](01_REFERENCE_MATRIX.md) 维护；
这些历史结果不用于猜测 OGPO 的失败原因。

## 4. 本机还保留什么

本机保留专题文档、只读源码镜像、轻量 audits/exports 和历史脚本，可用于定向追溯；精确盘点见
账本 CTX-0005。大型 checkpoint、完整 run root、数据和模型仍以服务器为准，OGPO 新分支只从
锁定的服务器 RLinf 基线建立。

## 5. 默认读取路径与修剪规则

以后每个 OGPO 新任务默认只读：

1. 根 `AGENTS.md`；
2. 根 `PROJECT_CONTEXT.md`；
3. 根 `HANDOFF.md` 的 OGPO 行/小节；
4. 本文件。

只有出现具体问题时才读：

- 算法争议 -> OGPO paper/code 对应公式、symbol；
- π0/RoboTwin 接口 -> RLinf/OpenPI/RoboTwin 官方路径；
- replay/resume/action contract -> `01_REFERENCE_MATRIX.md` 指向的 DSRL/RLT/QAM 窄章节；
- 某个已知故障 -> 对应 evidence/日志；
- 动态服务器事实 -> 新的只读现场检查；不能沿用本文件里的旧时间快照。

裁剪规则：

- 不全文加载 DSRL/RLT/QAM/Fast-WAM 作为 OGPO 默认上下文；
- 不复制旧专题参数进 OGPO 主文档，只记录“为什么读它、能提供什么、禁止继承什么”；
- source pin、实现决定、运行 evidence、动态状态分开；
- 同一事实只在一个 SSOT 详细维护，旧讨论与日志只由索引链接。

## 6. 已研究后的设计决定

本节不把未经研究的选项丢给用户。每一项都按“官方怎么做 -> 目标系统怎么做 -> 为什么要适配 ->
怎么验证”记录；仍需 live 信息的只保留为有明确收敛办法的缺口。

### 6.1 只实现 OGPO+CA

三个名称的关系先说清楚：vanilla OGPO 是“Q ensemble + group-Q baseline + whole-chain PPO”；
OGPO+ 再加入只模仿成功轨迹的 flow-BC；OGPO+CA 再把每个 Q head 的组内 advantage 做符号
一致性过滤。论文把 success BC 视为 OGPO+ 最关键的稳定化，把 CA 视为图像 critic 下的重要
稳定化；官方主脚本也同时开启两者。

**本项目决定**：生产配置、worker route 和训练入口只提供 **π0-adapted OGPO+CA**，不提供
`adv_strategy=vanilla`、OGPO+ 配置或三套 pilot。单元测试仍会直接验证“每个 head 先组内中心化、
全体同号才保留最小幅值”的数学，但不会把内部算子包装成另一种运行算法。

两项官方外围功能不进入首版：

- `use_success_buffer_q` 是对成功样本额外做一次 TD，并不是 success BC，也不是论文定义的
  OGPO+CA 必要部分，关闭；
- `best_of_n=8` 是推理时多采八个 chunk 再让 Q 选一个。论文称收益通常边际、推理昂贵时可省，
  3B π0 首版固定 `BoN=1`。

### 6.2 π0 的可训练范围

这项可以从 RLinf π0 PPO 抄，但只能抄其中的参数所有权，不能抄 PPO value head：

- 官方 OGPO 图像脚本冻结 PaliGemma encoder，完整训练它自己的 flow actor；
- RLinf `OpenPi0ForRLActionPrediction.freeze_vlm()` 在 `train_expert_only=true` 时冻结
  `paligemma`，保留 `gemma_expert`、action/state/time projection 与 action output projection
  可训练；这正是 π0 中最接近“冻结感知编码器、完整训练 GCP actor”的边界；
- OpenPI 自己的普通非-LoRA `Pi0Config.get_freeze_filter()` 默认是 full fine-tuning；所以
  “只训练 action expert”是 RLinf π0 PPO 的官方集成选择，不是 OpenPI 的通用默认；
- RLinf PPO 的 `add_value_head=true`、GAE/value loss 不属于 OGPO。OGPO 的 baseline 来自同一
  state 的 group-Q，不训练 PPO value head。

**本项目决定**：首版沿用 `train_expert_only=true`，训练完整 action expert 及其投影，冻结
VLM/prefix encoder，关闭 value head。首版不解冻 VLM。

这里的“一份 VLM”指一套逻辑参数，不是假设跨进程零拷贝：训练 worker 内 online expert 与
EMA target expert 共用冻结 prefix；rollout worker 仍按 RLinf 现有架构持有同步后的推理副本。
checkpoint 不重复保存冻结 VLM 两次，只保存 SFT/source fingerprint、online expert 和 EMA expert。
OGPO critic 是另外创建、另外优化的 action-conditioned Q ensemble，不复用 π0 PPO value head。

### 6.3 critic 看 deployment observation 还是 privileged state

官方代码同时支持 `critic_obs=state/image`，并在 image + frozen-PaliGemma 路径中把
`use_state=proprio` 设为默认；源码明确把 `full` object state 标为 privileged、禁止默认使用。
论文也显示 pixel critic 比 state critic 难学，并因此引入 conservative advantage，但没有要求
critic 必须获得部署时不可见状态。当前 RLinf RoboTwin adapter 暴露的是三路图像、qpos/state
和语言指令，没有一条现成 PPO 路径提供 object-pose privileged state。

**本项目决定**：首版 actor 和 critic 都只使用部署可见 observation。critic 输入采用
stop-gradient 的冻结 π0/PaliGemma 表征 + proprio；语言/图像 mask 与 π0 policy transform 保持
一致。首版不修改 RoboTwin 接口去增加 privileged state。

### 6.4 action chunk、有效回报与 `gamma^h`

先统一术语：π0 一次预测的 50 个动作叫 **model horizon** `H_model=50`；真正连续送给机器人、
然后重新看图规划的前缀叫 **execution chunk** `C`；其中每一个 14D 关节动作叫一个 primitive
waypoint。一个 200-waypoint episode 若 `C=50/20/10`，最多只有 4/10/20 次重新规划。

官方 OGPO 的 runner 也是先生成一个 chunk，再通过 action queue 逐个 `env.step`。不同点是它把
每个 primitive transition 都放进 replay，训练时从任意 replay state 采连续 `h` 个 transition；
官方 [`datasets.py`](https://github.com/simchowitzlabpublic/OGPO_public/blob/0b3be413cde766a41257c6b19c0c2b06393a557f/ogpo/utils/datasets.py#L178-L200)
与 [`q_helper.py`](https://github.com/simchowitzlabpublic/OGPO_public/blob/0b3be413cde766a41257c6b19c0c2b06393a557f/ogpo/agents/modules/q_helper.py#L328-L350)
形成：

```text
R_h = Σ_{i=0}^{h-1} gamma^i r[t+i]
y   = R_h + gamma^h * bootstrap_mask * Q_target(s[t+h], a_next_chunk)
```

2026-08-07 live 源码核对发现，现有 `RoboTwinEnv.chunk_step` 虽返回长度为 C 的 reward/done 张量，
但内部只调用一次 vectorized step；`obs_list` 只有块末 observation，reward 也只是合成到末格，
因此不能恢复真正的 primitive state 或提前成功位置。与此同时，`RoboTwinEnv.step` 已支持单个
waypoint。首版不再接受旧的整块 macro 近似，而是在 OGPO opt-in route 中让 EnvWorker 把
`action[:C]` 逐 waypoint 执行并收集 transition；PPO/GRPO 的原 `chunk_step` 路径保持不变。

**已确认数值**：`H_model=50` 不变，`C=10`，sequence 的真实 `h<=10`。RLT 已在同一
π0/RoboTwin 路径验证 C=10 的执行和 canonical 14D；C=10 让 200-step episode 有 20 次闭环重规划，
TD 最长跨 10 步，critic action 输入为 140D，同时不改变 π0 的 50-step 预测头。QAM 历史结果不
用于解释或支持这个选择。

旧方案中的 `gamma_macro`、开 H 次方根与“固定 gamma^50”全部删除。OGPO 配置把 gamma 明确
定义在 primitive waypoint 上；完整 10-step chunk 的 bootstrap 是 `gamma^10`，若第 h<10 步终止
则用真实 h。`terminated` 不 bootstrap；有真实 final observation 的纯 time-limit `truncated` 才
bootstrap。这个**公式与单位**已对齐官方。

数值需要和公式分开：此前 `.99` 是项目首版选择，不是普适官方默认。官方 Square 用 `.99`，
Toolhang/Transport/LIBERO 等长任务使用 `.999`；RLinf PPO 的 `.99` 又是旧 chunk-level 单位。对
200-step sparse RoboTwin，参数审计当前推荐 primitive `.999`，但在用户确认前不把它静默写成
已锁定值。完整来源见 `03_PARAMETER_PROVENANCE.md` §1。

primitive trace 用一组固定动作核对 singleton loop 与原 chunk 路径的终态和终止语义；若不一致，
先修正 adapter。

### 6.5 whole-chain likelihood、旧策略与 SDE correction

先区分四个对象：`raw chain` 是从初始高斯噪声到最终模型动作的完整生成轨迹；准确说它继承
官方逐步 `mean±3σ` noise clip，但处在 final action-bound clip 和环境 transform **之前**。`old_lp`
是 EMA/target actor 对自己采出的这条链的 log-prob；`current_lp` 是 online actor 对同一条链重算的
log-prob。raw final 的前 `C=10` 行、active 14D 作为 normalized canonical `action_q`；同一 raw final
另经现有 π0 output transform 和 14D projection 得到 `action_env`。OpenPI quantile transform 本身不
clip；若 RoboTwin 执行路径另有 action bound，也只作用于 execution copy，不写回 raw chain。

官方 OGPO 在每个 flow step 注入 tapered Gaussian noise，并加 drift correction；target actor 生成
G 条完整随机链和 `old_lp`，online actor 不重新采样，只对相同链计算 `current_lp`。这两个名字沿用
官方代码，但进入 ratio 前已做 normalization：初始高斯项和所有 stochastic transition 的 log
density 先求和，再除以贡献的 chain steps（full-chain 为 K+1）和 flattened action dimension（本项
目为 50×32）。因此 PPO 实际使用：

```text
score_online = joint_logp_online / ((K+1) * (H_model*D_model))
score_EMA    = joint_logp_EMA    / ((K+1) * (H_model*D_model))
ratio        = exp(score_online - score_EMA)

K4,H50,D32: ratio = (p_online(raw_chain)/p_EMA(raw_chain))^(1/8000)
```

所以它保留完整 raw chain，却是每 chain-factor、每 action scalar 的平均 log-score ratio；它不是
论文 Eq. 3.2 未归一化 joint importance ratio 的字面复制。两个 normalization 和 `.01` clip 一起沿
官方可执行语义移植，详细解释见 `03_PARAMETER_PROVENANCE.md` §3。

RLinf π0 默认 non-joint 路径只有一个随机选中的 denoise step 使用 SDE，其余走 ODE；其
`noise_level=.5` 进入的是另一套 `sqrt(t/(1-t))` schedule。已有 `joint_logprob=true` 会让全部 K 步
使用 SDE 并保存完整 chain，但输出仍是逐 action-coordinate 的 PPO 张量，SDE schedule/correction
也不是 OGPO 当前实现。因此新增两个隔离接口：

```text
sample_target_ogpo_chains(...) -> raw_chain, old_lp, raw_final_action
score_online_ogpo_chains(raw_chain, ...) -> current_lp
```

二者返回 `[B,G]` scalar log-prob，并复用 π0 的 batch prefix cache、velocity 和 transform。首版采用
OGPO tapered schedule、全部 K=4 transitions、`sigma_init=.01` 建议值和 score drift correction；
OGPO 的正向时间通过 `t_ogpo=1-t_pi0` 映射到 π0。actor PPO
始终评分完整 `[K+1,H_model=50,D_model=32]` raw joint chain；Q 和环境只消费其确定性投影
`[C=10,D_env=14]`。未执行 suffix/padding 是生成策略的辅助随机变量，不再实现 executed-mask 与
full-joint 两套方案。

其中后 40 个 action token 不进入环境、Q 或 replay；下一次观测后会整段重新规划。它们只作为
生成前 10 步时共同采样的辅助 latent 留在 score-function likelihood 中。π0 的 50 个 action token
在 action expert 内双向耦合，直接只乘前 C10×D14 的条件密度并不是 executed-action 的精确边缘概率；
精确裁掉 suffix 需要积分掉后 40 步，或把策略真正改成 H10。官方 OGPO 通常令生成 horizon 与执行
horizon 一致，并未直接覆盖 H50/C10；RLinf π0 PPO 的 executed-coordinate mask 也不是 OGPO joint
chain 的 oracle。因此当前 full-H ratio 是数学上成立的 augmented-latent score-function 适配，且比
直接截断密度更贴近 OGPO whole-chain 实现；代价可能是额外方差，待出现非零 actor signal 后作为
普通指标观察，不为此预置第二套 runtime。

官方当前 [tapered-SDE sampler](https://github.com/simchowitzlabpublic/OGPO_public/blob/0b3be413cde766a41257c6b19c0c2b06393a557f/ogpo/agents/modules/pg_helper.py#L230-L254)
在最终动作碰到环境边界时有一处记账不一致：`old_lp` 在边界裁剪前的 raw sample 上计算，保存给
current scorer 的 chain final 却已裁剪。例如 raw sample 为 1.01、边界为 1.0，old 和 current 实际
评分了两个不同点；tapered 路径的最后 transition 有非零 sigma，所以这会影响 ratio。π0 适配保留
raw chain 给 old/current likelihood，另生成 `action_q` / transformed `action_env`；同参数、同 raw
chain 时 ratio 应为 1。这也沿用 RLinf π0 PPO 已有的 raw `chains` / transformed env action 分字段
所有权，不是另造表示，也不为 π0 新增一个官方 OGPO 式 `[-1,1]` final clip。

这个修正优化的是 `E_{raw chain}[Q(T(raw final))]`，其中 `T` 是固定的执行前缀/坐标投影及已有
execution transform。即使下游 action bound 使 `T` 不可逆，score-function 仍对 raw latent 分布
成立，不需要给 clip 编造 Jacobian；边界处只可能增加估计方差。首版只保留
`online==EMA -> ratio==1` 的 same-chain fixture，不增加运行 gate 或第二条算法路线。

### 6.6 replay、成功缓冲与初始化

官方初始化是 BC policy + 随机 Q + 空 online replay + 空 success buffer。demonstrations 有三个彼此
独立的可选用途：先做 actor BC、online batch 按 `offline_ratio` 混入 transition、或另做 BC-Q/CalQL
critic pretrain；Q-only warmup 又是用 base actor 新采的 online 数据。发布主脚本虽然先做 actor BC，
却统一使用 `offline_ratio=0`，并关闭 BC-Q、CalQL 与 Q-only warmup。当前项目已有 π0 RoboTwin SFT
checkpoint，只把它作为 actor 参数初始化；其历史训练集名称和条数不进入 OGPO runtime 合同。

**本项目决定**：online replay 和 success view 都从空开始；actor 只从已选 SFT checkpoint 初始化，
不预填 critic replay。EnvWorker 把 C=10 的 primitive rows 放进现有 rollout trajectory；actor
worker 收到一个 rollout batch 后批量写入 ring，不新增跨 worker 的逐步 streaming 通道。官方式
online warmup 完成后才开始更新；当前首版建议 `start_training=20,000` primitive rows，最终写配置
时连同 replay row-size 一次复核。

每个 replay row 只保存：当前/下一 deployable raw observation（三路图像、14D proprio、prompt）、
实际执行的 model-coordinate `[32]` 与 canonical `[14]` action、reward、terminated/truncated，以及
`episode_id/step_id`。auto-reset 时 next observation 使用真实 final observation。source/norm/schema/
world-size 只放 run/checkpoint manifest，不逐 row 重复；首版不做 feature cache。训练采样器从同一
episode 的连续 rows 组成最长 C=10 的 sequence，并返回真实 h、discounted return 与 mask。value、
old log-prob、完整 H50 prediction、denoising chain 和整块 `forward_inputs` 不进长期 replay。

episode 结束且成功时，只把对应 row IDs 加入 success view，不复制图像。首版配置把一次 train
rollout 固定为完整 200-step episode（`auto_reset=false`、C=10），actor 收到的 trajectory 已有完整
成功标签，因此不再设计跨 checkpoint 的 pending-episode 状态。

success BC 从成功 episode 采连续 C=10 个 model-coordinate `[32]` action，按 π0 API pad 成
`[H50,D32]` 输入，并复用 RLinf OpenPI DAgger 已有的 `use_action_chunk_loss`，实际 loss 只覆盖
前 C10×model32；不足 C10 的 episode 尾部不作为 BC 起点。success view 没有样本时 BC 项为 0；
普通 replay 不参与 BC，保持 success-only 定义。

### 6.7 critic、target 与 actor 更新顺序

rollout copy 的 EMA actor 收集真实 primitive trajectory，actor worker 批量写 replay，并按**新增
primitive row 数**累加 Q/PI update credits。官方 `UTD-Q=UTD-PI=1` 表示每新增一行各做一次 update；
不能把一条 200-row trajectory 误算成“一次 TD + 一次 actor”。完整 trajectory 后批量结算 credits
只改变调度时点，不改变 UTD 计数。

论文 Algorithm 2 写的是 `critic -> actor -> targets`，固定 commit 的 fused `_update` 实际是
`actor -> actor EMA -> critic -> target critic`；两者都不是并发。本项目按“发布可执行实现优先”采用
后者，不提供顺序开关。每个 actor credit 从 replay 采 state，冻结 prefix 只算一次，再展开 G 条
EMA stochastic chains；当前 target Q 构造 CA，online actor 对相同 raw chain 重算 score，并联合
success flow-BC 更新，随后更新 actor EMA。接着用 online replay batch 做 TD critic update，再更新
target critic。下一 credit 自然看到刚更新的 target/online 状态。

success BC 与 π0 原始 flow matching 等价，但不再切回一次性 full-model `sft_forward`：冻结 prefix
cache 后，用同一 online action-expert velocity 路径构造完整 H50 noisy action，最后只对 C10×D32
求均值。其 OGPO 坐标 target 是 `action-noise`，等价于 π0 坐标的 `noise-action`；这条路径已在
PPO microbatches 之后的真实两卡 FSDP update 中验证，避免不同 forward 图读取 sharded view。

这不是标准 RoboTwin π0 PPO 已知的普遍故障：当前官方 PPO 配置默认没有打开 success/SFT co-train，
只沿 cached-prefix actor 路径更新；RLinf 虽另有 PPO+SFT 入口，其已发布示例也没有覆盖本项目的
`FULL_SHARD + use_orig_params=true` 组合。本项目是在 8 个 OGPO scorer microbatches 后首次切换到
旧 full-model SFT 图，才暴露两 rank 的 Gemma `q_proj` sharded-view shape mismatch。窄修复不改
FSDP、不关 BC，只把成功 BC 放回 π0 原生的 frozen-prefix/action-expert velocity 路径；准确名称是
“C10-masked π0 flow-matching BC”，不是完整 H50 成功轨迹 SFT。

训练采集用 EMA actor；确定性评估按官方使用 online actor。两者都注册在同一个 OpenPI/FSDP model
中并随现有 weight-sync 一起发送，rollout worker 按 train/eval mode 选角色，不修改 runner 生命周期。

Q 网络具体做成：冻结 π0 prefix 按“三个图像块 + 一个语言块”各 masked-mean pool（复用 QAM
已经实现的 π0 feature tap），再拼 14D proprio 与展平的 normalized `[C=10,14]` action；输入十个
参数独立、FP32 的 Q MLP，每个输出一个 scalar。网络深度优先对齐官方 PaliGemma image critic：
5×512 hidden + LayerNorm；target critics 是独立 Polyak copy。PPO value head 的 `V(s)` 只服务 GAE，
这里需要的是 `Q(s,a[1:C])`，两者不能替换。

同一个 state 的 G 个候选由每个 Q head 分别评分并分别减去该 head 的组均值：
`A[j,m]=Q[j,m]-mean_j(Q[:,m])`。候选 j 的十个 centered advantages 全正时取最小正值；全负时
取最大负值，也就是最接近 0 的负值；符号分歧就置零。这就是 CA，没有 learned value baseline，
也没有额外 CA alpha。得到的标量 CA stop-gradient 后直接进入 same-chain clipped PPO。

TD target 有两种官方常见聚合：`mean` 是十头平均；`subsample` 是随机抽两头取
较小值，比 mean 更保守。论文 Appendix 说同步 JAX 任务中 subsample 经常更好，但逐环境调参；
当前 released [image/PaliGemma runnable config](https://github.com/simchowitzlabpublic/OGPO_public/blob/0b3be413cde766a41257c6b19c0c2b06393a557f/scripts/ogpo/square_image_paligemma.sh#L58-L59)
实际使用 mean。本项目固定 TD=`mean`、CA=完整十头、BoN关闭，不保留 aggregation 运行分支。
PaliGemma runnable 还对每个 next state 采 8 个 next actions 做 Q-target variance reduction；这和跨
heads 的 mean 是另一层。本项目首版明确关闭该额外 8 倍 target sampling，保留标准单样本 TD。

首版结构超参分工如下：十个 Q heads 与按 primitive row 计的 UTD-Q/PI=1 来自 OGPO；π0 denoise
steps 保持服务器 resolved `K=4`，不强改成官方小 MLP actor 的 K=10；`H_model=50,C=10` 来自
π0/RoboTwin 目标系统适配。candidate group 固定首版 `G=8`：官方 practitioner/PaliGemma-Square
使用 32，但官方 LIBERO 使用 8，而 full π0 远重于官方 flow MLP。真实两卡 B4×G8 probe 已落定
flat candidate microbatch=32/rank：每卡 allocated/reserved 峰值约 33.19/36.26 GiB，单 microbatch
约 1.4 秒，same-chain score 差为 0。

G8 在 `B=64` 时已经是 512 条 chain/update；G32 是 2048 条。只有 B 个冻结 prefix 能跨 G 复用，
K4 action-expert generation、online scorer/backward 和 10Q 评分都随 G 近似线性。因此 G32 的
candidate 主体是四倍工作量，不是“batch 后只慢一点”；首版继续采用 LIBERO 的 G8。

group candidates 固定在 actor worker 生成，不进 EnvWorker：它们是从 replay observation 出发的
假想生成链，不调用 simulator。π0 现有 `predict_action_batch` 已支持 observation batch；OGPO 在
actor worker 把 B 个 prefix 展开为 B×G 候选并按候选维 microbatch。EMA 采样用 `no_grad`；online
scorer 对同一批 chain 保留梯度。rollout worker 只生成真实环境要执行的一条 EMA action chunk。

### 6.8 checkpoint / resume

checkpoint 分三层：RLinf DCP 保存同一 FSDP model 内的 online + EMA action expert，以及 actor
optimizer/scheduler/RNG；rank sidecar 保存 EMA FP32 shadow、online/target Q、critic optimizer、replay
ring、success IDs、采样 RNG 与 Q/PI/EMA counters；manifest 保存 source/norm/schema/world-size、
snapshot UUID、step/rows 与 completion。当前已实现 v3 原子 sidecar/manifest、全 rank preflight 与分阶段
恢复；小模型 round-trip、混合 snapshot 预拒绝和写失败无 complete marker 三项测试均通过。

### 6.9 Source-aligned 时间轴、当前 formal 预算与双卡布局

source-aligned 参照只有两个阶段：

```text
Stage A: SFT actor + random Q + empty buffers；收集 20k primitive rows，不更新
Stage B: 从 20k 到建议总量 250k；UTD-Q=1、UTD-PI=1，超参和 LR 全程不切换
```

用户于 2026-08-07 批准并已启动的一日预算 formal 保留同样的两阶段语义，只改四项预算：

```text
Stage A: rows 0..9,999；只采集，不更新
Stage B: rows 10,000..34,999；UTD-Q=.1、UTD-PI=.1
replay capacity: 40,000
```

因此本次严格写入35,000 primitive rows，产生2,500 actor updates与2,500 critic updates（合称
2,500 paired updates），以及 `2500×64×8=1,280,000` 条 imagined group chains。该数值明确标为
完整 π0 的计算适配，不改称 OGPO 官方规模；source-aligned 20k/250k/UTD1 仍保留为参照而非当前运行值。

`bc_pi/bc_q/calql/q_warmup/bc_refine` 全部为 0；success BC 在 success view 首次有样本后自然加入，
不另设阶段。250k 来自论文 LIBERO 的 image+language 近邻，而不是 PaliGemma Robomimic 的 2M；
它对应约 230k Q 和 230k actor credits、1,250 个满长 episode（提前成功时 episode 数更多）。
train metrics 每个完整 rollout batch 记录；本次保留 source YAML 的 eval/checkpoint 20k/50k，eval
每点20 episodes。故 baseline@0、首次跨20k、final@35k共三次评估；50k大于35k，中途没有恢复点，
runner 在 exact35k 执行唯一 final checkpoint。

真实 row-size probe 显示三相机 current/next observation 约需 1.44 MB RSS、1.386 MB sidecar/row：
250k ring 约 334.98 GiB RAM、单个满 ring sidecar 约 322.79 GiB。RAM 可容纳，但若每 50k 保留一份
完整 replay checkpoint，累计磁盘会超过当前 816 GiB。故 source-aligned `replay_capacity=250k`
与50k完整 sidecar不能保留所有历史快照。本次 formal 已收敛为 capacity40k、total35k、只在final
保存一次：按 row probe，35k replay 约46.90 GiB RAM/45.19 GiB sidecar，40k capacity ceiling约
53.60/51.65 GiB；加已测约13.13 GiB checkpoint固定底座，final落盘粗估约58.3 GiB。它解决本次
一天运行的窗口/磁盘问题，不反向证明250k长期路线已解决。

双 A800 首版布局是：actor/env/rollout 均在 GPU0–1，FSDP/rollout world size=2，无 TP/PP，
`pipeline_stage_num=1`；train 为 8 env×1 rollout wave，eval 为 4 env×5 waves；rollout、train、eval
三段不重叠。逻辑 state batch B64、G8、10Q vectorized；同一 state 的完整 G8 留在同一 rank，flat
candidate microbatch 固定为 32/rank。K4 denoise、C10 environment execution 和 optimizer
credits 在时间上串行。完整参数和来源见 `03_PARAMETER_PROVENANCE.md` §6–§7。

250k/UTD-PI=1 在 B64/G8 下等于 1.1776 亿 imagined chains；这是 source-aligned 预算的计算轴。
真实两卡一个 production paired credit 已测得：TD next action 2.00 秒、actor 9.85 秒、critic
0.89–0.90 秒，即约 12.75 秒/update，峰值约 35.31/39.57 GiB allocated/reserved。若把约 230k
joint credits 原样串行执行，更新计算本身约 33.9 wall-days、1,629 A800 GPU-hours，尚未计 simulator、
eval 与 checkpoint。因此 250k 仍是 source-aligned 交互建议，但已不能称作待直接启动的冻结 formal
预算；正式包必须明确接受该成本，或依据来源重新讨论 interactions/UTD，而不能静默缩短。pilot/
formal 审批包仍会同时列 interactions、episodes、policy queries、Q/actor updates 与 chains。

### 6.10 首轮实证后的预算耦合与第二轮落地

本轮最终用了26个外层cycle。每个cycle先同步权重，再并行模拟8条最长200-step episode；完整
trajectory回到actor后只写valid primitive rows，并按warmup后的新rows累积UTD credit，随后把本cycle
全部整数credit连续跑完，最后才检查eval/checkpoint阈值。典型cycle写入1,242–1,600 rows，故UTD`.1`
实际形成约124–160个连续paired updates；第25 cycle写1,107 rows到34,968并跑110次，第26 cycle
只写补足quota的32 rows并跑4次，最终严格2,500 updates。35k的175个满长episode-equivalents与
26×8=208条实际train episodes因此不矛盾。

本轮fixed eval为`5%@0 -> 35%@20,081 -> 5%@35,000`。把total从250k缩到35k时，只改用户批准的
四项预算、保留source YAML的eval20k/checkpoint50k，虽已在启动包披露“三次eval、final-only save”，
但没有重新论证它们与短预算的耦合；这是计划疏忽。直接后果是无法定位20k后何时退化，也没有保存
20k附近的最佳模型。以后每次改变total/warmup/UTD，必须同时重算eval点数、可恢复checkpoint数、
累计磁盘、wall和early-stop口径，不能再把运行频率当成无关默认值。

用户把下一轮频率收敛为**约10次fixed-policy eval、约3次完整resume checkpoint**，两者不再绑定。
本轮58.23 GiB final也不是“π0模型本身变成58 GiB”：FSDP/DCP两片共约10.27 GiB，两个OGPO
sidecar共约47.96 GiB；其中35k replay的三相机current/next observation等字段按probe约45.19 GiB，
其余才是Q/target-Q、optimizers、EMA shadow、success IDs、RNG与计数器。完整checkpoint之所以大，
主要是为了精确resume而携带在线replay；每次eval若只需选best，应另导出较轻model-only权重，不能
冒充完整恢复点。

按本轮实测速度，第二轮**fresh SFT actor + empty replay**的24小时主案为：total90k、warmup10k、
paired UTD`.05/.05`、capacity100k、eval interval10k、checkpoint interval30k。它会得到
`0,10,...,90k`共10个eval点，30/60/90k共3个完整checkpoint，4,000 paired updates和约204.8万
imagined chains；线性名义wall约23.71小时，考虑约±10%波动不视为硬上界。用户明确说明24小时只是
大概值并批准该主案，故未切换到85k，也没有设置24小时自动终止。若以后需要硬截止，
保守候选改为total85k/capacity90k，其余相同，预计22.35小时，eval仍为10点、checkpoint仍为3份。
90k三份完整checkpoint粗估累计约271.8 GiB；启动前live核对确认现有磁盘容得下。该主案已经形成
resolved packet并于13:27:34启动；`resume_dir/ckpt_path=null`，没有读取35k final，避免
从本轮退化策略续训而污染UTD对照。resolved SHA为`352f8e80752d60624a0c53c62d21dcc10bdc8e6712a433c18d0eec0ee1f56a36`。

UTD`.1`的final结论不是“已最优”：ratio最低`.133`、final`.942`，且20k后fixed eval回落，支持把
actor更新过密/EMA滞后列为下一轮主假设；但同样长度burst下ratio后期持续恢复，不能仅凭均值断言
总UTD就是唯一原因。源码调度对照进一步确认：官方OGPO在warmup后通常每个primitive `env.step`
后立刻做1次paired update；官方RLinf π0 PPO一次大rollout后约连续4次optimizer step；本项目因完整
8-env trajectory批量返回，UTD`.1`才形成124–160次连续paired updates。下一轮最简单的对照是paired
UTD`.05`，把burst降到约62–80次、墙钟主成本减半；`.05`不是官方超参，也不是C10推导值，而是
当前bulk-runner与完整π0的最小计算/调度适配。更针对actor drift的是保留Q`.1`、PI`.05`，但当前
worker要求二者相等，需要小幅调度改造和测试。
`.025`只剩625次联合更新，当前证据不足。用户选择先用当前可见性运行fresh`.05`对照，不在启动前
修改训练代码；ratio分位数/clip fraction、CA与Q排序等更细指标留给本轮现有产物能回答的范围和后续实现。

对“62–80次连续paired updates是否离谱”的进一步源码审计给出更窄结论：它不是计数/单位错误，
也有几十次actor+critic burst先例，但不是OGPO官方的交错节奏。官方OGPO在每个primitive
`env.step`后调用update，episode结束才把当前trajectory flush进replay；真实交互仍夹在updates之间。
[OpenAI Spinning Up SAC](https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/sac/sac.py#L318-L322)
默认先积累50步再连续做50次完整Q+actor+target update；
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3/blob/v2.7.0/stable_baselines3/common/off_policy_algorithm.py#L334-L354)
也明确支持`collect_rollouts -> gradient_steps`。REDQ/RLPD的高UTD主要增加critic、actor通常只更新一次，
不能替本项目的paired actor burst背书。故`.05`仍标为“有通用off-policy先例、未经OGPO+完整π0原样验证”
的项目适配；本轮用fresh起点做单变量对照。它表示每20个新rows做1次update，不表示每条row训练80遍。

并行轴首轮无异常：2×A800、train env8、B64/G8、flat32/rank、10Q vectorize在第二轮保持不变；
GPU峰约57.2/57.7 GiB、OOM 0。除非出现明确吞吐问题，不因ratio先改并行。总wall 44,622秒中，
rollout 12,714秒（28.5%）、paired training 30,733秒（68.9%）、三次eval约833秒（1.9%）；
update内部actor候选链与BC约占主要部分，故首要串行成本是actor update，不是环境采集或critic。

### 6.11 “RL起点低”必须按策略与评估协议拆开

旧π0 PPO/GRPO与DSRL、RLT、OGPO的首点不是同一个量，不能把它们合成一条“RL先把SFT打坏”
的证据链：

| 路线 | 首点实际测谁 | 已确认的关键协议 | 当前解释 |
|---|---|---|---|
| π0 PPO / GRPO历史 | 未更新的原始SFT π0 | C50；首批随机train rollout各256条，成功率78.1%/83.6% | 证明旧C50 train起点高，不是配对fixed eval |
| DSRL | 新latent actor接冻结π0 decoder | H50/N20；warmup latent沿H重复；首个formal eval已在800次SAC update后 | 不是原生π0的update-0 baseline |
| RLT | fixed eval从第一点就测fresh student MLP | C10；train warmup另由frozen π0 reference控制 | 早期0/20是student，不是SFT基线 |
| OGPO | SFT初始化的online π0（EMA初始同权重） | H50/C10；`use_ogpo=true`时绕开native ODE/SDE入口，改用每个flow step注入σ=.01的OGPO tapered-SDE sampler | 5%@step0是真低点，但同时改变C与sampler |

[RLinf官方RoboTwin结果](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html#visualization-and-results)
给出的adjust_bottle π0 SFT/PPO是76.56%/98.44%，其公开PPO配置使用C50和128个fixed reset IDs；
这支持用户记忆中的高SFT基线，但样本、随机性和执行协议不等于上述各首点。
[πRL Table 2 / Figure 11 / Appendix G](https://arxiv.org/html/2510.25889)进一步显示：同一个未更新SFT
在随机train与确定性eval之间可出现9.4%对63.8%的巨大差距；C5/C10/C20会明显影响结果，但论文的
SFT消融65.2/70.5/72.6%并不支持“C10必然把76%降到接近0”。原π0也通常只执行H50预测的前16或
25步再规划，而非固定执行50步。因此C50→C10/C20是重大混杂项，不是已证实共同原因。

当前最小因果审计固定同一SFT checkpoint、fixed reset/policy RNG、`demo_clean`、`unnorm_key`和
确定性native sampler，先测native-C50/C20/C10；再保持C不变比较native与OGPO sampler。每条路线
同时登记“原始base、算法接入但0 update、首个随机train rollout、首批update后fixed eval”四个点。
在完成这一配对前，RL改善只能称为适应新控制/采样协议，不能直接宣称超过原C50 SFT。

## 7. 自顶向下实施结构

### 7.1 以 RLinf π0 PPO RoboTwin 为起点，但替换算法内核

保留的主体链：

```text
train_embodied_agent.py selects EmbodiedOGPOFSDPPolicy
  -> existing EmbodiedRunner.run
     -> sync online+EMA expert model to MultiStepRolloutWorker
     -> rollout worker batch-predicts H50 actions
        train mode uses EMA; eval mode uses online
     -> training EnvWorker executes C10 through singleton RoboTwinEnv.step
        eval keeps the existing chunk_step path
     -> existing Trajectory channel returns primitive rows
     -> actor worker batch-ingests replay
     -> target chain G8 + target-Q/CA + online same-chain PPO + success BC
     -> update EMA expert
     -> critic TD update -> update target Q
     -> save DCP/sidecar/manifest
```

因此起点是 RLinf π0 PPO 的**调用骨架和模型 adapter**，不是复制它的 GAE、value head、current-
rollout minibatch 或 old-logprob 语义。详细 path/symbol 见 `02_CALL_AND_DATA_FLOW.md`。

### 7.2 已实施的文件级接缝

服务器 source lock 下的实际责任划分为：

| 目标位置 | 改动与输入输出 | 主要参考 |
|---|---|---|
| `examples/embodiment/config/robotwin_adjust_bottle_ogpo_openpi.yaml`（新） | 唯一 OGPO+CA opt-in 配置；复用 PPO 的 env/model/placement；固定 C10、完整 200-step rollout、无 auto-reset | RLinf π0 PPO + OGPO config |
| `examples/embodiment/train_embodied_agent.py`（薄改） | `loss_type=embodied_ogpo` 选择新 actor worker | 现有 SAC/RLT dispatch |
| `rlinf/config.py`（薄改） | 校验 OpenPI、C10/D14、无 value head、非 pipeline 等结构合同 | RLinf config validator |
| `rlinf/workers/rollout/hf/huggingface_worker.py`（薄改） | OGPO rollout 不请求 value/bootstrap；继续批量 π0 推理和通用 weight sync | RLinf rollout worker |
| `rlinf/workers/env/env_worker.py`（薄改） | `action[:10]` 逐个 singleton step，形成 primitive obs/reward/done rows；旧 `chunk_step` 不变 | OGPO action queue + RoboTwin step |
| `rlinf/models/embodiment/base_policy.py`（薄改） | 增加隔离的 OGPO flow forward type | RLinf model dispatch |
| `rlinf/models/embodiment/openpi/{__init__,openpi_action_model}.py`（薄改） | SFT 加载后初始化 EMA expert；train/eval 分别调用 EMA/online；接 sampler/scorer hook | RLinf π0 ownership |
| `rlinf/models/embodiment/modules/ogpo_modules.py`（新） | EMA action expert、FP32 shadow、坐标/时间适配；仅 OGPO 下修复 tied PaliGemma head 的 FSDP ownership | OpenPI π0 + QAM 同构 FSDP 窄先例 |
| `rlinf/workers/actor/fsdp_ogpo_policy_worker.py`（新） | 持有 online/EMA expert、online/target Q、replay、optimizers；执行 TD 和 actor update | RLinf FSDP worker + OGPO agent |
| `rlinf/algorithms/ogpo/core.py`（新） | h-step target、CA、whole-chain PPO 与 success-BC 合并的纯函数 | OGPO 官方公式 |
| `rlinf/data/ogpo_replay.py`（新） | primitive transition、连续 sequence、bounded ring、success membership、save/load | RLinf/QAM replay 形态 |
| `rlinf/models/embodiment/modules/ogpo_critic.py`（新） | 10-head FP32 Q 与 target copy，输入 prefix+proprio+`[10,14]` | OGPO image critic + π0 feature tap |
| `rlinf/models/embodiment/openpi/openpi_ogpo.py`（新）及 `openpi_action_model.py`（thin hook） | target full-chain sampler、online same-chain scorer，输出 `[B,G]` log-prob；cached-prefix success BC 复用同一 FSDP-safe online velocity | OGPO `pg_helper.py` + RLinf/OpenPI π0 |
| `rlinf/runners/embodied_runner.py`（薄改） | 按 online primitive rows 调度 eval/checkpoint/stop，resume 后对齐下一阈值并补 final eval | RLinf runner + OGPO row budget |
| actor worker 的 save/sync 方法（扩展） | 把非训练 EMA 参数纳入通用 sync；override save/load 保存 DCP+sidecar+manifest v3 | RLinf sync/DCP + DSRL/RLT resume |
| `tests/{algorithms,data,embodiment,workers}`（新） | 数学、replay、critic、adapter、checkpoint、玩具/真实两卡 FSDP 定向验证 | 各组件对应 oracle |

`RoboTwinEnv` 与 `embodied_io_struct.py` 不改；`EmbodiedRunner.run` 只增加 OGPO 的 row-based
调度薄分支。专用 actor worker 实现既有
`recv_rollout_trajectories / compute_advantages_and_returns / run_training / save/load` 接口；EnvWorker
把 `obs_trace/reward/done/valid/actions` 挂到现有 `Trajectory.forward_inputs`。

## 8. 实施与验证阶段

1. **live source freeze（已完成）**：实施基点为服务器 `/root/autodl-tmp/RLinf@6d0db56b`；独立
   `/root/autodl-tmp/RLinf_ogpo_pi0_robotwin`、branch `codex/ogpo-pi0-robotwin` 已创建，公共 dirty
   worktree 未改，也未从 DSRL/RLT/QAM 分支起步。
2. **连贯主体实现（已完成）**：opt-in config、primitive trajectory、actor-side replay、10Q、完整
   raw-chain sampler/scorer、CA/PPO/success-BC、target sync 和 checkpoint 已接通；没有旧路线或备用算法。
3. **服务器集中检查（正式 smoke 前已完成的范围）**：
   - OGPO 数学：h-step TD、CA、tapered SDE、same-chain ratio；
   - primitive trace tensor/schema 与 mixed early-done `final_obs` 合并 fixture；
   - opt-in 集成：真实 checkpoint 单卡语义、真实两卡 FSDP B4×G8 same-chain backward，以及
     B64/G8 production `target -> actor+BC -> critic` 完整更新；
   - save/load：EMA shadow 与 checkpoint sidecar round-trip/失败原子性。
   RoboTwin singleton trace 的真实 simulator 执行此前不在前置测试中，由下一项真实 smoke 闭合。
4. **首次 end-to-end smoke（已批准并完成）**：resolved config 锁定为 2×A800、train 8 env、
   `C10/B64/G8/flat32`、80 primitive rows、恰好 1 个 paired update、4-env C10 eval 和 1 个 checkpoint；
   exit 0，资源/sidecar/TensorBoard 合同均通过。逐命令见
   [`COMMAND_AND_CHANGE_INDEX.md`](evidence/COMMAND_AND_CHANGE_INDEX.md)。
5. **一日预算 formal（已批准并完成，exit 0）**：resolved SHA
   `77419258766880fca7b8d15d725dfb9f2fba5b88d2bfe21eb918fd760e9a7bc9`；仅把
   total/warmup/paired-UTD/capacity改为35k/10k/.1/40k，其余保持 source YAML。23:13:50 启动唯一
   driver和独立1秒监控；最终35k rows/2,500 updates、26 waves/208 episodes，fixed eval
   5%→35%→5%，OOM=0，完整`global_step_26` checkpoint。完整逐命令见
   [`COMMAND_AND_CHANGE_INDEX.md`](evidence/COMMAND_AND_CHANGE_INDEX.md) 的 FRM 系列。
6. **第二轮约24小时 formal（已结束，部分完成）**：fresh SFT、空replay；90k total、10k warmup、
   paired UTD`.05`、capacity100k、eval10k、checkpoint30k；resolved SHA `352f8e80…f56a36`。
   2026-08-08 13:27:34启动，2026-08-09 06:03:27以exit255结束；最终提交64,078 rows、2,703 paired
   updates。30k/60k两个完整checkpoint可恢复；Ray在228.21/240 GiB时主动杀worker，随后NCCL等待
   超时。根因是完整replay checkpoint后actor RSS阶梯式增加，不是GPU OOM。逐命令见命令账本FR2系列。

## 9. 决策登记与剩余实现数值

| 项目 | 当前结论 | 状态 | 还需什么 |
|---|---|---|---|
| 主方法 | 只提供 OGPO+CA；无 vanilla/OGPO+ runtime | 已决定 | 无 |
| π0 trainable scope | frozen VLM + full action expert/projections；无 value head | 已决定 | 无 |
| critic | 10 个独立 FP32 5×512 Q；四块 frozen prefix + proprio + `[10,14]` action；mean TD target | 已实现/结构测试 | 真实 prefix block 长度 `(256,256,256,48)` |
| critic observation | deployable frozen π0 feature + proprio；无 privileged state | 已决定 | 无 |
| chunk 数学 | `H_model=50,C=10,h<=10`；primitive trace；`Σγ^i r_i + γ^h Q` | 已实现并经真实 C10 smoke | formal沿用不变 |
| replay 初始化/ingest | 每轮actor从SFT fresh；online/success为空；trajectory到actor后批量写ring | v2在64,078 rows部分结束 | 60k checkpoint可恢复；续训前先修checkpoint RSS |
| whole-chain score ratio | EMA 采完整 raw `[K+1,50,32]` chain，online 同链评分；两项除以 `5×1600`；Q/env 投影 `[10,14]` | 已实现/真实模型验证 | two-rank same-chain delta=0 |
| group/UTD | G=8；首轮`.1`完成2,500 updates；第二轮`.05/.05`完成2,703/4,000；source参照为1 | v2部分完成 | `.05`降低计算但ratio按累计update与v1近似；不据此单独归因 |
| success Q / BoN | 关闭 / BoN=1；success-only BC + CA 保持开启 | 已决定 | 无 |
| 训练阶段/规模 | v2为10k纯收集+joint phase；无BC-Q/CalQL/Q-warmup；actor-first顺序 | 64,078/90,000 rows部分完成 | 已有7个fixed eval点；没有70k以后结论 |
| 训练数值 | state batch=64、actor/critic constant LR `5.6e-6/3e-4`、tau `.005/.05`、clip `.01`、BC coeff 1.0 | 来源已审计 | 见 `03_PARAMETER_PROVENANCE.md`；实现配置审阅时冻结 |
| 双卡并行 | train 8 env×1 wave；eval 4×5；FSDP/rollout world=2；B64/G8；flat32/rank；pipeline=false | v2保持不变且启动健康 | 只有明确吞吐问题才做单变量probe；不因ratio先动并行 |
| replay/checkpoint 资源 | v2 capacity100k、eval10k、checkpoint30k | 暴露CPU RSS问题 | 两次save后actor RSS约49→89→168 GiB；续训前需流式/无整份clone保存 |
| 服务器基线 | `/root/autodl-tmp/RLinf@6d0db56b`；独立 OGPO worktree；复用原 `.venv` | 当前无训练进程 | 代码`5d5c84e3` clean且已推；v2有30k/60k恢复点 |

## 10. 当前停点

- formal v2从`2026-08-08T13:27:34+08:00`运行到`2026-08-09T06:03:27+08:00`，exit255；当前无
  driver/monitor。最后完整提交64,078/90,000 primitive rows、2,703/4,000 paired updates，严格满足
  `floor((64078-10000)*.05)=2703`。共有45个已记账8-env waves，即360条train episodes，99成功
  （27.5%）；失败前可能另模拟了一波，但未ingest/记录，不计入训练账。
- fixed online eval完整可见曲线为`5%@0 -> 5%@10,088 -> 15%@20,360 -> 30%@30,959 ->
  10%@40,995 -> 30%@50,735 -> 40%@60,968`。它显示有学习信号但波动大；因异常结束，没有70k、80k、
  90k/final点，不能称收敛。latest ratio/actor loss/combined grad/BC为`.944/3.63e-5/.207/.025`；critic
  loss/grad为`.011/.536`，Q/TD mean均`.059`，全部有限。ratio从`.123`恢复，但现有均值仍不能替代
  clip fraction/分位数；Q≈TD也不证明candidate排序正确。
- Ray在05:33:37观察容器内存`228.21/240.00 GiB=95.09%`，达到`.95`阈值后主动杀掉一个ChannelWorker；
  collective断裂使另一rank等待，30分钟后NCCL watchdog超时，driver退出255。kernel cgroup的
  `oom=0/oom_kill=0`是因为Ray在kernel OOM前主动处置；后续NCCL错误是结果，不是根因。GPU峰仅
  57,477/57,464 MiB，故不是显存问题。
- 根因线索已经落到checkpoint路径：30k save前后actor RSS约48.6→89.1 GiB，60k save前后约88.8→
  168.5 GiB；两次增量分别接近42.7/81.5 GiB replay sidecar。`OgpoReplay.state_dict()`对每个slot
  clone CPU tensors，再一次性组装`sidecar_state`交给`torch.save`；返回后这些逻辑临时对象应失效，
  但进程RSS没有回落（可能是allocator arena保留，不能冒充已证明的活引用泄漏）。续训前应改成流式/
  分块保存、避免进程内整份clone；仅`del/gc/malloc_trim`可作窄验证，不是稳健主方案。
- 两个完整恢复点仍在服务器：30,959 rows的`global_step_22`约53.01 GiB，60,968 rows的
  `global_step_43`约91.78 GiB，`complete=true`。当前没有授权恢复或重启；不修保存路径直接从60k续跑
  仍可能再次撞内存阈值。
- 截至最后metric的有效wall为15:55:25：rollout22,125.5秒（38.6%）、paired training32,830.6秒
  （57.3%）、eval1,723.7秒（3.0%）、其他645.2秒（1.1%）；异常后等待另有40:29。两卡平均util约
  69.1%/73.4%，cgroup峰236.71 GiB，监控全程kernel OOM计数0，数据盘最低可用638.22 GB。
- 本机轻量终态包为`exports/ogpo_formal_90k_partial_64078_20260809_v2.zip`（2,430,833 bytes，SHA256
  `13e4e7b258cb800f6d95a077c1e3303b8fe5ad87939b36eddedeaf9535c8bcc4`），含原始driver/metrics/
  TensorBoard/1秒资源/config/provenance、两个checkpoint completion manifest、派生表和两张PNG；不含
  约145 GiB checkpoint正文或replay。
- 目标实现worktree仍为clean `5d5c84e3ac4efa1713a4139a05ac1b776e634ed3`，upstream 0/0；
  `personal/codex/ogpo-pi0-robotwin`远端HEAD相同，故实现代码已上云。本地根证据仓仍是无remote、无tracked
  file的initial repo，所以近期实验日志、指标包、图、文档和helper没有推云端。
- v1与smoke的完整历史仍由本节之前的阶段登记、`IMPLEMENTATION_LOG.md`和轻量包保存，不再在当前停点
  重复展开。下一步先讨论checkpoint保存修复与60k恢复策略；若要判断原SFT为何在新路线step0低，按§6.11
  做配对0-update协议审计。
