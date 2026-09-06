# Idea2 DVAC × π0-GRPO 训练规划流水账

日期：2026-08-20  
范围：本地与公开一手来源的只读调查、方案整理和文档更新。  
未发生：服务器登录/写入、代码实现、compose/import/test、smoke、训练、进程控制、安装、下载或checkpoint加载。

## TP-001 — 读取当前工作区规则与专题事实源

- 操作：完整读取根目录 `PROJECT_CONTEXT.md`、`HANDOFF.md`，随后读取当前专题
  `00_INDEX_AND_PLAN.md`、`01_SIGNAL_AND_DATA_CONTRACT.md`，并复核首轮分析与实现审阅章节。
- 目的：延续已经完成的推理telemetry/smoke/64-query分析，不重做旧工作，不把训练计划写进推理数据合同。
- 结果：当前服务器Idea2权威source为`61996e15…`；首轮数据最重要的未解因素是future-h位置效应；训练
  telemetry和权重此前均明确未实现。

## TP-002 — 本地Git状态检查

- 首次命令：`git status --short`。
- 结果：因当前sandbox Windows账号与目录owner不同，被Git `dubious ownership`检查拒绝。
- 解决：未修改全局Git配置；后续仅在单条命令中使用
  `git -c safe.directory=C:/Users/86136/Documents/rl status --short`。
- 复测：成功；根仓仍无可用tracked基线，现有专题目录和worktree均属于用户内容，本轮不清理、不提交。

## TP-003 — 恢复历史100-step π0-GRPO配置、命令和结果

- 读取：
  - `audits/20260717-084926-grpo-current/resolved-config.yaml`
  - `audits/20260717-084926-grpo-current/command.txt`
  - `audits/20260717-084926-grpo-current/analysis.json`
  - `audits/20260717-084926-grpo-current/remote-snapshot.txt`
  - `audits/20260717-084926-grpo-current/peak.txt`
- 辅助命令：`rg`只检索GRPO、batch、H/C/D/M、clip、offload、placement、seed等目标字段；PowerShell
  `ConvertFrom-Json`只抽取最后5步、peak与fatal计数。
- 结果：锁定两卡、16 env×16 rollout epoch、G8、B512/mb32、H=C50/D14/M4、flow_sde、chunk
  reward/logprob、GRPO、update2、lr5.6e-6、clip_grad1、100 steps/save10。
- 结果：运行自然到100；step100 rollout success 98.4375%；GPU峰约30 GiB/卡，cgroup峰241999 MiB，
  fatal/OOM计数为0。
- 边界：旧success是训练rollout指标，不是fixed-ID held-out eval；本轮没有live重查checkpoint正文。

## TP-004 — 训练代码shape与loss聚合只读审计

- 目标镜像：`.idea2-dvac-impl-worktree`；它当前只是服务器实现的本地文件镜像，Git HEAD仍显示base
  `6d0db56…`并含既有未提交文件，不能写成服务器`61996e15…`。
- 读取/检索：
  - `rlinf/models/embodiment/openpi/openpi_action_model.py`
  - `rlinf/workers/rollout/hf/huggingface_worker.py`
  - `rlinf/data/embodied_io_struct.py`
  - `rlinf/workers/actor/fsdp_actor_worker.py`
  - `rlinf/algorithms/utils.py`
  - `rlinf/algorithms/losses.py`
  - `rlinf/hybrid_engines/fsdp/fsdp_model_manager.py`
- 结果：rollout/actor在chunk reduction前均保留`logprob [B,50,14]`；历史配置在loss前把H,D求和成
  每query一个joint ratio。直接切`action_level`会改变ratio/clipping与loss尺度。
- 结果：train rollout使用flow_sde和随机`denoise_ind`；endpoint必须当场计算，不能从actor replay假装
  无损重建。
- 决策：首版保持chunk-level前向目标，在求和前使用detached straight-through `w[h]`只改反向贡献；
  `w=1`与off必须退化旧路径。

## TP-005 — DVAC与80/20系列一手来源核对

- DVAC：<https://arxiv.org/html/2606.03847v1>。
  - 结果：training-free inference；记录`V(h),V_total,tau,N_exec`及phase/成功/执行长度；没有训练权重。
- Beyond 80/20：<https://openreview.net/forum?id=yfcpdY4gMP>；官方代码
  <https://github.com/Shenzhi-Wang/Beyond-the-80-20-Rule-RLVR>。
  - 结果：categorical entropy、batch top-20%硬mask、按selected token数归一；官方README要求
    `137d8d1…`或更新commit。
- A3PO：<https://aclanthology.org/2026.acl-long.134/>；代码
  <https://github.com/txy77/Sample-Polarity-of-RLVR>。
  - 结果：正负rollout对confidence的含义不同；主设置被选token初始2倍并衰减到1。
- STEER：<https://aclanthology.org/2026.acl-long.1436/>；代码<https://github.com/zz-haooo/STEER>。
  - 结果：按估计entropy change做连续权重；论文主区间约0.7到1，hard binary消融更差；categorical
    一阶公式不能直接移植到flow。
- LESS：[ACL Findings论文](https://aclanthology.org/2026.findings-acl.1650/)与
  [作者代码](https://github.com/QWE-CXZ/LESS)。
- HTMR：[ACL论文](https://aclanthology.org/2026.acl-long.910/)与
  [作者代码](https://github.com/York-Gold/HTMR-EAE)。
- ResT：[ICLR OpenReview](https://openreview.net/forum?id=gNZlaKRWki)与
  [作者代码](https://github.com/1229095296/ResT_Tool_use_LLM)。
- DPPO：[ICLR论文页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c0749c39aaff9e9e4c91f7118bf21b1e-Abstract-Conference.html)
  与[作者代码](https://github.com/irom-princeton/dppo)。
- πRL：[论文](https://arxiv.org/html/2510.25889)与[RLinf](https://github.com/RLinf/RLinf)。
- 结果：这些工作分别支持polarity、低不确定任务关键位、连续权重、mean-one或denoise-step weighting；
  “连续、最终有界、per-query mean-one”是本项目综合STEER与ResT形成的工程选择，不表述为共同原公式。
  DPPO是denoise-i粒度，不证明future-h credit已经现成。
- 临时源码检查：只读clone
  `C:/Users/86136/AppData/Local/Temp/codex_pns_rlvr_20260820@9a8cb393…`
  （`Harryking1999/PNS_RLVR`公开复现路线）和
  `C:/Users/86136/AppData/Local/Temp/codex_less_20260820@ed929086…`；两者clean，只写系统Temp，未写workspace。
  Beyond 80/20的论文作者主仓仍以`Shenzhi-Wang/Beyond-the-80-20-Rule-RLVR`为引用权威。
- 网络问题：无。

## TP-006 — 形成最小实现、日志和验证计划

- 新建：`05_TRAINING_MODIFICATION_PLAN.md`。
- 核心决定：
  1. 复用旧成功GRPO全部任务、模型、并行、batch、optimizer、FSDP和预算参数；
  2. 新增`off/observe/apply`，先observe真实train-SDE signal；
  3. 全量存紧凑`V[L,H]`，raw z只抽样，不存全模型逐action梯度；
  4. T1先采raw signal，T2再产出带hash的`(h,denoise_ind)` baseline artifact；先消除这些nuisance，
     再讨论温和连续权重；20%不是机器人常数；
  5. 实现后仍需一个双卡完整runner-step smoke，随后才冻结正式映射。
- 本轮边界：只写本地规划文档；没有进入代码或服务器运行。

## TP-007 — 用户后续冻结更直接的首版训练选择（取代TP-006路线）

用户在后续讨论中明确：第一目标是把“内部不确定性小幅重分配action-level梯度”干净接入已跑通GRPO，
先完成修改版训练链路，不让per-h去趋势、`(h,denoise_ind)` artifact、mean-one或严格多run A/B成为前置。
因此TP-006第2–5项中的`observe→artifact→apply`路线只保留为设计演进，不再是当前计划。

当前冻结为：

1. 只有`off|apply`；`apply`内部第1个runner step以`w=1`照常GRPO并收集统计，第2步起加权。
2. `y=log(V_L3+1e-12)`；近期分布来自最近5个已完成step中、进入trajectory的4个action-query之全部
   `query×h`，不按loss mask筛、不包含terminal/bootstrap forward。
3. `z=(y-mu)/max(sigma,1e-6)`，`s=clip(z,-2,2)`，`w=1+0.1s∈[0.8,1.2]`。
4. online首版有意保留future-h位置效应；raw与per-h residual继续离线并排分析。
5. 不切`action_level` PPO；在`[B,H,D]`log-prob求和前用straight-through只改反向的per-h贡献，旧joint
   ratio/clip、advantage方向与global grad clip保持。
6. train-SDE `denoise_ind`继续记录作诊断，但不进入首版权重条件统计。
7. 实现、必要server前测和2-step warmup→apply smoke已获本轮授权；正式30/50/100-step训练须在smoke
   分析后重新讨论。

从第一次实现操作起，所有服务器命令、改动、问题、修复和复测转入
[TRAINING_IMPLEMENTATION_AND_SMOKE_LEDGER.md](TRAINING_IMPLEMENTATION_AND_SMOKE_LEDGER.md)，不在本历史规划账
重复维护动态状态。

## TP-008 — 收窄到当前RLinf逐action梯度挂点的近期文献复核（2026-08-23）

- 用户纠正调研范围：不再以“GRPO有哪些五层改法”为主线，只看与当前
  `per-action policy-gradient contribution x detached w_h`相同或非常接近的接口。
- 本地只读复核：
  - `tmp/idea2_residual_downweight_impl_source/rlinf/rlinf/algorithms/dvac_train_weighting.py`；
  - `tmp/idea2_residual_downweight_impl_source/rlinf/rlinf/workers/actor/fsdp_actor_worker.py`；
  - v3 formal YAML。
- 结果：当前挂点是`logprobs[B,H,D]`在joint query聚合前的straight-through梯度倍率；前向log-prob、
  ratio和query-level PPO clip不变，未被PPO截平时局部梯度等价于`A_eff(q,h)=A_q*w(q,h)`。
- 公开一手来源逐篇核对：
  - GRAIL：<https://arxiv.org/html/2606.04889>；
  - Covariance-Aware GRPO：<https://aclanthology.org/2026.acl-short.45/>；
  - STEER：<https://aclanthology.org/2026.acl-long.1436/>；
  - A3PO：<https://aclanthology.org/2026.acl-long.134/>；
  - OAR：<https://aclanthology.org/2026.acl-long.1132/>；
  - THR：<https://proceedings.iclr.cc/paper_files/paper/2026/hash/c6989f4c36acb6f0e0fdd60f1c12e8a0-Abstract-Conference.html>；
  - Beyond 80/20：<https://openreview.net/forum?id=yfcpdY4gMP>；
  - HTMR：<https://aclanthology.org/2026.acl-long.910/>；
  - DelTA：<https://arxiv.org/html/2605.21467>；
  - FIPO：<https://arxiv.org/html/2603.19835>；
  - GSPO-token、CISPO、SAPO与CE-GPPO作为detach/clip结构对照。
- 补充核对与当前挂点直接相关的正式工作：
  - Do Not Let Low-Probability Tokens Over-Dominate：<https://arxiv.org/html/2505.12929>；
  - On the Direction of RLVR Updates：<https://arxiv.org/html/2603.22117>；
  - Sparse but Critical：<https://proceedings.iclr.cc/paper_files/paper/2026/hash/c9b7e6175f2bd3c2824f24aa7ac313d6-Abstract-Conference.html>；
  - HICRA：<https://proceedings.iclr.cc/paper_files/paper/2026/hash/79322f3668888f8f7fc99bbd98fbbaed-Abstract-Conference.html>；
  - MINER：<https://aclanthology.org/2026.acl-long.237/>。
- 关键新证据：
  1. GRAIL直接将`sg(w_t)A_i`放入逐token PPO项，范围`[0.5,5]`，并发现wrong-rollout-only优于
     correct-only/all；其position U-shape也显示结构位置需单独分析。
  2. ACL 2026 Covariance-Aware GRPO用Gaussian权重平滑降权极端token并mean-one，说明同一挂点可以
     服务于“抑制更新风险”，不只服务于高signal增权。
  3. OAR/THR/A3PO共同支持signal magnitude、outcome influence和advantage polarity分开解释；强mask或
     大范围通常伴随更直接的outcome语义。
  4. DelTA把更新显式写成token-gradient vector加权和，支持用gradient cosine/projection和
     `w -> Delta log pi`衡量真实干预，而不只看权重上下界；其`[0.8,1.2]`主设置优于更宽范围，且同范围
     随机权重明显更差，说明权重与梯度结构的对应关系比单纯放大区间更重要。
  5. 两篇ICLR工作可从同一低概率现象分别推出降权与增权；差异来自它们是否验证了梯度干扰、方向性与
     critical-token干预，进一步确认不能只凭uncertainty magnitude冻结映射方向。
- 新建：
  `21_PER_ACTION_POLICY_GRADIENT_REWEIGHTING_LITERATURE_20260823.md`；并更新专题索引和根交接路由。
- 本轮未发生：服务器登录、动态状态刷新、训练控制、代码实现、测试、安装、下载或产物修改。
