# 深圳 current RLinf × π0 × DVAC-GRPO：短迁移计划

最后更新：2026-08-24  
状态：**global-z current port 已实现、测试、commit/push；真实2-step smoke 正在 physical GPU0--3 运行。**

## 0. 先给判断

- 旧实现的 Git 历史很干净：从 old RLinf `6d0db56...` 线性增加 telemetry、global-z
  权重、R-only 权重和配置，没有 merge，也没有整文件替换。
- 但它不是“只搬一份 YAML”：最终累计 `16 files, +3759/-18`；其中约 1,994 行生产代码、
  1,345 行配置、420 行测试。较大部分是 telemetry writer、provenance、对齐元数据和重复完整 YAML。
- 真正改变训练数学的核心较小：生成 $V_L$，用 recent-5 得到权重 $w$，然后在 actor 中用
  straight-through 形式只缩放每个 future-$h$ 的 log-prob 梯度。
- 深圳 current 版本已经有验证过的 π0 DVAC endpoint/telemetry；本次只需接上训练权重，不能把旧
  worker 整文件搬过来。
- 用户已选择首格复刻旧成功路线：只实现 `global_zscore`、权重 `[0,2]`；R-only/S/I 暂不混入本次 port。

## 1. 精确来源锁

旧分支是同一条线：

```text
old RLinf 6d0db56
  -> 61996e15  opt-in π0 DVAC evaluation telemetry
  -> 145fa810  global-z DVAC-GRPO
  -> afdaa2e2  R-only + [0,2] 两类配置
```

| 来源 | HEAD / base | 已核实用途 |
|---|---|---|
| 旧 telemetry | `6d0db56... -> 61996e15...` | endpoint chain、NPZ/CSV、query/episode join；默认关闭 |
| 旧 train weighting | `61996e15... -> 145fa810...` | global-z recent-5、ST 梯度挂点、两份训练配置 |
| 旧 residual | `145fa810... -> afdaa2e2...` | per-$h$ median/MAD residual；后三个 commit 仅增加配置 |
| 深圳 base | `7d07a421...` | current RLinf 唯一基线 |
| 深圳 current GRPO | `554c6dc8...` | current π0 RoboTwin GRPO recipe；只新增一份 YAML |
| 深圳 current DVAC | `f7cf0f60... -> 800baf80...` | current π0 endpoint/telemetry + real-query parity 工具，已真实 fixed-64 |

GitHub 精确 compare：

- [旧 telemetry](https://github.com/Yutenji-Nyamu/rlinf_fastwam/compare/6d0db56bf26f972cd27fa29535f5eb939e80e5bf...61996e15cc7f5a32bd6012b61b20893d94636c82)
- [旧 global-z 训练增量](https://github.com/Yutenji-Nyamu/rlinf_fastwam/compare/61996e15cc7f5a32bd6012b61b20893d94636c82...145fa810f1d8baee23012922b81e496661d61cf5)
- [旧 R-only 增量](https://github.com/Yutenji-Nyamu/rlinf_fastwam/compare/145fa810f1d8baee23012922b81e496661d61cf5...afdaa2e2aa59aa16128e89f47eb4aaf7a64badd8)
- [深圳 current telemetry](https://github.com/Yutenji-Nyamu/rlinf_fastwam/compare/7d07a4212ee6858cc333e1d4fab7a37256d1f839...800baf80d6eab64169cf0e691eb04a681a093ee9)
- [深圳 current GRPO recipe](https://github.com/Yutenji-Nyamu/rlinf_fastwam/compare/7d07a4212ee6858cc333e1d4fab7a37256d1f839...554c6dc8d586162d9444c01fa88308ed4f5203d0)

## 2. 旧实现到底改了什么

| 层 | 精确增量 | 是否改变训练 |
|---|---|---|
| endpoint telemetry | 6 files，`+1099/-9` | 不改变；模型热路径约35行，只旁路记录 $x$ chain / $z$ endpoint |
| global-z train | 8 files，`+1414/-9` | 改变；核心是新 `dvac_train_weighting.py` 和 actor ST 挂点 |
| R-only extension | 7 files，`+1271/-25`，其中4份 YAML 占826行 | 改变；feature commit本身4 files、`+650/-25` |

训练语义是：

1. rollout 在 π0 的 $M=4$ 去噪过程中得到 `z_endpoint [B,M,H,D]`；
2. 立即计算 $V_{L=3}[B,H]$，随后丢弃大 chain，训练 trajectory 只携带小的 `[B,H]` tensor；
3. 用最近5个**已经完成**的 outer step 建 baseline，当前 step 不进入自己的 baseline；
4. actor 在 shuffle 和所有 update epoch 之前冻结本 step 的 $w(q,h)$；
5. 原 advantage、GRPO group normalization、ratio、clip 和 global grad clip 全部不改；只用
   `detach(logp) + w * (logp - detach(logp))` 改 backward。

两个旧 mode：

- `global_zscore`：$y=\log(V_{L3}+10^{-12})$，recent-5 global mean/std；旧配置有 `[0.8,1.2]`
  和 `[0,2]`。
- `per_h_robust_residual`：每个 $h$ 单独用 median/MAD 得到 $R(q,h)$，再映射到 `[0.5,1.2]`
  或 `[0,2]`。

## 3. current 版本怎么接：最小实现面

推荐新分支：`codex/sz-current-pi0-dvac-grpo`，从 `554c6dc8...` 建独立 worktree。

| 旧落点 | current 落点 / 处理 | 原因 |
|---|---|---|
| old `openpi_action_model.py` | 直接合入 current `f7cf0f60...`；不重写 | current 版本已保留 RTC、`actions/model_actions` 和真实 endpoint |
| old `dvac_telemetry.py` | 直接复用 current 版本 | fixed-64 已验证；它是观测层，不是训练 loss |
| old `dvac_train_weighting.py` | 新增到 current，数学核心基本原样；writer 首版收窄为 step summary | 避免把旧 control trace/重型落盘混进训练 |
| old `fsdp_actor_worker.py` | 改 current `embodied_fsdp_actor_worker.py::EmbodiedFSDPActor` | current 已把 embodied actor 从 generic actor 拆出 |
| old rollout dict hook | 窄改 current `huggingface_worker.py::MultiStepRolloutWorker` | train 时请求现成 endpoint，算完 $V_L$ 后只放入 `PolicyOutput.forward_inputs` |
| old env query metadata | 首版不搬 | current Builder 会自动传递 `forward_inputs`；数学不需要旧 control-trace 元数据 |
| old schema/Builder 修改 | 不改 | current `PolicyOutput -> ChunkStepResult -> Builder -> Trajectory` 已能递归携带和 shuffle tensor |
| old `robotwin_env.py` control trace | 不搬 | 只服务可视化 provenance，不决定 DVAC 或 GRPO 梯度 |
| old full YAML | 不整份复制 | 从 current GRPO recipe 只增加 DVAC stanza、run/path 和 smoke 预算 |

明确不改：`losses.py`、GRPO advantage、ratio/clip、current schema、Builder、RoboTwin action decode、
RTC、π0 模型结构。

## 4. 真实适配点与坑

### 4.1 两个开关必须分开

- `rollout.dvac_telemetry`：standalone eval 的完整落盘。
- `algorithm.dvac_gradient_weighting`：训练时的 compact $V_L$ 与权重。

不能为了训练删除 eval guard；正式训练也不应默认写全量 NPZ/视频。

### 4.2 current typed data path

旧分支使用旧 dict/trajectory 路线；current 是
`PolicyOutput -> ChunkStepResult -> EmbodiedTrajectoryBuilder -> Trajectory`。权重必须经
`forward_inputs` 走完整链并随 current shuffle 一起重排，不能另建一份按原顺序索引的旁路数组。

### 4.3 4 actor ranks

旧正式实验是2 ranks；深圳 current recipe 可以是4 ranks。global-z sufficient statistics 要全局 reduce，
R-only 的 `[query,H]` history 要全局 gather，不能硬编码2。当前等长 rank 可直接复用；没有观察到不等长，
首版不预设 padding/fallback，只补4-rank focused test。

### 4.4 strict resume 是旧实现的真实缺口

旧两个 stats 类只有 `state_dict()` 用于产物，没有 `load_state_dict()` 或 checkpoint hook；恢复时 recent-5
会静默从头 warm-up。深圳 GRPO 已真实发生过中途退出，因此这不是臆想风险。

推荐在 formal 前增加一个很小的 DVAC sidecar：保存 recent-5 原始状态、mode、selected-L 和 world size，
在 generic model/optimizer load 后恢复；不把 history 塞进 FSDP optimizer state。fresh run 数学不变。

### 4.5 内存问题与 DVAC 分开处理

current Git recipe 是 `32 env × 8 rollout epochs = 256 trajectories/step、B512/mb32`，与旧 AutoDL
`16 × 16 = 256、B512/mb32` 的全局预算接近。此前深圳 PPO-matched formal 使用的是另一份
`128 × 4、B2048` resolved 配置，并因常驻 EnvWorker 主存触到 Ray 95% 阈值结束。

只保存 $V_L$ 和 $w$ 的 tensor 是 MiB 以下量级，不能把原有百 GiB EnvWorker 增长归因给 DVAC；formal
并发参数和 env offload 仍在 smoke 后单独讨论，不混入算法 port。

## 5. 已完成的实现与最小验收

1. 从 `554c6dc8...` 建独立 branch/worktree并合入 current telemetry；没有搬旧 worker/env/schema。
2. 当前生产增量为5 files、`+667/-2`；只实现 global-z、current rollout hook、actor ST、step tensor与
   exact-resume sidecar。
3. 10项 focused tests、Ruff、diff check通过；baseline/DVAC resolved config 除 `mode` 外完全一致。
4. commit `66c863bc5a45e90cb5161b30af54355b1104c810` 已推到
   `personal/codex/sz-current-pi0-dvac-grpo`，服务器工作树干净。
5. 真实2-step smoke 已自然 `exit 0`：Step1只建history；Step2首次应用非均匀权重，ESS=`0.811`，
   fixed64=`49/64`，完整 `global_step_2` 与4份resume sidecar均已保存。详见专题25号结果文档。

## 6. smoke 后只讨论一个参数问题

算法路线和恢复语义已经由用户锁定：首格 global-z `[0,2]`，并要求 exact resume。根据 smoke 与旧
深圳 GRPO 的主存证据，推荐 formal 使用 current/AutoDL DVAC 同预算的 `32x8/B512`，不继续高负载
验证用的 `128x4/B2048`；不在本次 port 中引入 R-only/S/I。启动 formal 前仍需展示最终 resolved packet。

规划期证据见
[`evidence/21_CURRENT_RLINF_DVAC_GRPO_PLANNING_LEDGER_20260824.md`](evidence/21_CURRENT_RLINF_DVAC_GRPO_PLANNING_LEDGER_20260824.md)；
实施与 smoke 逐命令见
[`evidence/22_CURRENT_RLINF_DVAC_GRPO_IMPLEMENTATION_AND_SMOKE_LEDGER_20260824.md`](evidence/22_CURRENT_RLINF_DVAC_GRPO_IMPLEMENTATION_AND_SMOKE_LEDGER_20260824.md)。
