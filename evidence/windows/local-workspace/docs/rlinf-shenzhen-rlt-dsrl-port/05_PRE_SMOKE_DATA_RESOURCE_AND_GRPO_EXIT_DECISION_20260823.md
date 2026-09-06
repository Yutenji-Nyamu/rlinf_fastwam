# 深圳 current RLT / DSRL：smoke 前的数据、资源与 GRPO 退出判断

日期：2026-08-23  
状态：本文件保留为smoke前决策快照；其后canonical数据与两算法smoke已完成，终态见
[06号结果文档](06_REAL_SMOKE_RESULT_AND_PARAMETER_DECISION_20260823.md)。GRPO仍未重启。

## 1. 直接结论

1. RLT Stage 1 的数据路线有完整旧成功依据：下载 **RoboTwin 官方 pinned raw clean-50**，再用
   RoboTwin 官方 `raw -> Aloha -> LeRobot` 两级 converter。不是现采，也不是下载第三方预转数据。
2. 深圳现有 clean-50 是 XPolicyLab HDF5；数量虽同为 50 episodes / 7,188 frames，但 schema 与旧
   LeRobot canonical 不同，不能直接填到 `dataset_path`。首条基线推荐重走旧 pinned 约 285 MiB raw
   下载与官方 converter，少写一个新 bridge，且与 AutoDL 最可比。
3. 首条训练配置均保留 AutoDL 的 **2 卡历史合同**：RLT Stage 1 `2卡`；RLT Stage 2 `2卡+8 env`；
   DSRL `2卡+4 env`。原因是可比性、吞吐和 world-size/replay 合同，不是单卡显存装不下。
4. GRPO `exit 255` 已定因：Step 53 rollout 4/4 后，Ray 的用户态 memory monitor 在节点内存越过
   95% 阈值时主动杀 4 个 worker；不是数值崩溃、kernel OOM、SSH timeout 或 RLT/DSRL 代码混用。
5. Mihomo 当前已恢复，且新配额为 500 GiB、剩余约 499.999 GiB、2026-11-23 到期；此前 SSL EOF
   是订阅拉取/所选节点瞬态，不是流量耗尽或仍处在旧到期状态。

## 2. AutoDL 的 clean-50 到底怎么来的

精确链路是：

```text
TianxingChen/RoboTwin2.0 @ 9dc9299c...
  dataset/adjust_bottle/aloha-agilex_clean_50.zip
  -> RoboTwin policy/pi0/scripts/process_data.py
  -> RoboTwin policy/pi0/scripts/convert_aloha_data_to_lerobot_robotwin.py
  -> pi0-aloha-clean50-v1
  -> 50 episodes / 7,188 frames / 50 FPS / 3 cameras / state14 / action14
```

raw ZIP 为 `298,659,710 bytes`。正式 Stage 1 loader 使用两卡，每 rank local batch 16、global batch 32。
完整证据见：

- [下载 pin、目标与下载后校验](../rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md#L1112)
- [官方 converter 来源与首条转换合同](../rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md#L1612)
- [full 50 条转换、manifest 与 loader](../rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md#L2531)
- [Stage 1 demo-first 方法与口径边界](../rlinf-robotwin-pi0-rltoken/02_STAGE1_SMOKE_AND_METHOD_ALIGNMENT_20260729.md#L47)
- [Stage 1 formal 数据与结果](../rlinf-robotwin-pi0-rltoken/03_STAGE1_FORMAL_TRAINING_20260729.md#L18)

为什么这条路线仍适用于 current AR：RLT Stage 1 本来就是从 demonstration batch 学 `z_rl`；current AR
改变的是 decoder 如何按 causal prefix reconstruction，不改变输入仍是 task-specific demonstrations。
所以旧 **Stage 1 权重**不能复用，但旧 **数据语义与转换路线**可以复用。current RLinf 官方文档也明确把
Stage 1 写成“prepare demonstrations -> train Stage 1”，并继续使用 OpenPI data pipeline；RoboTwin/OpenPI
官方路径同样把 Aloha 数据转换为 LeRobot 后训练。

需要区分：RoboTwin/LeRobot 现在也有更新的统一 v3 数据集，但首条迁移不应顺便换数据版本；那会把
“AR port”与“数据升级”混成两个变量。

## 3. 深圳现有 raw 能不能复用

可以作为转换源候选，但不能直接训练。最低要验证：

- 50 条 episode 的逐条长度与总 7,188 frames；
- 14D state/action 的时序语义，尤其旧合同 `state[t]` 对应 `action[t+1]`；
- 三相机映射、instruction/task、FPS 与 episode/frame/timestamp 闭合；
- 转换后的 LeRobot 列、shape、finite 与 OpenPI loader 真实 batch；
- 继续使用深圳 exact-π0 SFT checkpoint 的 norm stats，不用 demonstration 重算后替换。

由于这需要写 XPolicyLab -> LeRobot 窄 bridge，而旧 pinned raw 只有约 285 MiB，**首条基线推荐重新下载
旧 pin 并重跑已有官方 converter**。这不是必须多下载，而是用最少新代码换取最强历史可比性。

## 4. AutoDL 每轮到底做多少，时间花在哪里

### 4.1 RLT Stage 1

- offline，无 simulator rollout；每 optimizer step global 32 samples。
- 2,000 steps 共 64,000 sample presentations，约为 7,188 frames 的 8.90 遍。
- 2×A800、no-shard data parallel、micro16/rank、global32、accumulation1。
- 28m54s；steady p50 `0.777 s/step`。
- GPU 峰 `26,447 MiB/card`；overall util约 `83%/83%`，active mean约 `90.5%/90.6%`。
- matched RSS `38.51 GiB`、anon `39.53 GiB`；约 240 GiB raw cgroup 主要是约 230 GiB file cache。

结论：它已很舒展，不是显存不足造成的串行训练。单张 H100大概率装得下，但会把单 rank
accumulation改为2（或改micro-batch），且大概率更慢；整条 formal 只有约29分钟，首条保持2卡更干净。

### 4.2 RLT Stage 2

- formal 每 cycle 一次 batched rollout：`8 env × 1 episode`。
- 每 episode 最多 `200 primitive / C10 = 20 macros`，所以理论上最多160 macros/cycle；实际
  250 cycles 共34,851 macros，平均 `139.4 macros/cycle`。
- 科学预算为2,000 episodes；`4 env × 500 cycles` 与 `8 env × 250 cycles`等价，旧实验在三轮资源门后
  主动采用8 env，不是内存不足被迫串行。
- update 前普通cycle rollout均值 `119.8s`；进入训练后非eval约 `110.4s rollout + 56.5s update`，
  即约 `65.5%采样 / 33.5%更新`；稳定后约 `74.5% / 24.5%`。
- 250 cycles 用 `10h46m16s`；GPU峰 `19.37/19.51 GiB`，平均util `30.0%/27.7%`；matched RSS
  `87.42 GiB`、Env RSS `62.05 GiB`、anon `82.43 GiB`。

结论：主要瓶颈是 simulator/rollout；首条保持2卡8 env。以后如果提速，做同2卡下的 env 并发 A/B；
不要直接上4卡，因为 `warmup_min_size=10k/rank` 会把全局 warm-up 与 replay 合同一起改掉。

### 4.3 DSRL

- 每 cycle `4 env × 1 episode`，每 episode 最多 `200/N20=10 macros`，因此每轮4--40 macros；
  198 cycles共5,185 macros，平均 `26.19 macros/cycle`。
- 前12 cycles采472个warm-up macros、不更新；之后每轮严格
  `updates = 20 × globally-new macros`，典型40 macros就做800次SAC update。
- learned non-eval周期约 `33.8s rollout + 258.6s SAC`：SAC约 `88.2%`；完整1--197步墙钟中
  rollout `11.83%`、SAC `84.93%`、eval `2.8%`。
- 2×A800常态 `31--32 GiB/card`，DCP峰 `41,455/41,400 MiB`；整体GPU util约
  `24.0%/25.3%`。anon约 `65.9 GiB`；约240 GiB raw cgroup主要仍是约213 GiB file cache。
- train的4 env是真并发；eval的3 waves只是得到 `4 env × 3 = 12` 个固定评估回合，不是显存不足。

结论：瓶颈是大量小SAC更新。单张H100大概率能装下，但micro64下accumulation从2变4，正好让最慢部分
更慢。首条保持2卡4 env；后续最高信息的吞吐A/B是同2卡、global256不变、micro64 -> 128，而不是加env。

## 5. 首条配置决定

| 阶段 | 首条 smoke | 首个 formal | 不先改的原因 |
|---|---|---|---|
| RLT Stage 1 | 2×H100；MB16/rank；GB32；2 steps | 同batch；2k steps | 保持每步与总样本语义；旧2卡已高利用率 |
| RLT Stage 2 | 2×H100；4 env；fresh 1 cycle + resume 1 cycle | 2×H100；8 env；250 cycles | 旧8env formal闭环；主要瓶颈是仿真 |
| DSRL | 2×H100；4 env；fresh step1 + resume step2 | 2×H100；4 env；GB256/MB64；UTD20；约200 steps | 保持world-size、warm-up、ring与更新预算 |

两卡首先是历史合同和吞吐选择，不是“单卡一定OOM”。smoke后若资源明显富余：RLT Stage 2先讨论
8 -> 16 env的**等episode预算**实验；DSRL先讨论MB64 -> 128的**等global batch**实验。

## 6. GRPO 为什么退出，以及是否被 RLT/DSRL 影响

Step 53的4/4 rollout完成后，driver明确记录：

```text
node memory: 1914.32 / 2015.51 GB = 94.9793%
usage: 2056097792000 B > Ray 95% threshold 2055933788160 B
Ray killed 4 workers; one selected EnvWorker used 438.57 GB
Exiting main process due to a failure upon worker execution
```

四个 EnvWorker当时分别约 `528.25 / 438.57 / 430.64 / 406.83 GB`。resource observer在最后3分钟
记录cgroup从 `1920.021 -> 1929.455 GiB`，host available从 `123.152 -> 104.192 GiB`；Ray kill后下一分钟
available恢复到 `1986.929 GiB`。因此 kernel/cgroup OOM=0 与退出并不矛盾：是Ray在内核OOM前主动杀worker。

RLT/DSRL不能解释为“技术栈或组件混淆”：

- 三棵独立clean worktree；没有改GRPO worktree、source、config或run目录；
- 没有启动第二个Ray session、真实模型、RoboTwin或GPU检查；
- 两个较重CPU检查scope分别在kill前约156秒和56秒已deactivated，无残留进程；
- GRPO自己的四个EnvWorker合计已约1.8 TB，并在最后3分钟继续增长9.43 GiB。

所以可排除代码/组件直接干扰。工程经验仍应保留：当available已逼近100 GiB、接近Ray阈值时，连CPU级
检查也应暂停；但这只是避免给极小余量再加负担，不是本次根因归到RLT/DSRL。

## 7. 代理当前状态

管理员安全读取到的新订阅是：`500 GiB total / 0.001 GiB used / 499.999 GiB remaining`，到期
`2026-11-23 20:44:40 CST`。Mihomo服务自8月21日一直运行、restart=0、config mtime未变，说明是供应侧
账户充值/重置，不是服务器本地改配置。20:45复测代理访问GitHub/HF均HTTP200；GitHub直连200、HF直连
仍timeout。此前20:15 subscription pull EOF、部分节点timeout属于瞬态，目前已恢复。

## 8. 下一步边界

- 本轮不重启GRPO。
- RLT数据下载/转换尚未执行；如按本文件建议走旧pin，下载前再报精确目标路径、约285 MiB下载量与
  约1.6 GiB canonical展开量。
- 真实smoke前分别提交resolved YAML、完整命令、GPU、输出目录、样本/更新预算、预计资源、监控项与
  停止条件，再等待批准。
