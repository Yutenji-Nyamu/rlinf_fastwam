# 深圳 RLT / DSRL formal：训练指标、产物与资源现场刷新

更新时间：2026-08-24 10:35 CST

本轮只读：没有停止、重启或修改任何服务器进程、代码、配置和产物。图表所用日志与资源序列内容截止约
10:18 CST；最后的进程与 checkpoint 现场于 10:35 CST 另行刷新。动态状态以后续服务器现场为准。

## 1. 先给结论

| 训练 | 判断 | 现场事实 |
|---|---|---|
| RLT Stage 1 | 正常完成 | current-AR 2,000/2,000，exit 0，`global_step_2000` 完整，约 21 GiB |
| RLT Stage 2 | **不在正常推进** | 仅完整到 Step 24；Step 25 fixed-20 eval 已完成，随后 checkpoint 保存卡住约 8.9 小时 |
| DSRL | **正常推进** | 图表快照完整到 Step 146；10:35 live 已完整到 Step 150/200；已越过 warm-up，优化量 finite，最新 fixed-12 为 11/12 |
| 整机 | 健康 | host available 最低约 1.888 TiB；PSI、OOM、OOM-kill 均为 0；`/data` 约 3.04 TB 可用 |

所以不能笼统说“两条训练都正常”。准确说法是：**DSRL 正常；RLT Stage 1 完成，但 Stage 2 卡在首次
checkpoint，活着的 wrapper 和占用显存不代表训练仍在前进。**

## 2. RLT Stage 1：离线 current-AR 训练正常完成

- 2,000 个 scalar 点连续齐全；`vla_loss` 恒为 0，符合 frozen-VLA / token-only 合同。
- reconstruction loss 从 `4.3596` 降到 `0.5711`；首 100 步均值 `3.2577`，末 100 步均值 `0.5493`。
- grad norm 从约 `4.22` 收到 `0.72`，全程有限。
- 训练 loop 约 12.75 分钟；包含启动与 checkpoint 的 wrapper 总耗时约 14.05 分钟。
- 完整 checkpoint：
  `/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2/stage1/robotwin_adjust_bottle_rlt_stage1_current_ar_clean50_2k_v1/checkpoints/global_step_2000`

![RLT Stage 1 全历史](evidence/formal-live-analysis-20260824/01_rlt_stage1_optimization.png)

## 3. RLT Stage 2：尚未学到 actor，先卡在 checkpoint

### 3.1 截止卡住前实际完成了什么

- 完整 `24/250` cycles，共 `192` 个 train episodes，成功 `27/192 = 14.06%`；末 5 cycle 为 `15.0%`。
- global total transitions 为 `3,599`；决定是否 online 的 **最小 rank replay** 只有 `1,786/10,000`。
- `update_step=0`，actor/critic optimizer update 都是 0；按当前约 74.4 transitions/rank/cycle 的速度，
  原配置约到 cycle 135 才跨 warm-up。
- 首个 fixed-20 eval 已真实跑完，结果 `0/20`。这是 warm-up 期间的 student-only eval；不能把它写成
  “RLT 学习后失败”，因为此时 student 还没有做过一次 Stage 2 optimizer update。

![RLT Stage 2 warm-up 与 checkpoint 卡住](evidence/formal-live-analysis-20260824/02_rlt_stage2_warmup_and_checkpoint_stall.png)

### 3.2 卡在哪里

确定到的边界是：

1. 01:39:19 CST，runner 进入 `RLTACFSDPPolicy.save_checkpoint`。
2. 两个 rank 进入 PyTorch 2.11 `torch.distributed.checkpoint.dcp.save(...)` 的
   Stateful/FSDP optimizer-state 提取路径；最后现场输出位于 `_optim_utils.py:1173`。
3. 截至 10:35 CST，约 8.9 小时没有 driver、TensorBoard 或 checkpoint shard 更新。
4. `global_step_25` 只有 `complete=false` 的 99-byte manifest；DCP 目录没有 `.distcp` shard 和
   `.metadata`，也没有 model、target、replay 或 rank state。因此它明确 **不可恢复**。
5. GPU 4–5 在卡住后恒定约 `17.0 GiB/card`、utilization 全为 0；没有磁盘写入增长、OOM、worker crash
   或系统内存压力。

只读证据还不能把更细的根因唯一锁成某个 FSDP collective。最强相关线索是：同代码、同 torch 的 smoke
在真实做过 8 次 update 后能正常保存；formal 第一次保存发生在 `update_step=0`。这需要一次最小
pre-first-update checkpoint 复现才能定死，不能直接把相关性写成唯一因果。

## 4. DSRL：学习曲线和固定评估都在正常区间

曲线和优化量快照至 Step 146；之后的最终只读现场已完整到 Step 150：

- Step 1–12 是 Gaussian warm-up，训练成功率为 0；Step 13 开始 learned phase。
- 全部 `389/584 = 66.61%`；learned phase 均值 `72.57%`。
- 截至快照，最近 20 步均值 `86.25%`，最近 5 步 `85.0%`。随后 Step 147--150 的 train success
  依次为 `100%, 75%, 50%, 75%`；完整 Step 1--150 合计 `401/600 = 66.83%`，learned phase
  合计 `401/552 = 72.64%`，最近 10 步为 `82.5%`。
- fixed-12 eval：
  `25%, 75%, 75%, 75%, 100%, 75%, 83.33%, 58.33%, 91.67%, 75%, 91.67%`。
  最近 3 次均值 `86.11%`。中间有随机评估波动，但没有策略崩塌。
- 快照 replay resident `4,280`，累计 optimizer update 预算约 `76,000`；10:35 live Step 150
  为 `4,391` transitions / 约 `78,220` 次累计 update 预算。

![DSRL 成功率、replay 与耗时](evidence/formal-live-analysis-20260824/03_dsrl_success_replay_and_timing.png)

## 5. DSRL 优化指标

Step 146 快照：

| 指标 | 最新值 | 解释 |
|---|---:|---|
| actor loss | 4.668 | 最近 10 步均值 4.666，稳定 |
| critic loss | 1.167 | 最近 10 步均值 1.146，稳定 |
| alpha | 0.0034 | 连续自适应，没有跳变 |
| actor grad norm | 1.637 | 早期下降后稳定在约 1.5–1.7 |
| critic grad norm | 32.748 | 持续抬升后最近约 30–35；当前有限，但值得继续看至结束 |
| actor entropy | 约 -16 | 与 Q 值均保持有限、连续 |

日志没有 traceback、OOM、WorkerCrashed、RayTaskError、nonfinite、NCCL 错误。原始 grep 的 `ERROR`
命中只是 oneDNN 提示文字里的英文 `errors`，不是错误事件。

![DSRL 优化指标](evidence/formal-live-analysis-20260824/04_dsrl_optimization_metrics.png)

## 6. 时间和资源

### 6.1 DSRL 时间

- 最近 10 步平均约 `232.6 s/step`。
- 其中 actor/SAC training 约 `190.7 s`，rollout 约 `27.0 s`；其余为同步与环境 bookkeeping。
- 这印证旧 AutoDL 经验：DSRL 的主耗时是 UTD20 的大量小 SAC 更新，不是 simulator 采样。
- 以 live Step 150 和最近速度估算，剩 50 步约需 3.2 小时；自然完成后再给最终吞吐，不把线性外推当终态。

### 6.2 GPU 与主机

- RLT Stage 2 正常采样段 GPU 4/5 峰值约 `19.36/19.65 GiB`，util p95 都约 99%；卡住后显存固定
  `17.416/17.390 GiB`，util 全为 0。
- DSRL GPU 6/7 去掉约 93 秒启动装载短峰后，稳态 median 约 `33.30 GiB/card`，p95
  `33.59/33.62 GiB`；checkpoint 窗口约 `36.6 GiB`。启动装载短峰约 `61.75 GiB`，不是随 step 增长。
- host available 全程最低约 `1.888 TiB`、快照最新约 `1.899 TiB`，没有持续下降；PSI 和 OOM 计数为 0。

![并发 formal 资源时间线](evidence/formal-live-analysis-20260824/05_concurrent_formal_resource_timeline.png)

## 7. 主要产物

### RLT

- Stage 1：完整 `global_step_2000`，约 21 GiB，可用。
- Stage 2：`global_step_25` 只有 incomplete manifest，不可用；没有任何有效 Stage 2 checkpoint。
- Stage 2 的 Steps 1–24 driver/TensorBoard/resource/resolved 均有效；没有视频，因为 formal 显式关闭保存。

### DSRL

- `global_step_65`、`global_step_130` 各约 `33.59 GB`，均为可恢复 checkpoint。
- sidecar 的 `update_step` 分别为 `34,020` 和 `67,900`；learned phase、ring cursor/state、
  critic-only FP32 shadow 都齐全。
- FP32 shadow 只有 156 tensors / 2,672,362 parameters，确实没有复制冻结 3.5B backbone。
- 没有视频；正式训练配置关闭了 train/eval video。

## 8. 下一步建议（本轮没有执行）

1. 让 DSRL 继续自然训练，不做干预。
2. RLT 当前进程不会自行产生新 step，继续占着 GPU 4–5 没有训练价值；但停止属于进程控制，本轮没有擅自做。
3. 若获授权，精确停止 RLT v3 owned PGID 和它自己的 namespace actors，**不能 `ray stop`**，否则会伤到
   同一 persistent Ray 上正常运行的 DSRL。
4. 保留现有不完整目录和日志后，只做一次两卡短复现：warm-up 仍为 10,000、1 cycle、
   `save_interval=1`，验证 pre-first-update save。
5. 若复现，最小配置绕法是保持 eval 每 25 cycle，但把首个 formal save 移到 warm-up 后，例如 cycle 150；
   不降低 warm-up，因为降低它会改变算法预算。代价是前 150 cycles 没有 Stage 2 恢复点，需要你确认。

## 9. 复现材料

- 原始轻量文件、派生 CSV、summary JSON 与五张 PNG：
  `evidence/formal-live-analysis-20260824/`
- 绘图与解析脚本：
  `local_scripts/render_shenzhen_rlt_dsrl_formal_live_20260824.py`
- 本轮逐操作账：
  `evidence/FORMAL_LIVE_ANALYSIS_LEDGER_20260824.md`
