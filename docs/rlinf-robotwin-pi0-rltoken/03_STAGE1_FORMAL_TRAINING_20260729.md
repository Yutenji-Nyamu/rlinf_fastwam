# RLT Stage 1 formal training（2026-07-29）

> 任务：RoboTwin `adjust_bottle`，full clean-50，低预算 RLT 移植
> 状态边界：2026-07-29 19:38:41（Asia/Shanghai）已启动并通过早期健康门；不声称 endpoint 完成
> 机器证据：[`evidence/stage1_formal_20260729/`](evidence/stage1_formal_20260729/)

## 1. 结果先行

Stage 1 已从 clean、已推送的
`codex/rlt-pi0-robotwin@4ac48d54c63b3a83d99f551fb54f738297525acf` 启动。固定早期
快照为 step 172/2000，连续训练约 0.777s/step；两卡各 26,447MiB，matched RSS 峰值约
38.5GiB；`loss == rlt_loss`、`vla_loss=0`、grad/LR finite，OOM/CUDA/NCCL/Traceback/
rank-death 信号全为 0。

这表示数据、两 rank、冻结 π0、RLT reconstruction、optimizer 和 resource monitor 已进入
稳定执行，不表示 2k endpoint 已保存或表征质量验收完成。按用户要求，本轮不再持续轮询；
下次询问时重新刷新服务器进程、step、日志、资源和 checkpoint。

## 2. 数据与 manifest

```text
source:
  TianxingChen/RoboTwin2.0
  revision 9dc9299c163db059931898a9f0852098a61155a1
  dataset/adjust_bottle/aloha-agilex_clean_50.zip
canonical:
  /root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-v1
manifest:
  /root/autodl-tmp/datasets/robotwin2/manifests/pi0-aloha-clean50-v1.json
manifest SHA-256:
  12ce2ed68632e2b18cf96f52b717edec00bcebb6cc0a446f83da1670d81ef86c
```

合同结果：

- 50 个有效 episode，合计 7,188 frame，50 FPS；
- canonical state/action 为 14D，OpenPI loader 后 pad 到 32D；
- 三相机完整；正式 global32 probe 的两个 rank 均为 local16；
- 一级动作合同全量满足 `qpos=raw_t`、`action=raw_{t+1}`，最大绝对误差均为 0；
- Stage 1 继续使用 base checkpoint 的 stats，SHA-256
  `649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a`。

## 3. 冻结配置

完整 resolved config 见
[`formal_resolved.yaml`](evidence/stage1_formal_20260729/formal_resolved.yaml)，SHA-256
`5aa824fc9ac5cc361dace2b1162b2ef1bdf52adab3c775b8cd2e1ae468dfd67e`。

| 类别 | 正式值 |
|---|---|
| GPU/rank | GPU 0/1，2 ranks |
| batch | micro16/rank，global32，accumulation1 |
| FSDP | `no_shard`，`use_orig_params=true`，AMP off，gradient checkpointing off |
| trainable | 仅 `rlt_module.*`；π0 frozen；`rlt_train_vla=false`、`rlt_alpha=0` |
| bottleneck | image-only `[B,768,2048] → [B,1,2048]`，2+2 layers、8 heads、ratio4 |
| optimizer | AdamW `2.5e-5`，β `.9/.95`，eps `1e-8`，wd `1e-10`，clip1 |
| schedule | warmup100 + cosine，`min_lr_rate=.1` |
| endpoint | 2,000 steps；无 val；只在 step2000 save |

不提高 batch：S1-B 已证明 16/32 容量可行；增 batch 会改变固定 2k 下的样本预算和优化语义，
不是单纯利用多余显存。当前两卡已并行达到高利用率，无需为正式启动增加吞吐 sweep。

## 4. 命令与产物

精确命令见
[`exact_command.txt`](evidence/stage1_formal_20260729/exact_command.txt)。核心运行与目录：

```text
driver PID:
  650254
resource monitor PID:
  650255
run:
  /root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1
runtime/evidence:
  /root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/runtime
driver log:
  .../runtime/driver.log
resource CSV:
  .../runtime/resources.csv
expected checkpoint:
  /root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/
  robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
```

只在 endpoint 保存约 21GiB，避免形成中间 checkpoint 堆积；代价是 step2000 前机器中断要
fresh restart。18 小时 timeout 只是故障上限，不是 ETA；19:38 快照按最近 20 步中位数估算
剩余训练约 23.7 分钟，尚未计 endpoint save。

## 5. 早期健康证据与后续验收

[`early_health.json`](evidence/stage1_formal_20260729/early_health.json) 记录：

| step | LR | loss/rlt_loss | grad norm | step 时间 |
|---:|---:|---:|---:|---:|
| 1 | 2.5e-7 | 5.20 | 2.30 | 20.1s（冷首步） |
| 10 | 2.5e-6 | 5.19 | 2.34 | 0.780s |
| 20 | 5.0e-6 | 5.16 | 2.30 | 0.783s |
| 50 | 1.25e-5 | 4.51 | 1.94 | 0.777s |
| 172 | 2.49e-5 | 1.05 | 1.03 | 0.774s |

所有记录均 `loss=rlt_loss`、`vla_loss=0`；LR 与 100-step warm-up 一致。早期 loss 下降是
有用但不充分的直接证据。endpoint 后仍要做：

1. checkpoint 文件完整性与新进程 reload；
2. fixed-prefix step0→2k reconstruction loss 对比；
3. true-`z_rl` 对 shuffled/zero 的优越性；
4. frozen π0 数值 delta=0；
5. Stage 1 artifact manifest，记录 checkpoint、config、dataset、stats 和 Git hash。

正式 endpoint 不按未来 Stage 2 成绩回头挑选。

## 6. 本轮定向磁盘清理

启动前按用户授权删除：

- 12 个旧 RLinf DCP：两个 smoke、PPO step10、GRPO step10–90；
- 两个 `Motus_old/logs_single_*` 下 51 个 OPD/GKD 实验 `.pt`。

合计回收 `190,963,833,882 B`，约 177.85GiB；PPO step20、GRPO step100 以及轻量
log/config/metrics 保留，Motus 官方/base checkpoint 完全不在删除根内。删除清单在：

```text
/root/autodl-tmp/experiment_exports/rlt_pre_stage1_cleanup_20260729
```

被删权重不可恢复，除非服务器另有外部快照。
