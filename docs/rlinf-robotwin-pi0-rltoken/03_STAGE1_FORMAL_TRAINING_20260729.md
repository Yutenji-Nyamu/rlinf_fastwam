# RLT Stage 1 formal training（2026-07-29）

> 任务：RoboTwin `adjust_bottle`，full clean-50，低预算 RLT 移植
> 状态边界：2026-07-29 endpoint exit0，artifact acceptance 全门通过
> 机器证据：[`evidence/stage1_formal_20260729/`](evidence/stage1_formal_20260729/)

## 1. 结果先行

Stage 1 已完整训练 2,000 optimizer steps 并以 exit0 保存唯一 endpoint。总 wall time
`28m54s`；前100步 loss 均值 `3.9023`，后100步 `0.5551`，下降 `85.8%`；
`vla_loss=0` 全程成立。两卡峰值均 `26,447MiB`，matched rank RSS 峰值约
`38.51GiB`，无 OOM/CUDA/NCCL/rank-death。

新进程 artifact acceptance 又通过 strict reload、非RLT π0 tensor bitwise不变、
fixed prefix 一致和 true/shuffled/zero bottleneck 对照。因此现在可以称为
“可供 Stage 2 绑定的 task-specific RL-token artifact”；仍不能称为控制性能已提升。

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
endpoint:
  /root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/
  robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
runtime/evidence:
  /root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1
artifact acceptance:
  .../artifact_acceptance_v2
```

endpoint 总计 `20.56GiB`：

| 文件 | 大小 | 用途 |
|---|---:|---|
| `actor/model_state_dict/full_weights.pt` | 8.895GiB | 聚合加载与 Stage 2 frozen feature |
| `actor/dcp_checkpoint/__0_0.distcp` | 5.833GiB | rank0 恢复 shard |
| `actor/dcp_checkpoint/__1_0.distcp` | 5.832GiB | rank1 恢复 shard |
| `actor/dcp_checkpoint/.metadata` | 约1KiB | DCP metadata |

没有中间 checkpoint 堆积。高信息量日志/配置/指标/资源/manifest/验收源码已打包到：

```text
C:\Users\86136\Documents\rl\exports\
  rlt_stage1_formal_high_info_20260729_v2.zip
SHA-256:
  9d9e2c38789897479a27cc04ed15034a9d65175284c837f3c1c6f54ca0c2daa8
```

包大小约616KiB，不复制20.56GiB checkpoint。

## 5. 训练指标、资源与 artifact 验收

### 5.1 训练曲线

| 指标 | 结果 |
|---|---:|
| optimizer steps | 2,000 |
| step1 / step2000 loss | 5.203 / 0.579 |
| first100 / last100 mean | 3.9023 / 0.5551 |
| minimum | 0.515 at step1802 |
| grad norm mean / p95 / max | 0.995 / 2.12 / 4.22 |
| LR | 2.5e-7 → 2.5e-5 at step100 → 2.5e-6 |
| steady step p50 | 0.777s |
| sample presentations | 64,000，约8.90个 dataset-frame equivalents |

TensorBoard 有7类、每类完整2,000点。console step为1–2000，TensorBoard为0–1999，
只是编号习惯不同。

### 5.2 内存到底看哪个数

| 指标 | 峰值/最低值 | 解释 |
|---|---:|---|
| GPU | 26,447MiB/card | 两 rank 真实训练峰值 |
| matched rank RSS | 38.51GiB | 只加匹配训练 rank 的进程常驻集 |
| cgroup anonymous | 39.53GiB | 最接近容器不可回收训练工作集 |
| cgroup file cache | 229.94GiB | checkpoint/model page cache，大多可回收 |
| raw cgroup current | 约240GiB | anon+file cache等，不能当成私有训练内存 |
| host available minimum | 928.83GiB | 整机没有 RAM 压力 |

因此“总内存曲线”同时保留 cgroup anon/file/current 和 host available；RSS 只是相关进程视角，
raw cgroup 又包含大量可回收 cache，两者不能相互替代。1,455点曲线已放入下载包和当前
对话可视化。

### 5.3 artifact acceptance

fixed real batch=4：

| 检查 | 结果 |
|---|---:|
| fresh seed-0 proxy loss | 5.1977 |
| endpoint true-`z_rl` loss | 0.5338 |
| shuffled-`z_rl` loss | 1.7118 |
| zero-`z_rl` loss | 2.1027 |
| endpoint/fresh | 0.1027 |
| true/shuffled | 0.3118 |
| true/zero | 0.2539 |
| non-RLT changed tensors | 0 |
| RLT changed tensors | 54/62 |

所有 hard gates 为 true。manifest：

```text
ID:
  robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
manifest SHA:
  6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
full-weights SHA:
  7dddc268733b978bf382cda77257371cf9de4155f60ec3094cc8ffcfd6d74bd0
```

正式 endpoint 固定，不按未来 Stage 2 成绩回头挑选。

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
