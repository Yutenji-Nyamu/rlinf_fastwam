# 深圳 current RLinf × RLT / DSRL：真实 smoke 结果与参数判断

日期：2026-08-23  
状态：canonical clean-50、DSRL fresh/resume、RLT Stage 1、RLT Stage 2 fresh/resume及统一postflight均已真实闭环。

## 1. 先看结论

| 阶段 | 真实结果 | 资源判断 | 首个 formal 参数方向 |
|---|---|---|---|
| RLT Stage 1 | current causal AR、frozen π0、2步训练与 `global_step_2` 保存成功 | 2×H100 峰值约 `24.1 GiB/card`；不是显存瓶颈 | 保持历史 `2卡 / MB16 / GB32 / 2k steps`，不要从2步smoke外推墙钟 |
| RLT Stage 2 fresh | 4 train + 4 eval、fresh一轮、8个transition、8 critic + 4 actor updates、保存成功 | 峰值 `16,937/17,233 MiB`，仿真/评估远大于更新耗时 | 2卡不变；formal先回到旧成功 `8 train env × 250 cycles`，再讨论同2卡提高env并发 |
| RLT Stage 2 resume | fresh-process恢复成功；新增8个transition、20 critic + 10 actor updates，累计 `update_step=28` | 峰值 `17,230/17,690 MiB`，主机内存仍轻 | 与fresh共同证明strict resume闭环；formal仍沿2卡历史合同 |
| DSRL fresh/resume | fresh `40 macros/800 updates`；resume新增 `36 macros/720 updates`，累计 `1520 updates`；两次均自然退出0 | 常态约 `29 GiB/card`，初始化/恢复约 `34–36 GiB/card`；主瓶颈仍是UTD20更新 | 保持 `2卡 / 4 env / GB256 / MB64 / UTD20`作可比baseline；吞吐优化优先做 `MB64→128` 等global-batch A/B |

这轮没有证据支持把RLT或DSRL直接扩成4卡。RLT Stage 2资源富余，但瓶颈主要在simulator；DSRL也有显存余量，
但时间主要花在大量小SAC更新。增加卡或env会同时改变world-size、warm-up/replay或每轮更新预算，不是免费提速。

![RLT与DSRL五段真实smoke显存曲线和时间分解](evidence/real-smokes-20260823/01_rlt_dsrl_smoke_resource_and_time.png)

## 2. canonical clean-50：数据合同已经补齐

深圳本次重走了AutoDL旧成功线的official链路，没有现采数据，也没有把XPolicyLab HDF5冒充LeRobot数据：

```text
TianxingChen/RoboTwin2.0 @ 9dc9299c...
  aloha-agilex_clean_50.zip
  -> RoboTwin official raw-to-Aloha converter
  -> official Aloha-to-LeRobot converter
  -> pi0-aloha-clean50-v1
```

| 项 | 已确认值 |
|---|---|
| official source | `TianxingChen/RoboTwin2.0@9dc9299c163db059931898a9f0852098a61155a1` |
| raw ZIP | `298,659,710 bytes` |
| raw SHA-256 | `5554b6b30e37c6ed2f0bbc48079e8ad79d9512e9d4f910a5e71b0d5ad8fbe50e` |
| raw→Aloha source | RoboTwin `c3ddfa8b97d5519efa828b075999bd0006778e5e` |
| Aloha→LeRobot runtime | official RLinf历史LeRobot pin `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |
| canonical path | `/data/chenyiteng/datasets/robotwin2/canonical/pi0-aloha-clean50-v1` |
| final contract | 50 episodes / 7,188 frames / 50 fps / 三相机 / state14 / action14 |
| final size | 约 `1.6 GiB` |

current训练venv中的LeRobot 0.3.3与历史official converter API不兼容，因此转换使用独立、source-locked的converter
venv；训练venv、RLT代码和数据语义均未改变。代理当时访问official GitHub/HF均为HTTP 200，所以直接使用official
Hugging Face，没有切到mirror。

## 3. DSRL：fresh与fresh-process resume都已闭环

### 3.1 实际配置

| 项 | 值 |
|---|---:|
| physical GPU | `6,7` |
| actor world size / train env | `2 / 4` |
| action horizon / macro length / latent | `H50 / N20 / 32D` |
| global / micro batch | `256 / 64` |
| replay capacity | `25,000` global transitions |
| smoke warm-up | `4` global transitions |
| update budget | `UTD20 × globally-new macros` |
| critic | 10-Q；critic-only FP32 shadow |
| RTC / eval | off / off |

运行根：
`/data/chenyiteng/results/rlinf-current-dsrl/smoke-dsrl-pi0-robotwin-2gpu4env-warm4-20260823`

### 3.2 fresh step 1

- 自然退出 `0`，保存 `global_step_1`。
- 收集 `40` 个global macro transitions，执行 `800` 次critic update。
- 总step `429.239 s`：rollout `31.533 s`，training `338.1 s`。
- 有限指标：actor grad `4.342`、critic grad `0.188`、actor loss `-17.667`、alpha `0.926`、critic loss `0.016`。
- 常态显存约 `28.8 GiB/card`；observer峰值GPU6/7均为 `34,395 MiB`，cgroup峰 `53.80 GiB`。
- 主机available最低 `1,947.84 GiB`，没有Ray memory kill、OOM或worker crash。
- `global_step_1` checkpoint约 `32 GiB`，留在`/data`，未下载到Windows。

### 3.3 fresh-process resume step 2

- 从step 1 checkpoint新进程恢复，自然退出 `0`，保存 `global_step_2`。
- 本轮因episode提前结束，新增 `36` 个global macro transitions，执行 `720` 次critic update；累计
  `update_step=1520`。
- 总step `375.282 s`：rollout `34.598 s`，training `307.9 s`。
- rollout return `1.25`、success-once `0.25`；这里只证明真实环境链工作，不作模型效果结论。
- 有限指标：actor grad `4.793`、critic grad `0.106`、actor loss `-11.920`、alpha `0.795`、critic loss `0.0085`。
- 常态显存约 `29.3 GiB/card`；observer峰值GPU6/7均为 `35,843 MiB`，cgroup峰 `48.16 GiB`；
  host available最低 `1,948.91 GiB`。
- resume sidecar恢复后为learned phase，累计ring resident为 `76`；两rank分别 `36/40`，这种不均匀来自真实
  episode长度，不要求每rank机械相等。critic FP32 shadow为每rank `156` 个tensor。
- `global_step_2` checkpoint约 `32 GiB`，同样只留在`/data`。

### 3.4 DSRL参数判断

这轮重新证明了：DSRL不是rollout并发不够，而是更新占主导。fresh与resume中training分别约占完整step的
`78.8%`与`82.1%`；增加env会产生更多macro，并按UTD20同步增加更新数。

首个formal建议保持旧成功科学合同：

- `2×H100 / 4 train env / H50 / N20 / 32D`；
- `GB256 / MB64 / ring25k / warm-up500 / UTD20`；
- stochastic eval仍按旧口径单独锁定；
- 首个吞吐实验只比较同2卡、同GB256下 `MB64→128`，看每次小SAC update能否减少gradient accumulation。

MB128是吞吐候选，不应未经一次受控A/B直接写进formal；当前MB64配置已经完整通过fresh/resume。

## 4. RLT Stage 1：current AR真实训练与artifact保存成功

### 4.1 实际配置

| 项 | 值 |
|---|---:|
| physical GPU / world size | `4,5 / 2` |
| model identity | exact π0 |
| reconstruction | current causal AR |
| trainable boundary | frozen VLA、token-only、`vla_loss=0` |
| micro / global batch | `16 / 32` |
| optimizer steps | `2`，共64次sample presentations |
| data | canonical clean-50 |

运行根：`/data/chenyiteng/results/rlinf-rlt/smoke-stage1-current-ar-2step-20260823`

### 4.2 结果

- 自然退出 `0`，完成 `2/2` steps并保存 `global_step_2`。
- step 1：loss/RLT loss约 `4.39`，VLA loss `0`，grad norm `4.28`，LR `2.5e-5`。
- step 2：loss/RLT loss约 `4.39`，VLA loss `0`，grad norm `4.31`，LR `2.5e-6`。
- 两步driver记录约 `44 s`；单步wall约 `20.7/23.7 s`。第二步包含smoke endpoint保存，因此不能据此把
  2,000步formal线性外推成十几个小时。
- 采样到的显存峰约 `24.105 GiB/card`；主机available约 `1.9 TiB`。
- DCP、`full_weights.pt`与source-locked Stage 1 artifact manifest均成功生成。
- Stage 1 checkpoint约 `21 GiB`；其中 `full_weights.pt=9,556,454,857 bytes`。
- 统一postflight记录cgroup峰 `46.97 GiB`、host available最低 `1,953.20 GiB`。

### 4.3 Stage 1参数判断

首个formal保持 `2卡 / MB16 / GB32 / 2,000 steps`。理由不是显存不足，而是：

1. 这正是旧成功数据与每步样本预算，迁移变量只保留为parallel→current AR；
2. 2步smoke中初始化、Ray启动和终点保存占比过大，不能用于可靠吞吐外推；
3. 直接增batch会同时改变每optimizer step与2k总sample presentations，不是纯硬件利用率调整。

如果要测H100吞吐，应另做不频繁保存的短timing run；不应修改首个可比较formal的科学预算。

## 5. RLT Stage 2：fresh与fresh-process resume均已完成

### 5.1 实际配置

| 项 | fresh与resume共同值 |
|---|---:|
| physical GPU / world size | `4,5 / 2` |
| train / fixed eval env | `4 / 4` |
| primitive limit / executed chunk | `20 / C10` |
| global / micro batch | `512 / 128` |
| warm-up min / post-collect updates | `2 / 8` |
| actor weight warm-up / ramp | `4 / 8` updates |
| max updates per train step | `20` |

运行根：`/data/chenyiteng/results/rlinf-rlt/smoke-stage2-current-ar-fresh1-resume1-20260823`

### 5.2 fresh cycle

- 自然退出 `0`，保存 `global_step_1`。
- 收集 `8` 个global RLT transitions；执行 `8`次critic与 `4`次actor update；pending budget为 `0`。
- 总step `60.067 s`：rollout `20.371 s`，fixed eval `18.28 s`，actor training约 `0.743 s`。
- 有限指标：actor grad `5.288`、critic grad `4.618`、actor loss `2.193`、critic loss `0.020`。
- 20-step smoke的train/eval success均为0；该预算只证明链路，不评价策略效果。
- 统一postflight的显存峰为 GPU4/5 `16,937/17,233 MiB`；cgroup峰 `41.77 GiB`。
- strict sidecar记录 `saved_runner_step=1`、`update_step=8`、每rank `4`个local transitions、global warm-up total `8`。

### 5.3 fresh-process resume

- resume source：fresh `global_step_1`；启动前sidecar已验证world size 2、runner step 1、update step 8、每rank4 transitions。
- 新进程自然退出 `0`，保存 `global_step_2`。
- 本轮新增 `8` 个global transitions，累计 `16`；执行 `20`次critic与 `10`次actor update，sidecar累计
  `update_step=28`、每rank `local_total_transitions_added=8`。
- 总step `62.964 s`：rollout/generate `23.191 s`，fixed eval `36.197 s`，actor training `1.121 s`。
- 统一postflight的显存峰为 GPU4/5 `17,230/17,690 MiB`；cgroup峰 `41.92 GiB`、host available最低
  `1,953.12 GiB`。
- fresh与resume checkpoint各约 `51 MiB`；统一postflight结果为 `RLT_CURRENT_SMOKE_POSTFLIGHT_OK`。

### 5.4 Stage 2参数判断

fresh已经说明两点：

1. 2×H100显存和主存余量很大；
2. 在这条20-step smoke中，仿真与fixed eval远大于小actor/critic update耗时。

因此首个formal推荐回到有AutoDL完整成功依据的 `2卡 / 8 train env / 250 cycles`，保持
`GB512 / MB128 / H50 / C10 / D14`。是否把8 env进一步提高到16，应做同2卡、等episode总预算的并发A/B；
不应先扩4卡，因为会复制冻结π0/Ray worker并改变每rank replay与warm-up合同。

## 6. 本轮出现的唯一前置工程问题

RLT三个launcher最初沿用了其他配置中带空格的Hydra placement key；而current RLT base的真实key是
`actor,env,rollout`。第一次只执行到`--cfg job --resolve`便被Hydra拒绝，尚未启动模型或Ray。处理方式是把三个
launcher统一改成：

```text
cluster.component_placement={actor\,env\,rollout:4-5}
```

第一次空目录被可恢复地改名保存为
`smoke-stage1-current-ar-2step-20260823-attempt1-precompose-failed`，没有删除或覆盖产物。修正后Stage 1和
Stage 2 fresh均通过同一current placement合同。

## 7. 下一步只需做什么

1. 轻量日志、resolved config、resource CSV、artifact/sidecar摘要与资源图已经下载；checkpoint继续留在服务器。
2. 由用户决定：先保持历史formal参数，还是先做RLT Stage 2 env并发/DSRL MB128的单变量吞吐A/B。
3. 在用户确认前不启动RLT/DSRL formal，也不重启GRPO。

smoke后终态：8卡均0 MiB、Ray/GCS进程0，host available约1.9 TiB；`/`、`/home`、`/data`分别余
234 GiB、2.2 TiB、2.9 TiB。两份DSRL checkpoint约64 GiB、RLT Stage 1约21 GiB、Stage 2约102 MiB，
均位于`/data`；没有挤占根分区。Windows主证据包只有约0.35 MiB，下载时C盘仍余36.08 GiB。

## 8. 证据入口

- [专题主入口](00_INDEX_AND_MIGRATION_PLAN.md)
- [代码实现与配置建议](04_IMPLEMENTATION_RESULT_AND_CONFIG_RECOMMENDATION_20260823.md)
- [数据、AutoDL资源与GRPO退出审计](05_PRE_SMOKE_DATA_RESOURCE_AND_GRPO_EXIT_DECISION_20260823.md)
- [AutoDL并行与AutoDL/深圳GRPO内存机制核对](evidence/AUTODL_PARALLELISM_AND_GRPO_MEMORY_NOTE_20260823.md)
- [共同实施流水账](evidence/IMPLEMENTATION_LEDGER.md)
- [RLT实施流水账](evidence/RLT_IMPLEMENTATION_LEDGER_20260823.md)
- [DSRL实施流水账](evidence/DSRL_IMPLEMENTATION_LEDGER_20260823.md)
- [推荐下载：结果文档、资源图、脚本与轻量日志总包](../../exports/shenzhen_current_rlt_dsrl_real_smokes_summary_20260823.zip)
- [仅原始轻量日志、resolved config与资源CSV](../../exports/shenzhen_current_rlt_dsrl_real_smokes_light_20260823.zip)
- [五段真实smoke资源与耗时总图](evidence/real-smokes-20260823/01_rlt_dsrl_smoke_resource_and_time.png)；
  [可重复制图脚本](../../local_scripts/render_shenzhen_rlt_dsrl_smoke_summary_20260823.py)
