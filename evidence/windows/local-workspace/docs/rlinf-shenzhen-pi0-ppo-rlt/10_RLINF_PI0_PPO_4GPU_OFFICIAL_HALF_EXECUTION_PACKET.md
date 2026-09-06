# 深圳 RLinf π0 PPO：4×H100 official-half smoke 执行包

> 状态：2026-08-21 用户已授权开始；本文件取代09号8卡草案。只使用物理GPU 4、5、6、7，按官方
> 8×80GB配置保持每卡并发，把总env减半；三段顺序为SFT fixed-64 → PPO one-step → fresh reload fixed-64。

## 1. GPU与source合同

```text
RLinf       7d07a4212ee6858cc333e1d4fab7a37256d1f839
RoboTwin    RLinf_support@0008ae6800df9f75fc8de7098bacb01735fd8fd2
π0 SFT      92684e50dca1a5f75adc8d332046c4cf4fa7a3d0
physical GPU placement  4-7
driver CUDA_VISIBLE_DEVICES  unset
```

RLinf worker会按placement重写自己的`CUDA_VISIBLE_DEVICES`并禁止Ray二次映射。因此driver必须看见裸机8卡，
placement直接写物理`4-7`；不能采用外层mask 4–7、placement 0–3的常规写法。live probe确认driver
GPU count=8、8卡全空闲、物理4–7无compute process、无既有raylet/gcs_server。

## 2. 与official默认的预算diff

| 项目 | official 8卡 | 本次4卡smoke | 说明 |
|---|---:|---:|---|
| train env | 256 | 128 | 每卡仍32 |
| eval env | 128 | 64 | 每卡仍16 |
| actor micro batch | 32 | 32 | 保持official |
| train rollout epochs | 4 | 1 | 只缩smoke时间轴 |
| train steps/epoch | 200 | 50 | 每env只收一个H=50 chunk |
| global batch | 2048 | 128 | 等于本次全部128 records |
| update epoch | 2 | 1 | 精确一次optimizer step |

其余PPO/GAE、LR、clip/value clip、value head、FSDP/offload、π0 H/C=50/50、14D、三相机均保持官方。

```text
records = 128 env × 1 epoch × 50/50 = 128
per-rank records = 128 / 4 = 32 = one micro batch
optimizer steps = 128 / global_batch128 × update_epoch1 = 1
```

## 3. 三段执行

### A. SFT fixed-64

- 输出：`/data/chenyiteng/results/rlinf-shenzhen/ppo/sft-fixed64-4gpu4567-official-half-v1`
- 64 episodes、最多256 policy queries、12,800 primitive slots、0 update；60分钟上限。
- command：`local_scripts/remote_commands/shenzhen_rlinf_r2_run_sft_fixed64_gpu4567_v1.sh`。

### B. PPO one-step + inline fixed-64

- 输出：`/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-oneopt-4gpu128train64eval-v1`
- train：128 queries、6,400 primitive slots、精确1次distributed optimizer step；inline eval再64 episodes、
  256 queries、12,800 slots；120分钟上限。
- checkpoint：
  `robotwin_ppo_openpi/checkpoints/global_step_1/actor/{dcp_checkpoint,model_state_dict/full_weights.pt}`。
- command：`local_scripts/remote_commands/shenzhen_rlinf_r3_run_ppo_oneopt_4gpu128train64eval_v1.sh`。

### C. fresh reload fixed-64

- 输出：`/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-reload-fixed64-4gpu4567-v1`
- 新SSH/Python/Ray生命周期；从B的`full_weights.pt`加载，同一64 seeds；最多256 queries、12,800 slots；
  60分钟上限。
- command：`local_scripts/remote_commands/shenzhen_rlinf_r3_run_reload_fixed64_gpu4567_v1.sh`。

总上限44,800 primitive slots、896 queries、1 optimizer step、16 GPU-hours。三个output启动前均不存在。

## 4. 已resolved证据

服务器目录：
`/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821/resolved-packet-4gpu-official-half-v1/`

| 文件 | SHA256 |
|---|---|
| `sft-fixed64.resolved.yaml` | `b52e45cd42702f8c92937dc7e57c8f19dcbe855830f6cd0017a4d7d2b6618915` |
| `ppo-oneopt.resolved.yaml` | `3a0967a4b7592cf068956febcf2c273c79c50c1cef20d068588dbc5d2d8a931d` |
| `ppo-reload-fixed64.resolved.yaml` | `71da969514116ef528d1c4fc05f24effaf0fd2dcb0d0930f69454edb41e8649a` |
| `budget-and-seeds.json` | `c8791f808f904e67367b6ea2b0e5a0a220d630919fd1634f864acba72d1d3689` |

64个fixed seed的精确列表保存在`budget-and-seeds.json`；A、B inline eval、C完全相同。

## 5. 观察与停止

只观察4–7卡显存/利用率、host RAM、Ray stage/rollout/metrics、checkpoint与视频。CUDA OOM、Ray worker
fatal、SAPIEN/Vulkan fatal、NaN/Inf、checkpoint错误、workers ready后20分钟无stage/metric进展或硬超时，
只停止本轮owned process并保留证据；不预设复杂fallback。若128/64确有资源错误，才用一个独立run退到
64/32，并重新resolve，不覆盖本次路径。
