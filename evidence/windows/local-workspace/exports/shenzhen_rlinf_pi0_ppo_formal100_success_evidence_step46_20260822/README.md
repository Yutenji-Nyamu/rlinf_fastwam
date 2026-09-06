# 深圳 H100：RLinf π0 PPO formal-100 成功运行证据（Step 46 快照）

## 1. 证据结论与边界

这是一个轻量、高信息量的本地证据包，证明深圳服务器上的 current RLinf × RoboTwin × π0 PPO formal-100
在 **2026-08-22 20:58 CST** 的只读快照中已完整运行到 Global Step 46，并保持正常训练链路：当时 driver
alive、fatal 扫描为 0、cgroup OOM/oom_kill 为 0；Step 1–46 的主要优化标量均为有限值；fixed-64 在
Step 10/20/30/40 均完成；Step 10/20/30/40 checkpoint 均已落盘。

这个包不是“100 步已完成”的终态包。最新完整训练指标是 Step 46；Step 46 不是 save/eval 点，最新可恢复且
带 fixed-64 eval 的自然产物是 Step 40。服务器状态可能在该快照之后继续变化，本包不把历史快照冒充当前现场。

刻意不包含 checkpoint 权重、DCP shard、大视频、凭据、代理信息或重复的旧 Step 35/40 原始文件。包内
TensorBoard event、metrics log 和 PNG 都是截至 Step 46 的最新本地副本。

## 2. 运行身份、resolved config 与预算

- RLinf source：`7d07a4212ee6858cc333e1d4fab7a37256d1f839`
- RoboTwin source：`0008ae6800df9f75fc8de7098bacb01735fd8fd2`
- server run：`/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1`
- physical GPUs：4–7；train/eval env：128/64
- train：4 rollout epochs × 200 primitive steps，`H=C=50`，每个 outer step 2,048 records
- actor：global batch 2,048，micro batch 32，update epoch 2，即每个 outer step 2 次 distributed update
- runner：最多 100 outer steps；每 10 步 fixed-64 eval 与 checkpoint
- server-side `resolved.yaml` SHA-256：
  `48b4be79af300512d757ce47219ea2b27445155dbd6438be300459756b266926`

完整 `resolved.yaml` 没有复制到 Windows，因此不在本包中伪造一个替代品。其服务器路径、SHA-256、相对
official YAML 的参数差异和精确预算由
`docs/rlinf-shenzhen-pi0-ppo-rlt/11_FORMAL_CONFIG_AND_MEMORY_ANALYSIS.md` 记录；完整 Hydra overrides、启动前
source/output/GPU gates、resolved 保存和后台启动方式由
`local_scripts/remote_commands/shenzhen_rlinf_launch_ppo_formal100_4gpu128train64eval_v1.sh` 原样记录。

## 3. Step 46 关键结果与产物 inventory

- Step 46 train success：89.84%；Step 42–46 均值 90.82%；最近 10 步均值 91.68%。
- fixed-64 success：Step 10/20/30/40 为 `58/62/58/62`，即 `90.63/96.88/90.63/96.88%`。
- Step 46：KL 0.020、clip fraction 0.063、pre-clip grad norm 30.992、critic explained variance 0.447。
- Step 1–46 中位数：whole step 1,563.05 s、rollout 1,535.15 s、actor training 22.94 s；瓶颈为 simulator rollout。
- checkpoint：`global_step_10/20/30/40`，每个 18,475,261,914 B（17.21 GiB），其中
  `full_weights.pt` 为 8,067,778,207 B（7.514 GiB）；`global_step_50` 当时尚不存在。
- run 总量：74,093,166,525 B（69.01 GiB）；train/eval MP4 数量为 743/16，约 180.3/2.06 MiB。
- `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/formal100_live_step21_20260822/artifacts_inventory.md` 是较早的
  Step 21 只读结构快照，只用于证明单个
  checkpoint 的 DCP metadata、4 shards 与 full weights 文件合同；当前 checkpoint 数量与 Step 46 结论以
  本 README、运行 ledger 和曲线文档为准。

资源风险也保留在证据中：20:58 快照 cgroup 约 1.779 TiB，四个 EnvWorker RSS 合计约 1.679 TiB，整机可用
内存约 200.88 GiB；GPU 4–7 的中段瞬时显存约 69.3/59.6/61.1/65.1 GiB。训练数值正常，但主存余量很窄。

## 4. 推荐阅读顺序

1. `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/ppo_formal100_success_eval_step1_46_20260822.png`：训练 success
   与四个真实 fixed-64 eval 点。
2. `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/ppo_formal100_optimization_step1_46_20260822.png`：KL、clip、
   grad norm、critic 指标。
3. `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/ppo_formal100_timing_step1_46_20260822.png`：step、rollout、
   actor training 时间分解。
4. `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/ppo_formal100_resource_discrete_through_step46_20260822.png`：
   有限离散现场快照；不是伪造的逐 step 资源曲线。
5. `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/ppo_formal100_console_scalars_step1_46_20260822.csv`：最轻量的
   逐步表格；需要原始上下文时再读 metrics/event。
6. `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/06_RLINF_PI0_PPO_4GPU_RUN_LEDGER.md` 与
   `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/09_PPO_CURVES_LIVE_20260822.md`：命令、结果、问题、边界。

## 5. 文件清单与原件 SHA-256

README 之外的 13 个文件均直接复制自本地现有证据；下列哈希是复制前原件 SHA-256，打包后还会再次核对。

| 包内路径 | bytes | SHA-256 |
|---|---:|---|
| `docs/rlinf-shenzhen-pi0-ppo-rlt/11_FORMAL_CONFIG_AND_MEMORY_ANALYSIS.md` | 17,743 | `f991f175aedb4799fb6db1a601a84bd8150875f2ad3d2478932d0940db4f56ae` |
| `local_scripts/remote_commands/shenzhen_rlinf_launch_ppo_formal100_4gpu128train64eval_v1.sh` | 3,712 | `369f202ec6a467ba6e7f30840a97adfa0662a2217e7264409b28880b960a97ee` |
| `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/06_RLINF_PI0_PPO_4GPU_RUN_LEDGER.md` | 28,146 | `1c5dcfa7cc170090dd5ec2d135d39de398d06576f277f17efd2b132a7dc9df48` |
| `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/09_PPO_CURVES_LIVE_20260822.md` | 14,589 | `cef59b1c1f3b61cc60d0368d14f6f9ac03454990aefd54e1d76f9476f159ff84` |
| `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/formal100_live_step21_20260822/artifacts_inventory.md` | 6,185 | `d19ca3a5e91803d1cc2771acc8d337f0e3951c3e84137b59f21202ac91fe961b` |
| `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/ppo_formal100_console_scalars_step1_46_20260822.csv` | 2,958 | `e73a7320d9128222eb82a9a4a8afc35941638a4050d3a7beec662650b78cc626` |
| `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/ppo_formal100_events_step1_46_20260822.tfevents` | 133,122 | `8daa0b7aa15dbd3d8c12b286a5bdaae84f4c420f80579f2b159884f908bbe215` |
| `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/ppo_formal100_metrics_live_step1_46_20260822.log` | 309,980 | `ae633a12667cba3be32d7c47a667d84c77eb75404b06000a2ba718dae4b8de73` |
| `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/ppo_formal100_resource_discrete_samples_through_step46_20260822.csv` | 1,002 | `40138b698287e2767cf0495efcd8b9582bcc3e84e464934f1b390c4b69202624` |
| `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/ppo_formal100_success_eval_step1_46_20260822.png` | 55,345 | `0f2d6a124d9966229bcdb94cec7018f32d973bb7d2f5026caf7d1a4836603f4b` |
| `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/ppo_formal100_optimization_step1_46_20260822.png` | 111,579 | `cb6ee3dbcf681683e9b772902f13c5962f079a32e3acfb14a7d431fcf89ffc0d` |
| `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/ppo_formal100_timing_step1_46_20260822.png` | 71,036 | `46aa3966cc639322fc1ae790576d1c190cf9c81f17674bd52fd6da9be1aace14` |
| `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/ppo_formal100_resource_discrete_through_step46_20260822.png` | 97,093 | `d88be4365fd1cfaebb562636e3559c8916eb83e18b7bd4d631d4ff89adcc72c1` |

## 6. 打包验收口径

- 对文本、脚本、CSV、metrics log 与 TensorBoard event 做常见凭据模式扫描；不得命中私钥、GitHub token、
  已知服务器密码或 Authorization header。
- ZIP 必须包含本 README 和上表 13 个文件，且没有额外文件。
- 使用 .NET ZIP reader 打开 archive，并把每个 entry 的内容流完整读到 EOF；同时核对 entry 数、非空性、
  展开总字节数和上表 13 个原件哈希。
- ZIP 自身的 bytes 与 SHA-256 在生成后单独报告；不把 ZIP 自身哈希写进 ZIP，以避免递归自校验问题。
