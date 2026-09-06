# π0.5 两卡 GRPO pair：formal 切换前只读准备

刷新时间：2026-09-01 12:26 CST。服务器全程只读，未停止或启动进程。

## 1. 已验证且现场仍成立的输入

- 分支/远端：`codex/sz-pi05-robotwin-rl`。
- exact HEAD：`256eeeb4459b4bd5db85bfc6a0eb315771e8c38c`；服务器 worktree clean，
  `personal/codex/sz-pi05-robotwin-rl`同HEAD。
- worktree：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl`。
- 主配置：
  `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_pi05.yaml`，
  SHA256=`0c0d451363fa22aa7dd8881d72d742fb3f3b9e7e6d80bcde5706f610dbc7402c`。
- 环境：`/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin`。
- RoboTwin：`/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support`。
- π0.5 SFT：
  `/data/chenyiteng/models/rlinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed`，约8.0 GiB；
  3个safetensors、index与`physical-intelligence/robotwin/norm_stats.json`均在现场。
- shared Ray：`172.17.0.1:6389`；每项任务必须独立namespace、placement和绝对输出路径，同时显式设置
  `RLINF_CODE_WORKING_DIR`为上述worktree。

已通过smoke的resolved：

- clean：
  `/data/chenyiteng/results/rlinf-shenzhen/pi05/packets/pi05-grpo-smoke1-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2/resolved.yaml`。
- DVAC：
  `/data/chenyiteng/results/rlinf-shenzhen/pi05/packets/pi05-grpo-dvac-action-adv-w0p5to1p5-smoke2-2gpu64x4-g8-b512-u5-m5-phys23-localshard-v2/resolved.yaml`。
- 实际smoke续链启动器：本地
  `local_scripts/remote_commands/sz_pi05_continue_grpo_smokes_20260831.sh`；服务器生成的worker为
  `/data/chenyiteng/results/rlinf-shenzhen/pi05/smoke-chain-grpo-continuation-20260831-v2/worker.sh`。

## 2. formal pair 的共同合同

两项都应fresh到Step100，并共同使用：

- `64 train env × rollout4 = 256 trajectories/step`，G8，即32 groups；最多1024 query records；
- `GB512/MB32/update5`，每轮10次optimizer step；
- `H=C=50, D14, M5`，π0.5 checkpoint/config、actor LR=`5e-6`；
- fixed32/eval5、save10、`local_shard`；train/eval video均开启；
- 同一train/eval seed内容。π0与π0.5 worktree中的两份seed文件SHA逐一相同。

建议的尚未创建路径：

- clean packet/run：
  `/data/chenyiteng/results/rlinf-shenzhen/pi05/{packets,runs}/pi05-grpo-control-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys45-localshard-v1`。
- DVAC packet/run：
  `/data/chenyiteng/results/rlinf-shenzhen/pi05/{packets,runs}/pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys67-localshard-v1`。
- clean用physical GPU4/5并在旧namespace清净后首先获得`RLinf`；DVAC用GPU6/7并获得`RLinf_1`。

## 3. clean 与 DVAC 的全部正式差异

排除placement、命名和run-scoped路径后，只有5个方法叶不同：

| 叶 | clean | DVAC |
|---|---|---|
| `algorithm.logprob_type` | `chunk_level` | `action_level` |
| `dvac_gradient_weighting.mode` | `off` | `apply` |
| `.application` | `logprob_st` | `action_advantage` |
| `.weight_min` | `null` | `0.5` |
| `.weight_max` | `null` | `1.5` |

`selected_l=3 / warmup_steps=1 / window_steps=5 / save_step_tensors=true`在同一主配置内两边相同；clean只是
不应用。正式运行不再保留smoke中的`max_steps=1/2`差异，两边都为100。

非科学差异只有：physical placement `4,5`/`6,7`，两个experiment name，两个log path，以及train/eval
save/video路径和由log path插值得到的DVAC output path。

## 4. π0.5 clean GRPO 相对既有π0两卡clean GRPO

共同项：64/32 env、rollout4、256 trajectories、G8、actor-only GRPO、group filter、MB32、H/C/D、
fixed32/eval5/save10、offload，以及相同seed内容。

必要差异：

| 项 | π0.5 | π0 | 依据 |
|---|---:|---:|---|
| SFT/config | π0.5 / `pi05_aloha_robotwin` | π0 / `pi0_aloha_robotwin` | 模型身份 |
| denoise M | 5 | 4 | 官方模型合同 |
| noise level | 0.3 | 0.5 | 官方模型preset |
| actor LR | `5e-6` | `5.6e-6` | 各自官方/既有recipe |
| GB × update | `512 × 5` | `1024 × 2` | π0.5同模型PPO严格8→2卡缩放；每轮10次update；π0沿已跑通深圳壳每轮2次 |
| checkpoint | `local_shard` | 历史control为默认DCP | 当前双任务稳定保存路线；不属于算法差异 |

π0.5 preset还显式带`detach_critic_input=true`、value字段和`value_lr=1e-4`；clean GRPO关闭value head、
critic不参与，因此这些是模型preset继承项，不是额外GRPO方法改动。

## 5. 当前阻塞与资源边界

1. **尚无formal resolved packet和formal launcher**。现场只有smoke packet/worker；正式切换前必须先compose
   上述两份formal resolved，做leaf parity并保存command/contract，再启动。
2. 12:26 CST GPU4/5仍有π0 PPO Control；GPU6/7已空闲。shared Ray只有`RLinf`的15个named actor，
   `RLinf_1`已空。不能直接重用`RLinf`；切换时只清当前Control的owned PGID和精确namespace，不停shared Ray。
3. 停止当前Control后应等待其GPU job完全消失、RAM回收，再依次启动clean和DVAC；不要把π0.5 pair叠在旧Control上。
4. `/data`当前余约841 GiB。π0.5 smoke实测每个checkpoint约27 GiB；两条100步、save10约需
   `27 × 10 × 2 ≈ 540 GiB`，可以容纳但余量不宽。不要额外提高保存频率，也不需要删除历史数据。
5. π0.5 smoke峰值约59.3/61.9 GiB每卡；既定MB32和32 env/rank有余量，但不应在本次pair中单边扩并发。

