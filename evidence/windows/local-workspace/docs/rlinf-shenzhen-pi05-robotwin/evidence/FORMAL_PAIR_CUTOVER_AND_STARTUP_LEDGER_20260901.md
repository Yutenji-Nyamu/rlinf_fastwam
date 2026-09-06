# π0.5 GRPO / GRPO-DVAC 正式双实验切换与启动流水

## 结论

2026-09-01 12:35--12:46 CST，旧 π0 PPO Control 已按用户授权精确停止；旧 PPO-DVAC 在切换前已因 Ray 节点内存越过 95% 阈值退出。随后在同一 persistent Ray 上 fresh 启动：

- GPU 4/5：π0.5 GRPO Control formal-100；
- GPU 6/7：π0.5 GRPO-DVAC Action-Adv `[0.5,1.5]` formal-100。

12:46 CST 两个 wrapper 均存活，`RLinf` / `RLinf_1` namespace 各 15 actors；两边均完成首个 rollout epoch `1/4`，fatal/OOM=0。GPU 4--7 显存约 `34--41 GiB/card`，主机 `MemAvailable` 约 `1.71 TiB`。

## 配置合同

两边共同：两卡、`64 train / 32 eval`、rollout4、G8、256 trajectories、最多1,024 query records、`GB512/MB32/update5`、10次 optimizer call/outer、`H=C=50,D=14,M5`、LR `5e-6`、fixed32/eval5/save10、local-shard、100步、同模型与seed。

Control 与方法版的科学差异仅5项：

1. `chunk_level -> action_level`；
2. DVAC `off -> apply`；
3. `logprob_st -> action_advantage`；
4. `weight_min: null -> 0.5`；
5. `weight_max: null -> 1.5`。

其余 resolved 差异仅为 GPU placement、方法命名与 run-scoped 绝对输出路径；逐叶审计 `unexpected=[]`。

## 并发训练处理

- 保留一套 shared Ray head；不全局重启。
- 两个任务使用独立 namespace / Ray job / placement / 绝对数据与视频目录。
- 两边代码完全相同，所以可以共用同一 clean worktree；`RLINF_CODE_WORKING_DIR` 显式锁到该 exact HEAD。
- checkpoint 明确使用已验证的 `local_shard`，避开此前多任务并发下默认 DCP 首次保存协调卡住的问题。
- 旧任务仅按 owned PGID 与 exact namespace 清理；未触碰其他用户、shared Ray 或非目标 GPU。

## 精确来源与路径

- Branch：`codex/sz-pi05-robotwin-rl`
- HEAD：`256eeeb4459b4bd5db85bfc6a0eb315771e8c38c`
- Worktree：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-robotwin-rl`
- Control run：`/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-control-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys45-localshard-v1`
- DVAC run：`/data/chenyiteng/results/rlinf-shenzhen/pi05/runs/pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys67-localshard-v1`

## 旧 PPO 收尾

- Control：用户授权停止于完整 Step60；train `89.84%`，fixed32 `29/32`。
- PPO-DVAC：完整到 Step58；Ray 在 `1915.31/2015.51 GB=95.0282%` 时主动杀 worker，exit255；这不是算法数值或实现故障。
- 轻量终态包：`exports/shenzhen_pi0_ppo_control_dvac_w0p5to1p5_stopped_pair_light_evidence_20260901.zip`。

## 一个外层脚本细节

切换脚本最后一次 GPU readiness 检查早于 CUDA 完成分配，因而外层返回了 false-negative；两个 detached wrapper 与各15个actor未受影响。随后两次只读刷新已确认真实GPU占用和rollout推进。以后启动验收以 namespace 完整且日志进入 rollout 为准，不用“刚启动瞬间必须已有GPU进程”作为唯一判断。

## 13:05 CST 首个完整 step

- 两组均完整完成 Step 1 并进入 Step 2 rollout；wrapper、两个 namespace 与30个actors完整。
- Control：success `88.67%`，KL `0.116`，clip `0.173`，grad `5.131`，Step1 `25.55 min`。
- DVAC：success `83.59%`，KL `0.183`，clip `0.038`，grad `28.201`，Step1 `25.75 min`；Step1为预期warm-up，weight mean/sqmean/ESS均为1，Step2才开始非均匀加权。
- fatal/Traceback/CUDA或Ray OOM/worker death/Vulkan/nonfinite均为0；fixed评估从Step5开始、checkpoint从Step10开始，当前尚无二者符合配置。
- 本轮GPU峰值约`53.2 GiB/card`；即时约`31--32 GiB/card`。主机available约`1.58 TiB`，memory PSI=0。
- 依首步墙钟粗估100步约`42--43 h`；该值尚未包含后续fixed eval/checkpoint抖动。
