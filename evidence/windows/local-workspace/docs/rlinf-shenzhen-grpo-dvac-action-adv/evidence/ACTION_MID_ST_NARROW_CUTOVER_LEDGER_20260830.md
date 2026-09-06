# Action-Adv `[0.5,1.5]` / ST-DVAC `[0.8,1.2]` 切换流水账

日期：2026-08-30；机器：SZ-H100；用户：`chenyiteng`。

## 1. 授权与目标

用户授权结束并封存当前 Action-Adv Fix `[0,2]` 与已退出的 ST-DVAC `[0.5,1.5]`，整理轻量证据包，
随后在 GPU4/5 与 GPU6/7 fresh 启动两条100-step formal：

- Action-Adv Fix `[0.5,1.5]`；
- ST-DVAC `[0.8,1.2]`。

除权重端点、物理卡、run 名和独立输出路径外，继承已验证两卡 Control：`64 env × 4 rollout epochs`、
256 trajectories/step、G8、max1024 records、GB1024/MB32/update2、fixed32/eval5、save10、100步。

## 2. 准备

1. 从既有已运行双formal脚本机械派生两份新command-file，并逐项改为新run名和端点：
   - `local_scripts/remote_commands/sz_prepare_action_fix_mid_st_narrow_dual_formal100_20260830.sh`
   - `local_scripts/remote_commands/sz_cutover_action_fix_st_half_to_action_mid_st_narrow_dual_formal100_20260830.sh`
2. 新切换脚本按现场事实处理：旧Action仍活，需精确停止其owned PGID与`RLinf` namespace；旧ST已因
   Ray 95%内存保护exit255、`RLinf_1`为空，因此不重复杀进程，只写封存标记。
3. 所有服务器命令继续走固定host-key、进程内密码的Paramiko command-file路线；不触碰GPU0--3、
   shared Ray或其他用户任务。

后续逐命令结果按执行顺序追加。

## 3. Resolved packet 与切换

1. 在服务器生成两份 fresh resolved packet，并逐叶与两卡 Control 比较；两边
   `unexpected_reference_diff=[]`。共同合同为 `64 env × 4 rollout epochs = 256 trajectories/step`、
   G8/32组、max1024 records、GB1024/MB32/update2、fixed32/eval5、save10、100步。
2. Action 相对 Control 的方法差异仅为 `logprob_type=action_level`、
   `application=action_advantage`、DVAC apply、`weight_min/max=0.5/1.5`及独立命名/路径。
3. ST 相对 Control 的方法差异仅为 DVAC apply、`weight_min/max=0.8/1.2`及独立命名/路径；
   `logprob_type=chunk_level`保持不变。
4. 精确停止旧 Action 的 owned PGID/job/`RLinf` namespace；旧 ST 已因 Ray 内存保护退出，未再次杀
   进程，只写封存标记。shared Ray、GPU0--3与其他用户任务均未触碰。
5. 切换脚本在15个 actor 刚注册而 CUDA context 尚未出现的瞬间检查 GPU job，因采样过早返回1；
   wrapper并未被停止。随后独立只读复核确认两条新任务均有15 actors及真实GPU进程。

## 4. 新实验健康启动

2026-08-30 11:18:37 CST只读现场：

- Action `[0.5,1.5]`：wrapper PID `1413907` alive、exit pending、fatal=0，首轮 rollout `1/4`完成；
- ST `[0.8,1.2]`：wrapper PID `1416016` alive、exit pending、fatal=0，首轮 rollout `1/4`完成；
- `RLinf`/`RLinf_1`各15个named actors；GPU4/5与6/7进程严格分离；
- GPU4--7当时约24--44 GiB/card；主机MemAvailable约1.77 TiB，无failed unit；
  `/`、`/home`、`/data`分别余226 GiB、1.4 TiB、1.6 TiB；GitHub/HF代理均HTTP 200。

2026-08-30 11:42 CST再次只读刷新：两条均已完整Step1并完成首轮optimizer闭环，fatal=0；Action已进入
Step2 rollout `1/4`，ST已进入Step2 rollout `0/4`。GPU4/5约42.9/38.9 GiB，GPU6/7约50.4/51.0 GiB，
host available约1.50 TiB；GPU0--3显存仅69/4/9/9 MiB、利用率均0%，为空闲。本次没有干预进程。

## 5. 旧实验轻量收尾包

- Action-Adv Fix `[0,2]`按授权停在完整Step81；末步/MA5/MA10训练成功率为
  `94.53%/93.36%/92.73%`，fixed32累计`480/512`。轻量包：
  `exports/shenzhen_grpo_dvac_action_adv_fix_w0to2_stopped_light_evidence_20260830.zip`。
- ST-DVAC `[0.5,1.5]`此前在完整Step53因Ray节点达到95.5891%内存阈值退出255；
  末步/MA5/MA10为`93.75%/93.83%/92.07%`，fixed32累计`294/320`。轻量包：
  `exports/shenzhen_grpo_dvac_st_w0p5to1p5_ray_memory_exit_step53_light_evidence_20260830.zip`。
- 两包分别390,384与362,125 bytes，均通过ZIP自检；包含driver/resource/resolved、TensorBoard、
  逐步CSV和三张图，不含checkpoint、视频、Ray全量日志或逐action tensor。
- 首次制包遇到ST资源CSV终止行缺少GPU列；仅让制图器忽略不完整observer尾行，删除本轮生成的
  不完整本地产物后重建。没有删除服务器原始数据或训练产物。

## 6. ST Step10 DCP 卡住与 local-shard v2

2026-08-30 15:28:43 CST，ST-DVAC `[0.8,1.2]`完成Step10 fixed32后进入默认DCP保存，随后约一小时
无推进：`global_step_10`仅12 KiB且0个checkpoint文件，GPU6/7 actor停在
`save_checkpoint/ep_poll`，无fatal/OOM/磁盘压力。Action同时已用默认DCP保存Step10并继续到Step14，
说明这不是shared Ray或整机故障，但也不代表DCP在该拓扑可靠。

历史Prism v1曾出现完全相同的首次Step10现象。已验证的处理并非修改PyTorch DCP内部，而是使用RLinf
已有的rank-local保存路径：commit `306ce2e98a06b6f439a1070d8942e20132e48d49`只给
`fsdp_model_manager.py`增加14行格式校验/透传；Prism v2显式配置`local_shard`后，Step10/20/30/40/50
五次保存均成功。本次复发原因是ST head `0e28ac6f...`不是该commit后代，manager和resolved均无
`checkpoint_format`，launch manifest还明确写了默认DCP。

本轮精准修复：

1. 从ST原head建立独立分支`codex/sz-st-dvac-local-shard`，只cherry-pick上述一文件补丁；新head
   `f2a543da87afb7d5e3aa030ef52e53df9a266b28`已普通push。
2. fresh v2只新增`actor.fsdp_config.checkpoint_format=local_shard`及新source/run路径；与旧v1逐叶
   对比`unexpected=[]`，`64×4/G8/max1024/B1024/MB32/update2/fixed32/eval5/save10/[0.8,1.2]`
   全部不变。
3. 旧ST Step10无shard，不可恢复；仅停止其owned PGID `1416016`并清理精确namespace `RLinf_1`。
   Action PID `1413907`、job `88000000`及15个actor在切换前后不变；shared Ray未停。
4. 新run为
   `/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v2`，
   wrapper PID `2655627`、job `ab000000`、GPU6/7；16:49 CST已完成模型装载并进入首个
   `Generating Rollout Epochs 0/4`，fatal=0。Action现场已完整Step14且fatal=0。

切换脚本第一次在named actors刚齐而CUDA job尚未出现时，因Bash command substitution中的`set -e`
语义让空job未中止；随后的独立只读探针没有沿用该结论，而是确认真实job、FSDP加载和rollout。脚本已补
显式`test -n`，避免复用时把“actor已注册”误写成“GPU训练已启动”。真正验证checkpoint绕行成功的下一
个运行证据将是Step10目录出现`checkpoint_rank_0.pt/1.pt`；本轮按用户习惯在健康启动后停止主动盯守。

2026-08-30 18:43 CST只读刷新：Action已完整Step18，ST local-shard v2已完整Step4；两边wrapper、各15个
named actors均存活，fatal=0。Action最近grad norm为有限值`14.789--47.919`，DVAC weight mean约1、
ESS约`.947--.951`；ST最近grad norm为`21.297--33.732`，warm-up后ESS约`.991--.992`。ST尚未到
Step10，因此没有checkpoint属于预期，不能提前宣称本次保存已闭环。GPU4/5约61--64 GiB，GPU6/7约
48--50 GiB；GPU0--3空闲。host available约0.99 TiB，`/`、`/home`、`/data`使用率为21%/39%/53%，
inode充足；failed unit=0，Mihomo active，代理GitHub/HF均HTTP200。其他用户无GPU任务；liwenbo约
2.5 GiB RAM，zhangwei约0.45 GiB，未作任何干预。
