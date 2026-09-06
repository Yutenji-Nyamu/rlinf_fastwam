# π0.5 GRPO matched-update 切换流水

## 授权与目标

- 2026-09-01 用户明确授权停止当前两条 `GB512/update5` π0.5 formal，并 fresh 启动对应的 `GB1024/update2` formal。
- 科学参数只允许两处共同变化：`actor.global_batch_size: 512 -> 1024`、`algorithm.update_epoch: 5 -> 2`。
- Control 与 DVAC 的模型、采样、G8、评估、保存和方法差异均保持；旧 run 原样保留，不覆盖、不删除。

## 现场与实施记录

- 13:24 UTC 只读刷新：旧 Control/DVAC wrapper 均存活，分别占物理 GPU 4/5 与 6/7；日志继续更新，无 fatal/OOM。旧配置在进程命令中核实为 `GB512/update5`。
- 第一次执行在停止旧任务前的 resolved diff 审计退出：新 run 路径会连带改变 `algorithm.dvac_gradient_weighting.output_dir`；这是路径隔离叶，不是科学参数。旧训练未收到信号并继续运行。窄修为把该路径加入允许差异，并只清理本轮生成且没有 `packet_complete.txt` 的两个新 packet 目录后重试。
- 第二次执行生成并验证了两份完整新 packet；停止旧 Control 后，Ray actors随driver自动从15降为0，脚本因只接受“恰有15个待清理”而退出。结果是旧 Control已停、GPU4/5空闲，旧DVAC未动。窄修为清理函数接受0或15，并允许复用已完整packet、从明确断点继续。
- 待执行：先生成并核验两份新 resolved packet，再按 owned PGID 与 exact namespace 精确停止旧任务，保留 shared Ray 和其他用户任务；随后依次启动新 Control 和 DVAC。

## 结果

- 旧 Control 与旧 DVAC 均在完整 Step19 后停止；旧 run和Step10 checkpoint原样保留。Control第一次重试时已自动退出并清空namespace，DVAC随后按owned process tree和exact `RLinf_1`停止；shared Ray及其他用户任务未动。
- fresh新run：
  - GPU4/5：`pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2`；
  - GPU6/7：`pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys67-localshard-v2`。
- 21:36 CST 启动终检：两个wrapper存活，`RLinf` / `RLinf_1`各15 actors，均已进入首个`Generating Rollout Epochs 0/4`；GPU4--7均出现新任务显存，fatal/exit marker为0项。
- resolved共同合同为`64 train × rollout4 / G8 / max1024 records / GB1024/MB32/update2 / M5 / fixed32-eval5 / save10 / local_shard / fresh100`。Control为chunk-level且DVAC off；方法版为action-level Action-Adv `[0.5,1.5]`。旧到新逐叶审计除run路径外只有`actor.global_batch_size`与`algorithm.update_epoch`两项科学变化。
