# π0.5 GRPO128：从指数DVAC切换到既有top20方法

2026-09-14。用户已授权中止GPU4/5指数DVAC，复用此前80/20实现，继承clean128 seed42的全部方法外配置，完成短smoke后从原模型正式训练，再归档旧组与发布。**旧指数已停止；设备选择修复后41项CPU检查通过，v2真实两卡smoke于16:04:35验收通过；独立fresh正式组16:04:36启动，16:08:18启动健康验收通过。当前准备发布源码与轻量证据，尚未取得推送回执。**

## 方法与源码

当前源码提交：`06d4e1c9bd8e2d89f9b6dcefb9d8c2bca16d8082`，其父提交为初次移植`e0590471a2b214b89f52d4496612996c38ffa7c6`。分支：`codex/sz-pi05-grpo-top20-half-seed42-20260914`。

从当前clean128 seed42 HEAD `b51f185295e06bf22fbf51e94de05a15e8fef4eb` 建立独立worktree，其生产源码为`76c2ac4c40dd211d6e6eaab84022e75a61ea5ed7`。迁入此前已发布的top20实现`c780df928bc6c93c79f4b73f7a22271ef2c7e713`，没有继承旧top20的256条/B1024预算。

初次六文件移植中，helper、actor、两份top20测试逐字复用旧发布源码；rollout合入clean既有26行seed42与评估RNG隔离；recipe仅更正两行预算注释。随后仅修复helper的collective设备选择，并给原helper测试增加4个后端回归用例；方法值保持：

| 方法项 | 本组配置 | 含义 |
|---|---|---|
| `selection_domain` | `batch` | 本次完整Adam batch跨两rank共同选点 |
| `top_fraction` | 0.2 | 原始V最高20%的有效动作位置 |
| `selected_l` | 3 | 末3次去噪endpoint预测方差 |
| `full_update_probability` | 0.1 | 每次Adam有10%概率恢复全量系数1 |
| `method_seed` | 20260911 | 独立counter随机流，保持原实现 |

普通更新所选动作系数5、其余0，经过原ST梯度入口，前向chunk概率比与PPO裁剪保持；一次旁路决定由rank0产生并广播，整次Adam所有rank/micro共用。方法RNG与rollout42/43分离，保存独立sidecar。旧DVAC mode为off，不组合指数、MinMax、recent统计或Prism。

## 正式与smoke协议

| 项目 | 正式 | 本次smoke |
|---|---:|---:|
| 物理GPU | 4、5 | 4、5 |
| 每轮采集 | 64并行×2串行＝128条 | 64×1＝64条 |
| G / global batch / micro | 8 / 512 / 32 | 8 / 256 / 32 |
| U与Adam次数/轮 | U2，2次 | U2，2次 |
| 轮次 | 200 | 1 |
| 固定评估 | 每5轮32条 | 关闭 |
| 保存 | 每10轮 | 第1轮 |
| 初始化 | 原模型fresh，resume=null | 独立fresh，不作正式初始化 |

两者均沿用LR5e-6、50动作chunk、10次去噪、noise0.5、历史环境、原奖励与过滤、rollout seed42/43、评估RNG隔离。正式逐叶合同仅变化top20方法与运行身份路径；train/eval seed文件SHA保持。

**smoke保留U2的具体原因**：原method_seed20260911的前两次独立抽签归一值约0.02949、0.861；p0.1下第1次全量旁路、第2次稀疏0/5。一次64条采集即可覆盖两种路径。不更改method_seed或p来强制抽签，不新增GPU恢复运行。此前准备讨论中的U1仅会覆盖全量旁路，已被当前U2合同替代；正式预算始终保持128/B512/U2。

## 已验证结果

初次服务器CPU检查37项通过；首次真实smoke暴露原helper仅判断`dist.get_backend() == "nccl"`，在实际通信组上选到了CPU，随后`dist.broadcast`报`RuntimeError: No backend type associated with device type cpu`。该判断不能覆盖`cuda:nccl`等复合后端名称；故障现场未单独记录后端名称字符串。FSDP通过`init_device_mesh`建立通信组；已成功的`all_gather_object`使用PyTorch按group实际注册设备选择的路径。修复让`_collective_device()`复用该`_get_object_coll_device()`，保持原两个int64广播、旁路抽签、计数及sidecar格式。

修复后服务器复验：top20 helper30项（新增NCCL、CUDA composite、CPU+CUDA composite、Gloo四项）、actor4项、既有seed7项，共**41 passed、1 deselected，14.04秒**。源码差异与SHA检查通过。正式配置与方法预算核对未变；v1失败记录保留，v2使用新的smoke输出路径，不重放旧组停止。新分支尚未记为已发布。

**v2 smoke通过**：真实GPU4/5完成64条采集、B256/micro32/U2的一轮训练和checkpoint；`actual_config_diff={}`。两rank各2条update记录，full_update依次为true/false；每次共同选择1770/8850个有效动作位置，即0.2，top权重均值1、平方均值5。轮平均full_update=0.5、最终非零系数比例=0.60000002，真实梯度范数14.8158。两个已保存sidecar均记录update_index2/full_updates1、对应rank0/1与world2、B256/micro32/U2/rollout_epoch1/G8/H50。

单轮`time/step=403.04秒`，其中采集360.35秒、actor训练9.01秒。56次约10秒间隔的资源采样覆盖15:55:02至16:04:21；GPU4/5采样峰值分别60103/60393 MiB（约58.69/58.98 GiB），这是离散采样峰值。smoke于16:04:35通过验收，包含启动与保存的整个流程约9分33秒；本轮只验证接线与实际训练执行。

**正式启动健康通过**：16:04:36从原模型fresh启动128条/B512/U2/200轮正式组；16:08:18回执确认driver存活、15个actor存活、已进入真实采集，`actual_config_diff={}`、`unexpected_config_diff={}`，rollout seed42/43核验通过，无fatal日志、无退出码。`protected_unchanged=true`，GPU0–3/6/7及受保护原进程保持。启动观察至此结束；正式训练仍在进行，尚无本组成绩结论。源码和证据发布另以推送回执确认。

停止范围仅旧指数组的已核driver/namespace；保护GPU0–3/6/7原进程、共享Ray及其他用户。smoke在1轮完成、45分钟超时或运行故障时结束；正式在200轮、明确用户停止或不可恢复运行故障时结束，不设置成绩阈值。

## 路径与证据

新root：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-grpo-top20-half-seed42-20260914`。

正式run：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-top20-half128-seed42-formal200-phys45-20260914-v1`。

正式namespace：`RLinf_top20_half128_seed42_formal45_20260914`。

当前v2 smoke run：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/pi05-grpo-top20-half128-seed42-smoke1-phys45-20260914-v2`；namespace为正式名追加`_smoke`。失败的v1输出保留于同路径末尾`-v1`。

唯一运维根：`/data/chenyiteng/results/server-maintenance-20260914/grpo-top20-cutover`。停止、dispatch与launch使用本轮唯一回执，不重放。

- [正式解析配置](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/prepared/grpo/prepared-formal/resolved.yaml)与[正式合同](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/prepared/grpo/prepared-formal/contract.json)
- [smoke合同](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/prepared/grpo/prepared-smoke/contract.json)与[smoke限定差异](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/prepared/grpo/prepared-smoke/smoke-only-diff.json)
- [CPU日志](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/prepared/grpo/cpu-tests.log)与[源码安装回执](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/prepared/grpo/code-receipt.json)
- [设备修复回执](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/prepared/grpo/repair-receipt.json)，记录修复的两个文件SHA、v1/v2路径及正式配置不变结论
- [smoke验收](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/prepared/grpo/smoke-result.json)、[smoke资源采样](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/prepared/grpo/smoke-resource-samples.json)与[正式启动健康回执](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/prepared/grpo/STARTUP_VERIFIED.json)
- [本地移植清单](C:/Users/86136/Documents/rl/local_scripts/grpo_top20_cutover_20260914/source-manifest.json)、[具体接线与验收字段](C:/Users/86136/Documents/rl/local_scripts/grpo_top20_cutover_20260914/README.md)、[既有top20实现说明](C:/Users/86136/Documents/rl/docs/rlinf-shenzhen-grpo-dvac-action-adv/GRPO_DVAC_TOP20_IMPLEMENTATION_20260911.md)

正式命令已写入[command.txt](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/cutover-top20-20260914/prepared/grpo/prepared-formal/command.txt)：由本轮`ops.py driver grpo formal`读取正式run下`runtime/resolved.yaml`，Hydra不切换工作目录、不另存配置；新源码目录作为进程cwd。
