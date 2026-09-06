# GPU6：π0在线成功BC正式100轮执行合同

后续讨论路由：[最新问答与预算](BC_PARAMETER_QA_AND_UPDATE_BUDGET_20260905.md)。用户新提议eval16×2，U/global batch也待重新确认；下文旧GB1024/U2、eval32并行和offload建议均非已更新的可执行合同。本轮没有改生产/启动正式。

用户已明确授权“放正式训练”，但**截至13:39本合同尚不可执行，正式没有启动**。v6在13:20:25首次评估相机buffer分配失败退出，显存峰值79.17/79.65GiB；完整容量smoke未通过。下文resolved/命令保留为上一版预算合同，train/eval仍不offload；不能直接重用launcher。下一步仅建议现成的train/eval阶段offload，保持并发/学习预算，待用户确认后更新resolved、源码锁和新smoke目录。详细结果与选择见[本轮结果§3](BC_RESULT_AND_NEXT_RESOURCE_CHOICE_20260905.md#3-下一步建议保持32并行环境按阶段释放)。

## 1. 起点与唯一变更

- 独立源码：`codex/sz-pi0-online-bc`，生产代码`72a926041867cbdbf2565ab66a14d742c59a0dad`；轻量证据提交不算模型代码改变。
- 原始模型：`/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50`。`resume_dir:null / ckpt_path:null`，空成功池、Adam全新初始化；不用smoke/probe权重或成功数据。
- 完整[resolved](GPU6_FORMAL_RESOLVED_20260905.yaml)由服务器compose和validate_cfg生成；[逐叶差异](BC_FORMAL_CONFIG_DIFF_20260905.txt)仅允许100轮、eval5、save10、optimizer total200、实验名和派生输出路径。视频仍关闭，视频目录变化不意味着开启录像。
- 相同：GPU6、单卡32并行×1串行、C50/M4/D14/H200、三相机、原动作/图像归一化、augmentation off、train_expert_only、native precision null、相同SFT wrap/local_shard、MB32/GB1024/U2、LR2.5e-5常数、demo_weight0。无DVAC/teacher/critic/GRPO目标，不升级依赖或改OIDN。

## 2. 精确命令与输出

完整环境变量和实际入口保存在[formal wrapper](../../../local_scripts/bc_gpu6_formal_wrapper_20260905.sh)。核心命令：

```bash
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc/examples/embodiment/train_embodied_agent.py \
  --config-name robotwin_adjust_bottle_online_bc_openpi \
  runner.max_epochs=100 runner.val_check_interval=5 runner.save_interval=10 \
  actor.optim.total_training_steps=200 \
  runner.logger.experiment_name=pi0-adjust-bottle-online-bc32x1-m4-formal100-gpu6
```

输出：`/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-online-bc32x1-m4-gpu6-formal100-20260905-v1`；独占创建，不覆盖。连接现有Ray6389，不重启shared Ray。GPU7继续预留。

## 3. 预算和资源

| 单位 | 正式预算 |
|---|---|
| 新训练尝试 | 32×100＝3200 episodes |
| 新监督query上界 | 每episode≤4，共≤12800；只保留成功 |
| 优化器 | 每轮2次，最多200次；micro32，累计32个micro为global1024 |
| 回放样本呈现 | 每轮2048，最多204800；不是新轨迹或全池200个epoch |
| 固定评估 | 每5轮32条，共640条；不入成功训练池 |
| 保存 | 每10轮，最多10代模型/Adam/scheduler/RNG/成功池/learner sidecar |
| 时限 | 100轮自然结束，外层48小时安全上限；不是性能ETA或48小时必跑满 |

v6的GPU6采样峰值79.17/79.65GiB，首次评估buffer分配失败；不能以v5的60.43GiB或actor-only 27.29GiB当完整容量依据。13:39服务器可用RAM约1.7TiB、/data余612GiB，正式启动前再刷新。长期CPU池最大纯RGB约8.24GiB，10代池副本纯RGB合计上界约45.3GiB；还需计模型/Adam及原archive，不能只报数据池体积。v6没有checkpoint，故实际完整保存大小仍待新smoke测量。BC进程RSS与整机RAM分别记录，RSS相加会重复共享页，不当独占PSS。

## 4. 启动检查、观测与停止边界

新资源安排确认后，先在新run核实：exit0、完整2轮、4次optimizer、2次fixed32、两代真实模型与learner/replay保存、无fatal/实际分配失败/NaN；不得只检查PyTorch OOM字符串。原生导出全键及所检action_out_proj/Adam读回probe已通过，但不声称完整生产worker重启恢复已验证。旧v6 exit255不可通过该门槛，禁止复用其名称/目录或从其权重开始正式。

正式记录train/eval成功率、FM/梯度/LR、累计episode/query、更新时间/同步/保存、5秒GPU6显存和整机MemAvailable/PSI。空成功池显式跳过更新，不算成功训练。

fatal/OOM/NaN或实际长时间无进展时先保留证据，只处理本run；不自动降并发/batch/更新量，不自动从smoke续训，不动其他用户、Sidney或shared Ray。若磁盘可用低于100GiB或内存明显施压共享任务，暂停本run并讨论，不删除旧产物凑容量。48小时超时仅终止该wrapper的进程组，故障后不自动反复重启。
