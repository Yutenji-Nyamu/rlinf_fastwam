# GPU7：在线BC＋DVAC单卡smoke合同

用户2026-09-05授权独立实现、简要测试、单卡smoke。不是正式长训授权。源码基于当前BC生产cb01451f／证据385d4e75，独立`codex/sz-pi0-online-bc-dvac`，原BC/Sidney/shared Ray不改。

## 完整配置与精确命令

完整[smoke resolved](DVAC_SMOKE_RESOLVED_20260905.yaml)，与原正式BC逐叶差异[JSON](DVAC_FORMAL_BASE_TO_SMOKE_DIFF_20260905.json)。启动环境与精确入口为[wrapper](../../../local_scripts/bc_dvac_smoke_wrapper_20260905.sh)。

```bash
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc-dvac/examples/embodiment/train_embodied_agent.py \
  --config-name robotwin_adjust_bottle_online_bc_openpi +bc_dvac=default
```

输出：`/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-bc-dvac32x1-b1024-u10-gpu7-smoke2-20260905-v1`。拒绝覆盖现存run目录。连接既有Ray6389，显式独立code working dir，RLinf分配独立namespace。

## 预算与继承

| 项目 | 本次smoke |
|---|---|
| 卡／起点 | GPU7单H100 80GB；同adjust_bottle π0 SFT，空池，非BC正式第5轮或旧smoke权重 |
| 模型 | M4、C50、D14，三相机；只训练action expert及原相关投影；原生精度、增强off |
| 采样 | 每轮32并行×1串行，horizon200；2轮共64尝试、最多256新query／12800指令action槽 |
| 监督更新 | micro32/global1024，累积32，U10；总20次Adam、20480次chunk呈现；不是遍历成功池20遍 |
| 优化器 | 原LR2.5e-5／Adam(.9,.95)／eps1e-8／wd1e-10／clip1；constant、无warmup |
| 新方法 | 同次推理L3 endpoint方差；过去5轮log统计、首轮等权；α0.25/zclip2，权重均值1且[0,2]；入池固定，回放不重算 |
| 评估 | 暂按原通过的smoke：每轮16×2固定初态，两轮共64条；未获新选择前不擅自改评估预算 |
| 保存 | 每轮native checkpoint＋full weights＋原replay/learner＋新增DVAC标定sidecar，共2代 |
| 时间 | 参考原同预算smoke约36分钟，新增旁路应很小；硬上限90分钟／1.5 GPUh，不承诺耗时 |

与原正式BC不同：方法字段、GPU/独立源码及输出路径、100轮→2轮、eval5→1、save10→1、optimizer total1000→20。与原已通过BC smoke相比，只有方法字段、GPU及必要路径/名称不同；基线主配置与种子文件逐字节相同。不改奖励、成功过滤、均匀回放、示范混合0、FM噪声或时间分布、渲染依赖。

## 风险、资源与停止条件

18:21现场：GPU7 4MiB且无compute，RAM available1373GiB，/data余439.43GiB（现场具体bytes为准）；SSH/代理active，memory/io PSI0。18:23源锁clean。两代保存按既有BC文件估计约37GB十进制，另加成功池小量数据；不清理原产物。

重要新事实：原BC正式已于17:44:40在第6轮推理OOM，首次评估后EnvWorker占53.11GiB，整卡只余469MiB。**本次两轮smoke即使通过，也不能宣称已解决该长程资源边界。** 用户已被询问是否改用不创建评估环境的既有备选；此版本合同先保留已授权的原smoke预算，不擅自加入offload/缩训练并发/allocator设置。

每5秒记录本卡GPU及主机RAM/PSI、本人env FD/RSS。首个OOM、NaN/Inf、原生fatal、结构/shape错误、依赖/同步错误即停止本次验收，不自动调参反复重试；wrapper90分钟超时只约束本run。出现新外部GPU占用或空间不足则不启动。只允许处理本run进程，不动另一实验或共享服务。

验收：两轮采集/20次Adam、有限loss/grad；第1轮权重全1，第2轮有真实action-level信号和非平凡权重；成功archive/query对齐，两代原checkpoint实体与DVAC sidecar存在且可读，新标定状态恢复一致。该scope不包含生产worker完整重启恢复，也不凭短测成功率宣布方法提升。
