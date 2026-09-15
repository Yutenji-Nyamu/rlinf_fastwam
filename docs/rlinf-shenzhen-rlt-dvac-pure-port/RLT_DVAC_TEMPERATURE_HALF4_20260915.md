# RLT DVAC 温度配对执行记录

用户确认：基于当前GPU7正在运行的Clean4采样实配，仅方法变化；两组success_scale=1，τ分别0.5和1。用户已授权实现、简要测试、smoke、停止旧GPU6/7、正式启动及推送。

## 固定配置与实现

- 基线：/data/chenyiteng/results/rlinf-rlt/current-single-gpu-clean-half4env-fresh600-phys7-20260914-v1/tensorboard/config.yaml
- 新分支：codex/sz-rlt-dvac-temperature-half4-20260915；生产源码7c9372c6f7560ca65e6270d08cd4ae29a2a1ed0c。
- 4env×1、fresh600、B512/micro256、UTD5、critic:actor=2:1、初池10000、初始化15000、更新cap800、BC/Q课程10000+25000；模型、Stage1、LR、种子、评估固定20每25轮、保存25均继承。
- mapping=two_level_batch，factor_mapping=exp_mean；两层alpha1、成功倍率1，无4→2或方向课程；teacher的rlt_feature_model.openpi.rlt_dvac_mode=apply。
- 每层先沿原域MinMax，再exp(z/τ)除本域均值；内层每个成功query的C10，外层全B512中成功queries。失败W1；成功平均W1。τ小分配更集中，不是整体loss倍率。旧linear_centered默认及旧恢复合同不变。
- CPU：93 passed，7.67秒；仅新增26个温度case，余为现有相关回归。
- GPU smoke：先只停旧GPU6，在GPU6顺序验证两τ；每次固定同512条已有真实回放、生产FSDP/损失/Adam，4critic+2actor。smoke时钟35000只用于激活在线BC/Q，正式从0开始；smoke不包含新环境采集。

## 启动配置、命令、输出与停止条件

停止条件：600轮、用户明确停止或不可恢复运行错误；不设置效果阈值。GPU0–5及共享Ray保护；smoke期间GPU7旧组继续跑。

### t05: GPU6, 两层τ=0.5

- 输出：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-exp-t0p5-half4env-fresh600-phys6-20260915-v1`
- namespace：`RLinf_rlt_dvac_exp_t05_half4env_fresh600_phys6_20260915`
- 完整resolved：[t05 resolved](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/rlt-temperature-pair-20260915/verified/t05/prepared-formal/resolved.yaml)
- 与当前Clean4全叶差异：[t05 diff](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/rlt-temperature-pair-20260915/verified/t05/prepared-formal/baseline-diff.json)

```bash
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -B /data/chenyiteng/results/server-maintenance-20260915/rlt-temperature-pair/ops.py driver t05 formal --config-path /data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-exp-t0p5-half4env-fresh600-phys6-20260915-v1/runtime --config-name resolved hydra.run.dir=. hydra.output_subdir=null hydra.job.chdir=false hydra/job_logging=stdout
```

### t1: GPU7, 两层τ=1.0

- 输出：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-exp-t1p0-half4env-fresh600-phys7-20260915-v1`
- namespace：`RLinf_rlt_dvac_exp_t1_half4env_fresh600_phys7_20260915`
- 完整resolved：[t1 resolved](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/rlt-temperature-pair-20260915/verified/t1/prepared-formal/resolved.yaml)
- 与当前Clean4全叶差异：[t1 diff](E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/rlt-temperature-pair-20260915/verified/t1/prepared-formal/baseline-diff.json)

```bash
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -B /data/chenyiteng/results/server-maintenance-20260915/rlt-temperature-pair/ops.py driver t1 formal --config-path /data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-exp-t1p0-half4env-fresh600-phys7-20260915-v1/runtime --config-name resolved hydra.run.dir=. hydra.output_subdir=null hydra.job.chdir=false hydra/job_logging=stdout
```

## 执行回执

两种τ的CPU/GPU检查全部通过：93项CPU（7.67秒）；GPU6顺序双smoke合计约18.6秒，每组4critic+2actor真实Adam更新。固定512查询含253成功，成功权重ESS占比τ0.5为61.55%、τ1为87.81%，均重1；失败逐项1。该数字验证τ1分配更温和，不代表正式训练效果。

切换：GPU6旧组清空17:05:14.174829，双smoke通过17:05:32.818594，新τ0.5启动17:05:33.109395；GPU7旧组清空17:05:36.314516，新τ1启动17:05:36.498849。GPU0–5进程身份保持；共享Ray保持。清空至新wrapper分别18.93秒（含双smoke）/0.18秒，模型与环境初始化另计。

新driver分别396206/start234985216、409315/start234985555，uid1003；17:11:10两组启动验收通过，各12actors、实际resolved与预备配置差异为空，teacher采集开关apply、源码哈希和保护进程通过。两组已完成R2；R1均0/4、初始回放80条、update_step=0，仍为正常预收集阶段。恢复配置和Stage1沿用Clean4；正式是fresh Stage2，不加载旧两组检查点。首轮验收证明环境/teacher采集和入池链正常，温度实际更新由GPU smoke验证；当前正式组尚未进入learner更新，不能由早期采集差异推断τ效果。

独立源码核对：相对Clean4的SACworker差异仅全批方法hook与metrics；off时原batch原样返回。全叶配置审阅：Clean4实配313叶，新330叶；非方法/身份差异为空；配对归一身份后仅两个τ及GPU编号不同。

本地准备中尝试用PyYAML读取配置时发现未安装，未安装新依赖；转用服务器已有OmegaConf做配对全叶检查。该准备脚本问题发生在切换前，不涉及训练。所有项目CPU/GPU测试均在深圳执行。

精确回执位于唯一运维根`/data/chenyiteng/results/server-maintenance-20260915/rlt-temperature-pair`：共享`cutover-result.json`，两组各`t05/STARTUP_VERIFIED.json`、`t1/STARTUP_VERIFIED.json`及`smoke-result.json`、`stop-receipt.json`。旧组归档分别由`closeout/rlt_scale4to2`和`closeout/rlt_clean_half`收集、校验并发布回各自原分支；模型与回放保留。新实现和此处双组轻证据发布到同一个新分支，生产源码SHA以7c9372c6为准，后续文档提交不改运行代码。
