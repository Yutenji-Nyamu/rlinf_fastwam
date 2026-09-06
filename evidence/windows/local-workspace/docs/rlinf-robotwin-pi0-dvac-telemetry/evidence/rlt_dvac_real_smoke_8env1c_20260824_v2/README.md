# RLT teacher-DVAC 真实 smoke v2 轻量证据

日期：2026-08-24  
结论：**通过**。driver `exit=0`，训练、fixed-20评估、DVAC apply、分布式指标归约和完整
`global_step_1` checkpoint均自然完成。

## 1. 运行合同

- source：`513dbcb7f31ebb639afa5267dff218028d07187b`；
- base：历史成功RLT的2×A800、8 train env、C10、200 primitive steps、4 eval env×5 epochs；
- 方法：冻结π0 teacher产生H50的`V_L2/L3/L4`，选择L3前C10，以冻结global-z映射到`[0,2]`，只缩放
  `Q -> student action_h -> actor`反向贡献；BC、critic TD、reward、route和前向动作不改；
- smoke-only：`max_steps=1`、`val/save=1`、reference warmup缩至本轮真实进入apply所需的最小预算；
- 资源监视只记录，不控制运行。

## 2. 结果

- wall：`18:09:38--18:17:51 CST`，约8分13秒；runner step time `393.959 s`；
- train rollout：8 episodes，2成功，`success_once=25%`；
- fixed-20 eval：0/20。它是只有一个cycle、student几乎未训练的机制smoke，不用于判断最终效果；
- updates：8 critic、4 actor；replay 72 trajectories；
- actor/critic grad norm：`3.804/3.259`，loss均finite；
- frozen baseline：两rank完全一致，`count=1440`、mean=`-4.8754106`、std=`0.5060390`；
- run级方法指标：weight p05/median/mean/p95=`0.263/0.959/0.991/1.820`，ESS=`0.880`，
  top-20% mass=`0.298`；
- 8个低频NPZ共320个实际`[B,10]`权重：min/p05/median/mean/p95/max=
  `0/0.301/0.996/1.011/1.825/2`，per-query ESS均值=`0.887`，top-20% mass均值=`0.295`；
- 资源：GPU峰值`17,111/17,193 MiB`，cgroup峰值`57.18 GiB`，OOM/OOM-kill=`0/0`；退出后两卡归零。

## 3. checkpoint

服务器保存了完整的`global_step_1`，约54 MiB；本轻量目录只带两个5 KiB trainer-state和complete manifest，
不复制模型与replay正文。两rank均为`update_step=8`，保存的DVAC baseline完全相同，complete manifest中
`complete=true`且列出两rank文件SHA256。

## 4. v1失败与v2修复

v1已经完成rollout和8个DVAC trace，但两个actor rank根据本地reward/route是否存在而条件性添加telemetry key，
导致最终metric all-reduce tensor长度不同并触发NCCL watchdog。v2只把这些分组统计改为固定`sum/count` schema，
跨rank归约后再求mean；不改DVAC公式、RLT目标或实验参数。窄单测从4项增至5项并通过，v2随后自然完成。

## 5. 文件说明

- `driver.log`、`metrics.log`：完整driver与最终指标表；
- `resolved.yaml`、`exact_command_v1_reference.txt`、`launch.sh`、`run_foreground.sh`：运行合同；
- `resources.csv`、`resources_after.txt`、`analysis_summary.json`：资源与离线方法汇总；
- `actor_rank*_update_*.npz`：两rank update 0/2/4/6的低频DVAC trace；
- `checkpoint_rank_*.pt`、`rlt_trainer_state_complete.json`：轻量恢复状态与完整性manifest；
- `source_head.txt`、`started_at.txt`、`finished_at.txt`、`exit_code.txt`：生命周期。

完整逐命令记录见上级
[RLT_DVAC_REAL_SMOKE_LEDGER_20260824.md](../RLT_DVAC_REAL_SMOKE_LEDGER_20260824.md)。
