# RLT 单卡 matched-width 双实验启动

日期：2026-08-26  
状态：旧 pair 已停止并保留；新 control / DVAC-BC pair 已进入首个 rollout。

## 1. 本轮只改了什么

相对上一版单卡 pair，两边共同只改：

- `actor.micro_batch_size: 128 -> 256`；单卡梯度累积由4轮恢复为2轮，global batch仍为512；
- `warmup_min_size: 10000 -> 20000`；恢复历史双卡约20k全局transition后再开始online update；
- replay `cache_size/sample_window_size: 50000 -> 80000`；容纳fresh 480步理论最多约76.8k transition；
- 新的实验名和绝对输出路径。

方法版仍是`success_episode_bc`、`strength=0.25`、成功C10内mean-one DVAC重分配；actor-Q、critic TD、环境数、训练/评估预算、loss、LR与seed均未改。

## 2. 逐叶审计

- 新control与历史成功双卡resolved：14个差异，全部是上述4个训练字段、placement、fresh入口及路径；其他差异为0。
- 新control与method：23个差异，全部是GPU/run路径及method已有字段；其他差异为0。
- 新worktree与历史worktree的train/eval seed文件SHA256分别完全一致。

配置commit=`848b61278687702ea717c56b3734f1486cea3b95`，已push到`personal/codex/rlt-dvac-success-episode-bc`。

## 3. 运行入口

```text
control / GPU0
/root/autodl-tmp/experiments/rlt_single_gpu_control_matched_width_formal480_20260826_v3

DVAC-BC / GPU1
/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3

shared Ray
172.17.0.9:52001
```

12:27 CST现场：两条wrapper均alive、compose均exit0，placement分别`[[0]]/[[1]]`，replay明确建立为80k，
两条均显示`Generating Rollout Epochs: 0/1`。cgroup约75.0 GB，GPU0/1约21.2/10.8 GiB（仍在首轮初始化），
OOM/OOM-kill为0。日志里的Curobo import traceback是历史已有的可选planner告警；本配置使用MPLib，未中止训练。

旧pair原样保留约1.6/1.8 GiB，两边最新完整checkpoint均为Step125。

逐命令见[实施流水账](evidence/RLT_SINGLE_GPU_MATCHED_WIDTH_FORMAL480_LEDGER_20260826.md)。

