# RLT-DVAC规划与AutoDL只读现场流水账

日期：2026-08-23  
范围：只读刷新当前训练、只读核对clean RLT source、本地形成实现计划。没有停止进程、修改服务器source、
启动RLT测试或训练，也没有push。

凭据只从当前任务会话读取并注入当前PowerShell子进程的`SEETA_SSH_PASSWORD`；helper结束后立即删除。
凭据值未进入脚本、文档、命令参数或输出。

## 1. 逐项记录

### RDP-001：21:21 CST 当前训练、资源与身份只读探针

命令：

```powershell
<bundled-python> local_scripts/remote_exec_autodl.py run `
  --command-file tmp/autodl_live_read_rlt_planning_20260823.sh
```

command-file SHA256：
`afdc0f8ce01caabf1d99a894f455c53ddfaa185e131b0e680aaa3bf9fcea24a9`。

结果：

- identity=`autodl-container-nekaqbwt43-6ce5babb /root uid=0`；
- 当前唯一相关训练为global-z `[0,2]` GRPO，PGID=`820638`；
- 完整到Global Step 4，Step 5 rollout当时为6/16；
- 两张A800约25.3/24.4 GiB；六个核心worker alive；
- cgroup约157 GiB/240 GiB，`oom=0, oom_kill=0`；
- `/root/autodl-tmp`余669 GiB；没有发现fatal。

### RDP-002：21:23 CST clean RLT source只读核对

命令：

```powershell
<bundled-python> local_scripts/remote_exec_autodl.py run `
  --command-file tmp/autodl_rlt_source_read_20260823.sh
```

command-file SHA256：
`d42d1b20e876c9da92afefce721a7e45d0dbf9f23ea815232a62ffab61d19c6c`。

结果：

- source=`/root/autodl-tmp/RLinf_rlt_pi0_robotwin`；
- branch=`codex/rlt-pi0-robotwin`，HEAD=`2b8199d8ab2e7b110994fd3234bf7007196c3af9`；
- worktree clean，跟踪`personal/codex/rlt-pi0-robotwin`；
- 成功formal配置`robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250.yaml`存在；
- source中没有既有DVAC符号，因此后续使用单独opt-in实现，不复用当前GRPO进程的worktree。

### RDP-003：clean server文件小范围SFTP副本

通过同一Paramiko helper的`get`子命令只读下载到
`tmp/rlt_source_2b8199d8/`：

```text
rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py
rlinf/models/embodiment/openpi/openpi_action_model.py
rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py
rlinf/algorithms/rlt/rollout.py
examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250.yaml
```

结果：五个文件均下载成功；只用作当前设计核对。现有本地`.rlt-impl-worktree`有用户未提交内容，后续不在其中
实施。

### RDP-004：本地精确调用链核对

命令族：

```powershell
rg -n "forward_actor|qf_pi|bc_loss|actor_loss|extract_rlt_obs|_sample_actions_with_prefix_cache|..." `
  tmp/rlt_source_2b8199d8/*
```

结果：

- `extract_rlt_obs()`在`torch.no_grad()`中运行冻结π0 reference ODE；内部H50，最后切C10×14；
- student输出flattened C10×14；
- `forward_actor()`先用student action计算Q，再用原action计算BC；
- 最窄挂点是在action进入default/CrossQ critic之前创建forward-identical的`pi_for_q`；
- 成功formal的`warmup_min_size=10000`、`warmup_post_collect_updates=30000`、8 train env、4×5 fixed eval；
- RLT已有独立trainer-state save/load，可扩展保存冻结baseline。

### RDP-005：21:32 CST 第二次当前训练只读刷新

重复执行RDP-001同一只读command-file。

结果：

- Global Step 4仍是最新完整step；Step 5 rollout推进到13/16；
- GPU0/1=`27084/26561 MiB`；相关核心worker均alive；
- `memory.current=178265047040 bytes`，即约166.02 GiB/240 GiB；
- `memory.events: oom=0, oom_kill=0`；host available约885 GiB；
- 当前不适合同时启动会加载Ray/模型/模拟器的RLT测试；独立source编辑和Git操作不会改变现有训练进程。

## 2. 问题与解决

1. 本地旧`.rlt-impl-worktree`不是当前clean authority，并有用户未提交内容。  
   解决：只读下载服务器clean `2b8199d8...`的精确文件，正式实现另开一个server worktree/branch。
2. GRPO使用recent-5 on-policy统计，而RLT会反复采样旧replay。  
   解决：首版在原有10,000-transition reference warmup结束时冻结global mean/std；replay保存raw V，恢复时
   同时恢复baseline。
3. 当前两张GPU和约166 GiB cgroup RAM正在被global-z formal使用。  
   解决：本轮只设计；实现阶段可在隔离worktree编辑，project tests等资源释放后再做。

## 3. 本轮文件变化

- 新建`28_RLT_DVAC_IMPLEMENTATION_AND_RECORDING_PLAN_20260823.md`：RLT首版默认、记录合同、代码清单和执行边界。
- 新建本文：逐项记录现场与source核对。
- 更新`26_DVAC_PORT_PLAN_FOR_RLT_AND_DSRL_20260823.md`：标记DSRL暂缓，并把RLT active路线指向28号文档。
- 更新`00_INDEX_AND_PLAN.md`与根`HANDOFF.md`：增加28号入口并刷新AutoDL当前状态。
