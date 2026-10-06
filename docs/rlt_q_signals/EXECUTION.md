# 深圳2执行记录

本轮授权：GPU6 Q-U、GPU7 Q-Norm优先，原RLT候补最后接续；EXPO4/5及共享Ray保持。

正式配置锁定：历史π0.5 Clean4 `adjust_bottle`、fresh800、N4/U5、B512/micro256、C10/D14、200动作、10k初池/15k初始化、固定20每25轮评估保存。原完整Stage1复用，不重训。

smoke为独立输出/namespace和新回放，验证信号与Q更新/保存；smoke状态丢弃后正式从新池开始。正式运行至800轮或用户停止/不可恢复错误。OOM、非有限值、绑定越界、身份冲突或断点/回放不完整停止本次并保留证据；只清理本任务精确身份。

控制目录：`/data/chenyiteng/deployment-20261006/rlt-q-signals-g67-v1`。独立源码：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-q-signals-20261006`，分支`codex/rlt-q-signals-20261006`。

| 方法 | 卡与UUID | 正式输出（位于`/data/chenyiteng/results/rlinf-rlt/`） |
|---|---|---|
| Q-U | 6 / `GPU-dc5d6921-fa81-b666-bac7-566c126f1dd4` | `rlt4-q-u-g6-formal-800-1006-v1` |
| Q-Norm | 7 / `GPU-3c6321c1-3e58-c071-3867-533391152fe7` | `rlt4-q-norm-g7-formal-800-1006-v1` |

两个smoke仅把路径中`formal`换为`smoke`。smoke继承原短测：2轮、B32/micro16、U1、初池4、初始化更新2、CP2；本次使用原完整Stage1，Q系数0.05非零。进入正式前要求真实非均匀Q权重、有限loss、非零梯度、完整CP2与非空回放，以及namespace/进程/计算和图形释放。

启动入口（替换`LANE`为`u`或`norm`）：

```bash
CUDA_VISIBLE_DEVICES='' /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -u -B /data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-q-signals-20261006/tools/rlt_q_signals/runner.py --stage /data/chenyiteng/deployment-20261006/rlt-q-signals-g67-v1/LANE owner
```

先启动`coexist_owner.py`接管原EXPO CPU资源监视器，训练driver PID3503441/start410425820保持。原owner的结束流程会直接等待4–7全部释放，需接入本次Q优先级才能并行。新协调器将旧RLT保持候补；旧恢复配置缺逐卡渲染约束，待本次正式启动后再处理该后续验证。

已完成服务器CPU验证：37项loss/信号/回放/合同测试、1项生命周期测试、4套配置/绑定边界检查，Ruff检查通过。正式配置diff只含Q方法字段、信号生产开关、输出路径和SZ2逐卡放置。启动前源码SHA、权重SHA和配置pins将写入每组`plan.json`。

本地操作与轻量回执：`local_scripts/rlt_q_signals_20261006/{ops,evidence}`。不上传模型、回放或完整原始日志。

2026-10-07 00:41：源码`b1d2d8d553fa057435e0169846ec99c3a2c48c28`已推并核远端。原Stage1 CP2000共10018009814字节，源读取及SZ2落盘回读SHA均为`f2462366fd7822afde29fe245bdfaaadb37fc6cac39fef81b1e36d098bb2cad1`。实际teacher checkpoint与norm-stats SHA另进入新Q合同，历史resume元数据保留；历史`stage1_manifest_path`是遗留标识，不参与实际模型加载。

CPU协调器owner3008973/start410890278接替原owner18670，EXPO driver3503441/start410425820保持。两组smoke owner3110508/3110509、driver3110602/3110603已启动；00:45模型加载完成，6/7分别出现本组计算及C+G环境进程，0–3无上下文，尚无首次更新验收。

00:53两组smoke已通过，CP2均saved_runner_step=2/update_step=4，回放160条；两个namespace、精确进程树、计算/图形上下文均释放。首轮Q-U/Q-Norm权重std分别0.10437/0.09972，actor梯度8.7477/9.0887，weighted-Q 0.00639/-0.00835；BC的DVCA开关为0。smoke成功率0/4只用于接线验收，不评价方法收益。

独立正式启动：Q-U owner3110508/start410893910、driver50987/start410961868；Q-Norm owner3110509/start410893914、driver50956/start410961692。正式从原Stage1和空回放开始，smoke状态未续入；首轮实采核验待补。

01:00正式首轮已核：每组真实采集80条chunk，U/Norm均值0.014518/487.388；预采集update0符合原10k门槛。两个owner持续刷新，6/7仅本组C/C+G，0–3无上下文；EXPO原driver心跳12.6秒，仍4/5。数据盘余6.1TiB、root余32GiB。CP2在CPU实际重载160条回放，抽32条后信号[32,10]、Q权重非均匀，trainer/resume合同SHA匹配；未作GPU断点续训或沿用smoke状态。

启动回执、解析后配置和差异、smoke指标/释放、CP回读及正式首轮记录保存在发布副本`docs/rlt_q_signals/evidence/`。正式运行源码保持`b1d2d8d55`；后续文档提交不改变该运行checkout。
