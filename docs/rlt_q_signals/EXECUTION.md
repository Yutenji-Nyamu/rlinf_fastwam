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
