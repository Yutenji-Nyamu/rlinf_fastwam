# RLT 4采样：温度1.5与成功倍率2

2026-09-16。用户授权先启动新实验，再归档旧组、继续机制分析。

## 实验合同

| GPU | 两层τ | success_scale | 其他参数 |
| --- | --- | --- | --- |
| 6 | 1.5 | 1 | 继承已完成的4采样τ1实配 |
| 7 | 1 | 2 | 继承已完成的4采样τ1实配 |

两组fresh600，每轮4条；B512/micro256、UTD5、critic:actor=2:1、10k预收集池、15k初始化更新、每轮更新上限800、BC/Q课程10k+25k、回放上限80k、actor/critic LR1e-4、原Stage1和原训练/评估种子、20条固定评估每25轮、每25轮保存不变。teacher的DVAC记录保持apply，不引入方向或成功倍率阶段调度。

以旧τ1的tensorboard/config.yaml原生解析配置为基准。GPU6方法仅temperature_local/chunk两叶1→1.5；GPU7方法仅success_scale一叶1→2。其他差异仅GPU放置、名称、源码/输出路径；训练/评估种子文件内容SHA一致。源码未修改，继承生产7c9372c6及已发布文档头68d901d2，无本轮新增CPU/GPU smoke。

## 路由与执行

- 新root：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-dvac-next-pair-half4-20260916`
- 新branch：`codex/sz-rlt-dvac-next-pair-half4-20260916`
- 唯一运维：`/data/chenyiteng/results/server-maintenance-20260916/rlt-next-pair`
- GPU6 run：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-exp-t15-half4env-fresh600-phys6-20260916-v1`
- GPU7 run：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-exp-t1s2-half4env-fresh600-phys7-20260916-v1`
- namespace：`RLinf_rlt_dvac_exp_t15_half4env_fresh600_phys6_20260916` / `RLinf_rlt_dvac_exp_t1s2_half4env_fresh600_phys7_20260916`

原τ0.5/τ1均自然完成600轮，exit0、原driver及namespace已退出。没有执行停止或清理。新两wrapper于11:14:56.573/11:14:56.750启动；两次启动间隔0.177秒。GPU0–5原进程身份在启动前后核对保持，未修改共享Ray。

命令为项目Python调用本轮`ops.py driver t15 formal`及`ops.py driver t1s2 formal`，完整argv和环境保存在各run/runtime。停止条件：600轮、用户明确停止或不可恢复运行错误；不按表现阈值自动停止。

11:20:39两组首轮验收通过：各12个ALIVE组件、配置差异为空、源码SHA一致、无所检fatal、首轮池80/update0/online0，GPU0–5原进程身份保持。GPU6已记录R3、GPU7已记录R1。driver为4146841/start241521562与4146854/start241521580（UID1003）。首次进入teacher预收集阶段时student尚未更新，不把首轮成功率作为方法效果。启动检查至此结束；发布结果见唯一publish回执。

## 解释

τ增大使成功样本内外的权重分配更平缓；GPU6检验τ1是否仍扰动偏大。GPU7保持τ1的相对分配，成功reference-BC权重整体乘2，失败保持1，检验成功模仿强度。它并非把Q或整批BC统一乘2。

旧两组完整指标、五张曲线、配置及检查点清单各自归档，模型与回放保留服务器。大产物在E盘`rlt-next-pair-20260916/closeout`。禁止重放本轮install、cutover或publish。
