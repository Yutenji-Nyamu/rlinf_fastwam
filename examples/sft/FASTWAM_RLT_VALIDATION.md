# Fast-WAM 接 current RLT：实现与验证

2026-09-13。分支 `codex/sz-fastwam-rlt-20260913`。基于干净 current-AR RLT
`77d0a673`，迁入 Fast 接口 `0d5daf6f`；官方 Fast 源码保持 `7faa711` 原样。

## 改了什么

冻结 Fast-WAM，导出首帧 video expert 最后 post-block hidden，交给原来的
两层 RLT encoder＋两层 causal-AR decoder。使用同任务 clean50 原始三相机、
state、任务文字重新提取 Fast 特征；没有复用 π0 Stage1 权重。

Stage2 沿用原学生、Q、BC/Q 课程和回放。Fast 预测32步、学生模仿并提交前24步；
teacher/student/Q 都使用 Fast 原生 z-score 动作坐标，执行前原生反归一。
学生输出显式使用 `identity`，π0默认仍为 `tanh`。本分支不含 DVAC。

| 项目 | 配置 |
|---|---|
| Stage1方法 | 原完整AR重建；z2048、global32、2000更新、LR2.5e-5 |
| Stage2正式方法预算 | 8env、B512/micro256、UTD5、critic/actor=2、600轮 |
| 原课程 | 20k初始query池、30k初始化更新；20k＋50k的BC/Q课程 |
| 模型适配 | H50/C10→H32/C24；特征120×3072；学生identity |
| 整除适配 | episode/rollout预算200→192；query和物理动作量不再相同 |

## 已通过

- 服务器CPU **19/19**：新接口4项、原AR4项、RoboTwin路由/回放11项；
  正式与smoke配置解析通过。
- GPU6真实Fast前向确认 hidden为 **[1,120,3072]**。
  新出口每层K/V与官方prefill逐位一致，解码后的teacher物理动作逐位一致。
- 从clean50提取8帧，完整原AR模块训练2步，保存后恢复到第3步。
- 使用真实观测与teacher reference，调用原RLT损失，确认学生和Q均发生更新。
  此独立probe的bootstrap转移是合成的，只验证接口。
- 该单个query中，teacher动作标量 **20.54%** 超过归一域±1；
  说明identity接口确有使用场景，不代表全任务比例或成功率收益。

## 在线验证：已通过

3轮真实RoboTwin/Ray smoke正常退出，所有指标有限。
使用1env、B2/micro1、原critic/actor=2和新的3步Stage1产物。

| 检查 | 实测 |
|---|---|
| teacher→student路由 | actor_switch_rate依次0、1、1，真实完成切换 |
| 更新 | 每轮2次critic＋1次actor，合计6／3 |
| 回放 | 7→15→23个chunk转移 |
| 保存 | R3完成marker为true；update_step=6，rank状态文件存在 |
| 采集成功 | teacher轮1/1；两轮初始学生各0/1 |

这验证了环境、特征、动作解码、更新和保存链路，不能据三轮smoke判断方法性能。

当前Stage1产物标记 `smoke:true`，不是成熟压缩器；尚未完成全clean50的
2000步Stage1，也未进行600轮正式Fast RLT。运行说明见
[FASTWAM_RLT_README.md](FASTWAM_RLT_README.md)。

轻量证据保存在深圳
`/data/chenyiteng/results/server-maintenance-20260913/fastwam-bc-rlt/`：
`rlt/cpu-summary.json`、`gpu-smoke-v1/rlt-interface.json`、
`gpu-smoke-v1/rlt-restored/stage1-manifest.json`、`rlt/online-acceptance.json`；
随分支保留[轻量验证摘要](fastwam_rlt_validation.json)，大模型不纳入Git。
