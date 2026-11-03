# 每层依据与GRPO开发经验的复用

## 高层目的与切口

在Dojo扔瓶子上得到可重评、可恢复、成本可接受的π0.5强化学习。用户改为单卡DSRL、另一卡RLT；不是继续双卡GRPO，也不为H100修复整套Isaac。

```
原Dojo环境 → 已验收的批量观测/RPC → 冻结Dojo π0.5
                                    ↑32维噪声控制器
真实下一观察/成功/终止 → 已有紧凑回放 → 已有SAC更新
```

| 层/问题 | 主依据 | 具体复用与本轮增量 |
|---|---|---|
| 原权重、tokenizer、三相机 | XPolicyLab π0.5原实现，固定10ab265；原Dojo checkpoint59999 | 同一转换后的813个参数；不换成Sidney权重，不重新做SFT |
| 动作/状态14维排列 | 原Dojo pack/unpack、ALOHA变换；RLinf PR1754、FluxVLA PR93交叉核对 | 左右臂/夹爪重排只做一次；12臂关节delta+2夹爪absolute；q01/q99保留 |
| 生成时间与步长 | 原Dojo π0.5 N10、H50/C50 | DSRL只改初始噪声来源，速度积分与输出反变换照旧 |
| 冻结边界 | 旧DSRL `freeze_vlm`、`fsdp_sac_policy_worker` | 冻结VLM、动作专家和投影；只优化小噪声策略与Q，不是GRPO更新动作专家 |
| 噪声空间 | 历史π0/π0.5 DSRL实配 | 32维Gaussian预热、随后tanh策略；在H50上重复。与原π0.5独立50×32噪声有差异，起点评估记录为DSRL起点 |
| SAC超参与优化次数 | 历史Clean DSRL固定提交ff64209；原DSRL官方代码 | batch256、actor1e-4/Q3e-4、10Q均值、γ.999、目标τ.005、UTD20；保持FP32目标Q影子，U关闭 |
| 奖励/终止接口 | Dojo成功定义 + 旧 `project_dsrl_trajectory` | 原生首次成功与结束标志真实对应；投影为成功0/否则−1；剔除结束后块；超时允许bootstrap。不能把仿真错误塞成正常失败 |
| 回放与供数 | 旧 `dsrl_transition` | 64×64主图、14维状态、32维噪声与next_obs；不存无用视频/大chain；容量25000、预热500 |
| 环境并发 | 原生多环境和GRPO原生RPC；用户最新小批采样要求 | 正式8局/轮、100轮，共800训练局。32仅用于接入短测；不把每轮多采样当作等价提速 |
| 初态/验证 | 原Dojo保存布局 + GRPO已验收切分 | 训练0/2、验证1，排重；DSRL不要求G8同初态，所以group1；8×4验证批显式轮换32布局，跨评估重复同一组；仍跳物理坏布局、不筛成功 |
| 渲染/取图 | GRPO R15真实动作对照 | 保留每动作渲染，只省无用CPU图像拷贝；此前“直接省渲染”未通过，不能再启用 |
| Isaac生命周期 | GRPO R19已通过自然换批 | 每波新进程，策略/回放保留；只有明确75重启。物理错误有限重试，其他错误退出接RLT |
| 续训 | 旧DSRL严格保存/恢复 + 同框架FSDPModelManager | 策略、Q、目标Q/FP32影子、优化器、温度、阶段和回放一并保存；补齐单卡offload下加载前搬回GPU的现成步骤；不能只存推理模型 |
| 卡位 | 已验收RLT `priority_pair_handoff` | 单卡预约、精确PID清理、released自动接回。无需停止4/5或共享Ray |
| 纯工程合批 | 历史 `microbatch_benchmark.py` | DSRL保持global256，micro256一次算完；关闭模型offload试常驻。RLT保持8局、global512/micro256；micro512小网络计时已通过，整体收益很小，未改健康训练 |

## GRPO过程里已经付过的成本，这次不重付

1. 同权重JAX/PyTorch动作对齐通过；新DSRL旧源码接入只复核同一保存观察与噪声的结果，不重新转换或再启动JAX。
2. 曾漏掉原生reward注册导致一步误成功，已修复；本轮直接复用注册与非空检查，绝不重写成功判断。
3. 物体编号不等于物体位置，组内初态曾受随机物理影响；本轮单样本DSRL取消G8要求，但保留物理有效性筛查。
4. 省略中间渲染改变了策略输入，优化已撤回；只保留验收通过的CPU读取优化。
5. 同Isaac进程多次reset造成显存增长和不稳定；保持换波次进程，不继续修底层。
6. GPU7两次旧Xid109；本轮新单卡实例又出现GPU崩溃转储，已按用户授权回Robotwin RLT。普通RLT能跑不等于Isaac可用。

## 公开参考的适用范围

- [DSRL官方π0代码](https://github.com/nakamotoo/dsrl_pi0)：算法设计；我们的已跑通实现是部署主依据。
- [RLinf DSRL](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html)：噪声空间SAC、−1/0奖励、温度与Q指标；不同平台的环境参数不整份抄。
- [RLinf PR1754](https://github.com/RLinf/RLinf/pull/1754)：仍Open，注册/动作排列/转换/评估接口可借；作者说明未做真实Isaac端到端测试，不能当RL验收。
- [FluxVLA Dojo SFT](https://github.com/FluxVLA/FluxVLA/pull/93)：核对输入输出契约，非在线RL方案。
- [RLT论文](https://arxiv.org/abs/2604.23073)：先适配RL token、再训练小actor-critic。Dojo没有现成可确认的阶段一权重，不能冒用Robotwin阶段一；7卡渲染门槛失败后本轮不继续该路径。

完整旧证据留08/09。本轮代码切口为 `tools/dojo_offpolicy/{integrate,prepare,owner,training_entry,model_contract}.py`，仿真直接复用 `tools/dojo_grpo/native_entry.py` 与 `native_supervisor.py`。
