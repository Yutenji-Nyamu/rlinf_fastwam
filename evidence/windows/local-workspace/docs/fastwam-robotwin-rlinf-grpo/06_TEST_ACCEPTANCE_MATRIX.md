# 最小测试与验收矩阵

更新时间：2026-07-19

> 2026-07-18 更新：默认 P2 合并为一次 fresh `lr=1e-6` 真更新 smoke；`lr=0`仅在首次ratio异常时追加。一步只能证明初始同步，更新后同步由第二步/P3验证。当前命令见实施日志第8节。

测试服务于实现，不把流程拆成大量微型 gate。设计和来源冻结后，先连续完成一组相互依赖的主体代码；随后集中运行三组高信息量检查。只有发现会污染后续结果的实质错误时才停下修正，诊断测试按真实故障补充。

| 集中检查 | 覆盖范围 | 最低通过标准 |
|---|---|---|
| P1 离线来源与数值 parity | registry、adapter、builder/canonicalization、deterministic batch、Flow-SDE/replay | 一个组合 fixture 同时验证：B=1 与官方 deploy oracle 对齐；B=2 保持 left/right、样本置换与无 cache 串样；官方 schedule/shift/signed delta 与 RLinf OpenPI mean/std 对齐；同权重 old/new 逐元素一致；canonicalization 前后 action 一致且只有 canonical action expert 可训练。生产路径无 singleton fallback。真实 BF16 门限：official B1 max abs≤2e-3；同 batch shape 的 B2 permutation max/mean≤1e-3/1e-4；chunk2 对两次 B1 的跨 batch-shape 漂移 max/mean≤1e-2/1.5e-3 且 max/reference≤2% |
| P2 真实 RLinf smoke | driver/Worker 构建、FSDP2、192-step rollout、GRPO、bucket sync | fresh `lr=1e-6, noise_level=0.1`完成一个runner step；恰好8个完整24-action chunks；optimizer前ratio/clip/approx_kl可解释；无NaN/Inf/OOM；只有action expert有grad/update；产出step1 DCP与资源峰值。ratio异常时另跑lr0诊断，不用actor自产old掩盖；更新后同步留给第二步/P3 |
| P3 生命周期 | DCP fresh-process resume、pin原生no-dist model-state抽取、官方deploy export、π0最小回归 | resume恢复step/model/optimizer/scheduler/RNG并继续一步；exact schema通过；同一fixture下“恢复后的RLinf模型action”与“导出后官方loader模型action”一致；π0路径不变。非零更新模型不要求等于原base checkpoint |
| PPO-P1 静态/接口集中检查 | observation feature、head dtype/梯度、old/new/bootstrap、optimizer/sync/DCP/deploy schema、Hydra数量关系 | GRPO无head属性；PPO last-cache-V严格为`[B,120,3072]→[B,3072]`且不读action/k/noise；feature detach、head有梯度；actor/rollout head合同一致；BF16 head输出FP32 value；deploy忽略head而DCP保留。当前结果：Hydra resolve通过，配置合同7 passed，Fast-WAM集中测试61 passed |
| PPO-P2 真实两卡smoke | 官方base冷启动、FSDP2 actor-critic、192-step rollout、GAE/PPO、head同步、step1 DCP与资源 | smoke保持4路env并发，以`rollout_epoch1`产生32 transitions，`global32/update1`完成恰好一次optimizer update；old/new value与ratio有限可解释；action expert和value head分别有grad，冻结conditioner无grad；actor/rollout head hash同步；无NaN/OOM并产出step1 DCP。尚未执行，不能以PPO-P1替代 |

执行规则：

- 实施前的服务器 HEAD/status、standalone resolved `H/N/S`、checkpoint/stats 身份和`evidence/fastwam-golden/adjust_bottle-official-45d8e145`黄金fixture是一次短只读锁定，不单独扩展成长期gate流程。
- P1 可以由少量组合测试完成，不要求“一函数一测试”。固定 batch 分块时，共享 `k`、initial latent、SDE epsilon 在逻辑 batch 外一次生成；至少比较 unchunked 与实际采用的一个 chunk size，不穷举所有切分。
- 固定 eval singleton latent 保证同一随机输入、数学语义和 permutation 等价，不承诺 BF16 在不同 batch shape 下 bitwise 一致；不能因此强制生产逐样本 B=1 或改成 FP32。P2 的 rollout plain-BF16→actor FSDP-BF16 ratio/clip 仍须独立严格验收。
- 官方逐样本路径只作 B=1/小 batch oracle；生产只允许真实 batch core 或配置化固定分块，失败时保留证据并修正根因。
- P2 同时覆盖真实 rollout→FSDP actor 边界；不得用 actor 当前 logprob 替换 behavior old logprob 来制造 ratio≈1。
- PPO-P2必须在没有其他`train_embodied_agent.py`进程时单独执行。PPO从官方base fresh启动；不得用当前GRPO DCP暖启来掩盖head初始化、初始同步或dtype问题。
- 10-step固定seed学习趋势是P3完成后的尾项，用于观察学习、遗忘和资源，不是checkpoint/export正确性的组成部分。
- proprio 不属于第一版。以后启用时只增加一个针对性检查：actor 必须重跑 encoder，`grad/update` 非伪零，并明确 frozen video conditioner 的梯度边界。
- 训练图继续从 step 0/首个记录 step 到最新完整 step，并同时提供交互图和手机 PNG。
